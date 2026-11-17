#!/usr/bin/env python3
"""Variant B -- per-hour quantile regression on log ratios, with shrinkage.

Method (see README.md): for each forecast hour h = 2..24, fit a ridge
regression of ln(viewers_h / viewers_1) across past launch comparables on
features computable from the case alone (log hour-1 print, log hour-1
channels, pre-release 72h log slope).  The fitted value is median-centred
with the in-sample residual median; the spread comes from the residual
dispersion, smoothed across neighbouring hours, then lightly shrunk toward
the house baseline spread.  The median is shrunk 15% toward persistence
(ratio 1).  Stdlib only, no network, deterministic.
"""
import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta

Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
TAU = [0.05, 0.25, 0.5, 0.75, 0.95]
HOURS = list(range(2, 25))
RIDGE_LAM = 1.0
SHRINK_MU = float(os.environ.get("VARIANT_B_SHRINK_MU", "0.15"))
SHRINK_SIG = float(os.environ.get("VARIANT_B_SHRINK_SIG", "0.10"))
# Extra median shrinkage (up to +0.20) when the target's log hour-1 print
# lies outside the training range -- guards against extrapolating the
# log-size effect to a launch bigger (or smaller) than any comparable.
ADAPT_MAX = 0.20


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def safe_log(x):
    return math.log(max(1.0, float(x)))


def clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def solve_linear(A, b):
    """Gauss-Jordan with partial pivoting for a small dense system."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            continue
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col] / pv
                for k in range(col, n + 1):
                    M[r][k] -= f * M[col][k]
    out = []
    for i in range(n):
        out.append(M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0)
    return out


def ridge_fit(X, y, lam):
    k = len(X[0])
    A = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for xi, yi in zip(X, y):
        for i in range(k):
            b[i] += xi[i] * yi
            for j in range(k):
                A[i][j] += xi[i] * xi[j]
    for i in range(1, k):  # do not penalise the intercept
        A[i][i] += lam
    return solve_linear(A, b)


def hourly_points_from_history(case):
    """Whole-hour means before the checkpoint: prefer history_hourly,
    else aggregate the (minute) history reads into hour buckets."""
    hh = case.get("history_hourly")
    if hh:
        pts = []
        for p in hh:
            try:
                pts.append((parse_ts(p["from"]), float(p["viewers"])))
            except (KeyError, TypeError, ValueError):
                continue
        if pts:
            return sorted(pts)
    hist = case.get("history") or []
    buckets = {}
    for p in hist:
        try:
            t = parse_ts(p["at"])
            v = float(p["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        key = t.replace(minute=0, second=0, microsecond=0)
        s, n = buckets.get(key, (0.0, 0))
        buckets[key] = (s + v, n + 1)
    return sorted((t, s / n) for t, (s, n) in buckets.items())


def pre_release_slope(points, day_start):
    """OLS slope (log viewers per day) over the 72h before day_start,
    using whole-hour points strictly before day_start."""
    xs, ys = [], []
    for t, v in points:
        if v > 0 and day_start - timedelta(hours=72) <= t < day_start:
            xs.append((t - day_start).total_seconds() / 86400.0)
            ys.append(math.log(v))
    if len(xs) < 12:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return clip(slope, -3.0, 3.0)


def features(log_p1, log_ch, slope):
    return [1.0, log_p1, log_ch, slope]


def target_features(case):
    known = (case.get("known") or [{}])[0]
    p1 = float(known.get("viewers") or 0) or 1.0
    if p1 < 1:
        p1 = 1.0
    ch = known.get("channels")
    log_ch = math.log(max(1.0, float(ch)) / 1000.0) if ch else 0.0
    log_ch = clip(log_ch, -4.0, 4.0)
    day_start = parse_ts(case["day_start"])
    slope = pre_release_slope(hourly_points_from_history(case), day_start)
    return p1, features(math.log(p1 / 1e5), log_ch, slope)


def comparable_rows(case):
    """One row per usable launch comparable: features + log ratios.
    The GTA VI category's own pre-release history is NOT a launch day:
    exclude it (by name and by missing hour-1 point)."""
    rows = []
    for c in case.get("comparables") or []:
        name = str(c.get("name") or "")
        if "Grand Theft Auto VI" in name:
            continue
        hourly = c.get("hourly") or []
        by_time = {}
        for p in hourly:
            try:
                by_time[parse_ts(p["from"])] = p
            except (KeyError, TypeError, ValueError):
                continue
        try:
            ds = parse_ts(c["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        h1 = by_time.get(ds)
        if not h1:
            continue
        try:
            p1 = float(h1["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if p1 < 1:
            continue
        pts = sorted((t, float(p["viewers"])) for t, p in by_time.items()
                     if isinstance(p.get("viewers"), (int, float)))
        slope = pre_release_slope(pts, ds)
        ch1 = h1.get("channels")
        log_ch = math.log(max(1.0, float(ch1)) / 1000.0) if ch1 else 0.0
        log_ch = clip(log_ch, -4.0, 4.0)
        ratios = {}
        for h in HOURS:
            p = by_time.get(ds + timedelta(hours=h - 1))
            if p and isinstance(p.get("viewers"), (int, float)) \
                    and float(p["viewers"]) > 0:
                ratios[h] = math.log(float(p["viewers"]) / p1)
        if len(ratios) < 12:
            continue
        rows.append({
            "x": features(math.log(p1 / 1e5), log_ch, slope),
            "log_p1": math.log(p1 / 1e5),
            "ratios": ratios,
        })
    return rows


def forecast_case(case):
    p1, xt = target_features(case)
    rows = comparable_rows(case)
    base_sig = {h: 0.15 * math.sqrt(h - 1) for h in HOURS}
    if len(rows) < 5:
        # Too few comparables: persistence median, baseline spread.
        out = []
        for h in HOURS:
            qs = [p1 * math.exp(Z[i] * base_sig[h]) for i in range(5)]
            out.append((h, qs))
        return p1, out
    log_p1s = [r["log_p1"] for r in rows]
    lo, hi = min(log_p1s), max(log_p1s)
    excess = 0.0
    if xt[1] < lo:
        excess = lo - xt[1]
    elif xt[1] > hi:
        excess = xt[1] - hi
    shrink_mu = clip(SHRINK_MU + ADAPT_MAX * min(1.0, excess), 0.0, 0.6)
    # Per-hour ridge fits; collect raw residual sds for neighbour smoothing.
    mu_reg, sd_raw = {}, {}
    for h in HOURS:
        data = [(r["x"], r["ratios"][h]) for r in rows if h in r["ratios"]]
        if len(data) < 5:
            # Thin hour: borrow the pooled median log ratio of this hour
            # and its neighbours instead of defaulting to persistence.
            pool = [r["ratios"][hh] for r in rows for hh in (h - 1, h, h + 1)
                    if hh in r["ratios"]]
            mu_reg[h] = statistics.median(pool) if pool else 0.0
            sd_raw[h] = clip(statistics.pstdev(pool) if len(pool) > 1
                             else base_sig[h], 0.08, 2.5)
            continue
        X = [d[0] for d in data]
        y = [d[1] for d in data]
        beta = ridge_fit(X, y, RIDGE_LAM)
        pred = sum(b * x for b, x in zip(beta, xt))
        res = [yi - sum(b * x for b, x in zip(beta, xi))
               for xi, yi in zip(X, y)]
        mu_reg[h] = pred + statistics.median(res)
        # Residual sd with a degrees-of-freedom correction (n >> k here,
        # so this is small) so the spread is not understated by the fit.
        sd = statistics.pstdev(res) if len(res) > 1 else 0.5
        dof = max(1, len(res) - len(beta))
        sd *= math.sqrt(len(res) / dof) if dof < len(res) else 1.0
        sd_raw[h] = clip(sd, 0.08, 2.5)
    # Smooth the spread across neighbouring hours (0.5 / 0.25 / 0.25).
    sd_smooth = {}
    for h in HOURS:
        vals = [(0.5, sd_raw[h])]
        if h - 1 in sd_raw:
            vals.append((0.25, sd_raw[h - 1]))
        if h + 1 in sd_raw:
            vals.append((0.25, sd_raw[h + 1]))
        w = sum(v[0] for v in vals)
        sd_smooth[h] = sum(v[0] * v[1] for v in vals) / w
    out = []
    for h in HOURS:
        mu = clip((1.0 - shrink_mu) * mu_reg[h], -3.0, 9.0)
        sig = (1.0 - SHRINK_SIG) * sd_smooth[h] + SHRINK_SIG * base_sig[h]
        sig = clip(sig, 0.05, 3.0)
        qs = [p1 * math.exp(mu + Z[i] * sig) for i in range(5)]
        qs = [clip(round(q, 2), 1.0, 1e9) for q in qs]
        for i in range(1, 5):  # enforce non-decreasing after rounding
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        out.append((h, qs))
    return p1, out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input/")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output/")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        cid = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        _, rows = forecast_case(case)
        obj = {"hours": [{"hour": h, "q": qs} for h, qs in rows]}
        text = json.dumps(obj, separators=(",", ":"))
        with open(os.path.join(out_dir, cid + ".out"), "w",
                  encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
