#!/usr/bin/env python3
"""Polymarket NVIDIA CEO forecast — make.fun entry (polymarket-nvidia-ceo-forecast-01).

Forecasts the Yes price of every market of Polymarket's "Next CEO of NVIDIA?"
event at every minute of the 4 hours after each checkpoint, as five quantiles.

Model (v4, per market, from its own 72h minute-price history + the cross-market sum):

  Median: asymmetric bounce around a 120-minute median anchor (as in v3).
    gap = x - median(last 120 min), clipped to +/-0.05.
    Up-gap (price jumped above its anchor): reverts down,
      med(h) = x - 0.6 * (1 - exp(-h/40)) * gap.
    Down-gap (price fell below its anchor): continues down mildly,
      med(h) = x + 0.5 * (1 - exp(-h/80)) * gap.
  Spread: a 60/40 blend of two shapes, both centred on the median:
    (a) the v3 parametric shape — jump-robust per-market 1-step vol
        (6h MAD/IQR blend, sigma = 1.1 * 2.5 * (0.6 * MAD_sigma +
        0.4 * IQR_sigma), components clamped to [0.0002, 0.02] before
        blending), with a peaked/skewed quantile shape (inner 25/75%
        offsets x0.5, outer 5/95% offsets x1.6, 0.2 left skew) and a
        per-market adaptive widening for recently-jumpy markets
        (inner down to 0.35, outer up to 2.0, skew up to 0.4, alpha up
        to 0.55, weight from the 30-min jump/sigma ratio, lo=8, hi=60);
    (b) the EMPIRICAL h-step quantile shape (new in v4) — for each
        market, the quantiles of the realized h-step changes p[t+h]-p[t]
        over the last 600 minutes of its own history, split by jump
        state at t (jumpy if the biggest |1-min change| in the last
        30 min is >= 8x the market's robust 1-step vol, else quiet),
        evaluated at h = 1,2,3,5,8,12,20,30,45,60,90,120,180,240 and
        interpolated in log(h), centred on the empirical median so it
        contributes shape only, not location. The blend is 60%
        parametric + 40% empirical. Why it works: the full-72h empirical
        shape is far too wide (it contains dead regimes), but the last
        10h of history is the live regime — its realized quantiles carry
        the true fat tails and asymmetry the Gaussian-ish shape misses.
        The effect is smooth in (window, blend): every (win 240-720,
        b 0.3-0.5) beats v3, and win=600/b=0.4-0.5 are the best points
        (0.2495-0.2496).
  Sum: the 24 Yes prices should sum to ~1. After computing all medians,
    each minute's median vector is scaled (multiplicatively, pull 0.95) so
    its sum reverts from the checkpoint sum toward 0.95 with a 120-min
    time constant.

Practice loss (local scorer, 6 cases p00-p05): 0.249570
  p00 0.3801, p01 0.2877, p02 0.3218, p03 0.2713, p04 0.1642, p05 0.1234.
  Beats v3 (0.266052) on 6/6 cases. Strict LOCO (tune window/blend on 5
  cases, test on the 6th): wins 5/6 (only p02 loses when tuned without it).
  v3: 0.266052. v2: 0.272770. v1: 0.320587. House baseline: 0.363589.
  Top-10 practice bar is ~0.245 (rank 1: 0.2407) — v4 closes the gap from
  0.021 to 0.005 but does not reach it; the remaining loss is dominated by
  unpredictable jumps (oracle-median analysis: with a perfect median our
  spread alone would score 0.059).

Rejected in the v4 round (all in dev/REPORT.md): full logit space
(0.330-0.339, delta-method sigma blows up for small prices), logit median
alone (0.267, 2/6 wins), full-72h empirical quantiles (0.326-0.417, far
too wide), kNN analog median (0.34-0.52, analog futures are noise),
jump-arrest median (0.281-0.339, prices do NOT revert fully to the
anchor), logit_mix stacking on v4 (0.2508, no gain).

Stdlib only, deterministic (no randomness, no network, no wall-clock use).
Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR (else /output).
"""
import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

# --- median (asymmetric bounce) ---
ANCHOR_WIN = 120        # minutes for the anchor median
R_UP = 0.6              # up-gap reversion strength
TAU_UP = 40.0           # up-gap reversion time constant (minutes)
R_DN = 0.5              # down-gap continuation strength
TAU_DN = 80.0           # down-gap time constant (minutes)
GAP_CAP = 0.05          # clip |gap| (regime changes must not fully revert)

# --- spread (robust vol + peaked/skewed shape, adaptive) ---
WINDOW_MIN = 360        # last 6 hours of 1-minute changes
SCALE = 1.1 * 2.5       # multiplier on the robust 1-step vol
W_MAD = 0.6             # weight of MAD estimator in the blend
INNER = 0.5             # multiplier on the 25/75% offsets (quiet markets)
OUTER = 1.6             # multiplier on the 5/95% offsets (quiet markets)
SKEW = 0.2              # left skew: down-side wider, up-side narrower
ALPHA = 0.5             # spread grows as h ** ALPHA (quiet markets)
# adaptive shape: how the shape widens for recently-jumpy markets
ADAPT_WIN = 30          # minutes to look back for the biggest jump
ADAPT_LO = 8.0          # jump/sigma ratio where widening starts
ADAPT_HI = 60.0         # jump/sigma ratio where widening saturates
ADAPT_D_INNER = -0.15   # inner shift at full weight
ADAPT_D_OUTER = 0.4     # outer shift at full weight
ADAPT_D_SKEW = 0.2      # skew shift at full weight
ADAPT_D_ALPHA = 0.05    # alpha shift at full weight

# --- empirical h-step quantile shape (v4) ---
H_GRID = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240]
EMP_WIN = 600           # minutes of history used for the empirical tables
EMP_JUMP_RATIO = 8.0    # jump/sigma ratio that marks a "jumpy" state
EMP_BLEND = 0.4         # weight of the empirical offsets (0.6 parametric)
EMP_MIN_SAMPLES = 50    # below this, fall back to the pooled table

# --- cross-market sum pull ---
SUM_PULL = 0.95         # how far toward the sum target (0 = off, 1 = full)
SUM_TARGET = 0.95       # sum the medians revert toward
TAU_SUM = 120.0         # sum reversion time constant (minutes)


def price6(x):
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def quantile(sorted_vals, q):
    """Linear-interpolated quantile of a sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(n - 1, lo + 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def interp_logh(grid, vals, h):
    """Interpolate vals (aligned with grid) at h, linear in log(h)."""
    if h <= grid[0]:
        return vals[0]
    if h >= grid[-1]:
        return vals[-1]
    lh = math.log(h)
    for i in range(len(grid) - 1):
        if grid[i] <= h <= grid[i + 1]:
            l0, l1 = math.log(grid[i]), math.log(grid[i + 1])
            w = (lh - l0) / (l1 - l0) if l1 > l0 else 0.0
            return vals[i] * (1.0 - w) + vals[i + 1] * w
    return vals[-1]


def robust_sigma(prices):
    """Jump-robust 1-step volatility from the last WINDOW_MIN minutes.
    sigma = SCALE * (W_MAD * MAD_sigma + (1-W_MAD) * IQR_sigma), each
    component clamped to [0.0002, 0.02] BEFORE blending and scaling."""
    n = min(len(prices) - 1, WINDOW_MIN)
    if n < 10:
        return 0.0025 * SCALE / 2.5
    start = len(prices) - 1 - n
    changes = sorted(prices[i + 1] - prices[i] for i in range(start, len(prices) - 1))
    m = len(changes)
    med = changes[m // 2]
    devs = sorted(abs(c - med) for c in changes)
    mad = devs[m // 2]
    mad_sigma = min(0.02, max(0.0002, 1.4826 * mad))
    q1 = changes[m // 4]
    q3 = changes[(3 * m) // 4]
    iqr_sigma = min(0.02, max(0.0002, (q3 - q1) / 1.349))
    return SCALE * (W_MAD * mad_sigma + (1.0 - W_MAD) * iqr_sigma)


def emp_offsets(prices):
    """Empirical h-step quantile offsets for one market, jump-conditioned.

    From the last EMP_WIN minutes of history, for each h in H_GRID, the
    quantiles of p[t+h]-p[t] split by the jump state at t (jumpy if the
    biggest |1-min change| in the last 30 min is >= EMP_JUMP_RATIO times
    the market's robust 1-step vol). Returns [5 x len(H_GRID)] offsets
    vs the empirical median (shape only, no location). Falls back to the
    pooled table when a state's sample at some h is too small."""
    n = len(prices)
    sig1 = max(1e-6, robust_sigma(prices) / SCALE)
    t_min = max(30, n - EMP_WIN)
    ch = [abs(prices[i + 1] - prices[i]) for i in range(n - 1)]
    # jump state at each t in the window
    state = {}
    for t in range(t_min, n - 1):
        jmax = max(ch[t - 30:t]) if t >= 30 else 0.0
        state[t] = "jumpy" if jmax / sig1 >= EMP_JUMP_RATIO else "quiet"
    # checkpoint state
    jmax_now = max((abs(prices[k] - prices[k - 1])
                    for k in range(max(1, n - ADAPT_WIN), n)), default=0.0)
    cur = "jumpy" if jmax_now / sig1 >= EMP_JUMP_RATIO else "quiet"
    all_ts = sorted(state.keys())
    offs = [[], [], [], [], []]
    for h in H_GRID:
        ds = sorted(prices[t + h] - prices[t] for t in all_ts
                    if state[t] == cur and t + h < n)
        if len(ds) < EMP_MIN_SAMPLES:
            ds = sorted(prices[t + h] - prices[t] for t in all_ts if t + h < n)
        qs = [quantile(ds, q) for q in QUANTILES]
        for j in range(5):
            offs[j].append(qs[j] - qs[2])
    return offs


def forecast_case(case):
    horizon = case["horizon_minutes"]
    infos = []
    for m in case["markets"]:
        p = m["prices"]
        x = p[-1]
        anchor = median(p[-ANCHOR_WIN:]) if len(p) >= 5 else x
        gap = max(-GAP_CAP, min(GAP_CAP, x - anchor))
        sigma = robust_sigma(p)
        # adaptive shape: how jumpy has this market been in the last
        # ADAPT_WIN minutes, relative to its own robust 1-step vol?
        sig1 = max(1e-6, sigma / SCALE)
        jmax = 0.0
        for k in range(max(1, len(p) - ADAPT_WIN), len(p)):
            d = abs(p[k] - p[k - 1])
            if d > jmax:
                jmax = d
        ratio = jmax / sig1
        w = max(0.0, min(1.0, (ratio - ADAPT_LO) / (ADAPT_HI - ADAPT_LO)))
        a_inner = INNER + ADAPT_D_INNER * w
        a_outer = OUTER + ADAPT_D_OUTER * w
        a_skew = SKEW + ADAPT_D_SKEW * w
        a_alpha = ALPHA + ADAPT_D_ALPHA * w
        mult = [a_outer * (1 + a_skew), a_inner * (1 + a_skew), 0.0,
                a_inner * (1 - a_skew), a_outer * (1 - a_skew)]
        # empirical quantile offsets (v4), interpolated to every horizon
        offs = emp_offsets(p)
        emp_h = [[interp_logh(H_GRID, offs[j], h) for j in range(5)]
                 for h in range(1, horizon + 1)]
        infos.append({"id": m["id"], "x": x, "gap": gap, "sigma": sigma,
                      "mult": mult, "alpha": a_alpha, "emp_h": emp_h})

    rows_all = []
    for inf in infos:
        rows = []
        for h in range(1, horizon + 1):
            if inf["gap"] > 0:
                rho = R_UP * (1.0 - math.exp(-h / TAU_UP))
                med = inf["x"] - rho * inf["gap"]
            elif inf["gap"] < 0:
                rho = R_DN * (1.0 - math.exp(-h / TAU_DN))
                med = inf["x"] + rho * inf["gap"]
            else:
                med = inf["x"]
            s = inf["sigma"] * (h ** inf["alpha"])
            e = inf["emp_h"][h - 1]
            rows.append([price6(med + (1.0 - EMP_BLEND) * Z[j] * inf["mult"][j] * s
                                + EMP_BLEND * e[j]) for j in range(5)])
        rows_all.append(rows)

    # Cross-market sum pull: scale each minute's medians so their sum
    # reverts from the checkpoint sum toward SUM_TARGET.
    if SUM_PULL and len(infos) > 1:
        s_now = sum(i["x"] for i in infos)
        for h in range(horizon):
            s_med = sum(rows_all[i][h][2] for i in range(len(infos)))
            if s_med <= 0:
                continue
            rho_s = 1.0 - math.exp(-(h + 1) / TAU_SUM)
            st = s_now + (SUM_TARGET - s_now) * rho_s
            f = (st / s_med) ** SUM_PULL
            for i in range(len(infos)):
                row = rows_all[i][h]
                shift = row[2] * f - row[2]
                rows_all[i][h] = [price6(v + shift) for v in row]

    return {"forecasts": [{"id": infos[i]["id"], "q": rows_all[i]}
                          for i in range(len(infos))]}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.endswith(".in"):
            continue
        cid = fname[:-3]
        with open(os.path.join(INPUT_DIR, fname)) as f:
            case = json.load(f)
        if case["quantiles"] != QUANTILES:
            raise ValueError(f"{fname} asks for quantiles {case['quantiles']}")
        fc = forecast_case(case)
        with open(os.path.join(OUTPUT_DIR, f"{cid}.out"), "w") as f:
            f.write(json.dumps(fc) + "\n")


if __name__ == "__main__":
    main()
