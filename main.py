#!/usr/bin/env python3
"""Polymarket NVIDIA CEO forecast — make.fun entry (polymarket-nvidia-ceo-forecast-01).

Forecasts the Yes price of every market of Polymarket's "Next CEO of NVIDIA?"
event at every minute of the 4 hours after each checkpoint, as five quantiles.

Model (v3, per market, from its own 72h minute-price history + the cross-market sum):

  Median: asymmetric bounce around a 120-minute median anchor.
    gap = x - median(last 120 min), clipped to +/-0.05.
    Up-gap (price jumped above its anchor): reverts down,
      med(h) = x - 0.6 * (1 - exp(-h/40)) * gap.
    Down-gap (price fell below its anchor): continues down mildly,
      med(h) = x + 0.5 * (1 - exp(-h/80)) * gap.
    (v3: slower reversion than v2's 10/40-min taus — a jump that persists
    is more often a regime change than a spike; LOCO-validated.)
  Spread: jump-robust per-market 1-step vol (6h MAD/IQR blend, as in v1),
    sigma = 1.1 * 2.5 * (0.6 * MAD_sigma + 0.4 * IQR_sigma), each component
    clamped to [0.0002, 0.02] before blending. Quantiles use a peaked shape:
    inner (25/75%) offsets x0.5, outer (5/95%) offsets x1.6, with a 0.2
    left skew (down-side wider), because the price is sticky most minutes
    but jumps hard occasionally — a Gaussian shape over-spreads the middle
    and under-spreads the tails.
    ADAPTIVE SHAPE (v3): a market that jumped hard in the last 30 minutes
    gets a fatter-tailed shape (inner down to 0.35, outer up to 2.0, skew
    up to 0.4, alpha up to 0.55), because jumps cluster — after a big
    1-minute move the next-240-minute move is ~5x larger in median
    (diagnostic: median |dy240| 0.0415 vs 0.0085). The weight is
    w = clip((ratio - 8) / (60 - 8), 0, 1) where ratio = (max |1-min
    change| in last 30 min) / (robust 1-step vol). Quiet markets keep
    v2's shape exactly, so the widening costs nothing on quiet cases.
  Sum: the 24 Yes prices should sum to ~1. After computing all medians,
    each minute's median vector is scaled (multiplicatively, pull 0.95) so
    its sum reverts from the checkpoint sum toward 0.95 with a 120-min
    time constant. (The sum drifts down over 4h; the proportional pull
    corrects a systematic median bias when the sum is far from 1.)

Practice loss (local scorer, 6 cases p00-p05): 0.266052
  p00 0.4288, p01 0.2950, p02 0.3476, p03 0.2833, p04 0.1739, p05 0.1253.
  LOCO (leave-one-case-out): beats v2 on 5/6 cases (only p04 loses, by
  0.0014). v2: 0.272770. House baseline: 0.363589. v1: 0.320587.
  Top-10 practice bar is ~0.245 (rank 1: 0.2407) — v3 closes the gap from
  0.028 to 0.021 but does not reach it; the remaining loss is dominated by
  unpredictable jumps (oracle-median analysis: with a perfect median our
  spread alone would score 0.059).

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
        infos.append({"id": m["id"], "x": x, "gap": gap, "sigma": sigma,
                      "mult": mult, "alpha": a_alpha})

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
            rows.append([price6(med + Z[j] * inf["mult"][j] * s) for j in range(5)])
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
