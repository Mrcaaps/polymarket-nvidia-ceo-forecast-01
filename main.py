#!/usr/bin/env python3
"""Polymarket NVIDIA CEO forecast — make.fun entry (polymarket-nvidia-ceo-forecast-01).

Forecasts the Yes price of every market of Polymarket's "Next CEO of NVIDIA?"
event at every minute of the 4 hours after each checkpoint, as five quantiles.

Model (v2, per market, from its own 72h minute-price history + the cross-market sum):

  Median: asymmetric bounce around a 120-minute median anchor.
    gap = x - median(last 120 min), clipped to +/-0.05.
    Up-gap (price jumped above its anchor): reverts down fast,
      med(h) = x - 0.6 * (1 - exp(-h/10)) * gap.
    Down-gap (price fell below its anchor): continues down mildly,
      med(h) = x + 0.5 * (1 - exp(-h/40)) * gap.
    (Diagnostic: up-gaps revert ~60%, down-gaps revert only ~18% and
    moderately continue; a symmetric reversion scored worse.)
  Spread: jump-robust per-market 1-step vol (6h MAD/IQR blend, as in v1),
    sigma = 1.1 * 2.5 * (0.6 * MAD_sigma + 0.4 * IQR_sigma), each component
    clamped to [0.0002, 0.02] before blending. Quantiles use a peaked shape:
    inner (25/75%) offsets x0.5, outer (5/95%) offsets x1.6, with a 0.2
    left skew (down-side wider), because the price is sticky most minutes
    but jumps hard occasionally — a Gaussian shape over-spreads the middle
    and under-spreads the tails.
  Sum: the 24 Yes prices should sum to ~1. After computing all medians,
    each minute's median vector is scaled (multiplicatively, pull 0.8) so
    its sum reverts from the checkpoint sum toward 0.95 with a 90-min
    time constant. (The sum drifts down over 4h; the proportional pull
    corrects a systematic median bias when the sum is far from 1.)

Practice loss (local scorer, 6 cases p00-p05): 0.272770
  p00 0.4376, p01 0.3127, p02 0.3635, p03 0.2856, p04 0.1725, p05 0.1268.
House baseline on the same practice: 0.363589. v1 (this entry): 0.320587.
Top-10 practice bar is ~0.245 (rank 1: 0.2407) — v2 closes about half the
gap from v1 but does not reach it; the remaining loss is dominated by the
median error on unpredictable jumps (oracle-median analysis: with a perfect
median our spread alone would score 0.059).

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
TAU_UP = 10.0           # up-gap reversion time constant (minutes)
R_DN = 0.5              # down-gap continuation strength
TAU_DN = 40.0           # down-gap time constant (minutes)
GAP_CAP = 0.05          # clip |gap| (regime changes must not fully revert)

# --- spread (robust vol + peaked/skewed shape) ---
WINDOW_MIN = 360        # last 6 hours of 1-minute changes
SCALE = 1.1 * 2.5       # multiplier on the robust 1-step vol
W_MAD = 0.6             # weight of MAD estimator in the blend
INNER = 0.5             # multiplier on the 25/75% offsets
OUTER = 1.6             # multiplier on the 5/95% offsets
SKEW = 0.2              # left skew: down-side wider, up-side narrower
ALPHA = 0.5             # spread grows as h ** ALPHA

# --- cross-market sum pull ---
SUM_PULL = 0.8          # how far toward the sum target (0 = off, 1 = full)
SUM_TARGET = 0.95       # sum the medians revert toward
TAU_SUM = 90.0          # sum reversion time constant (minutes)

# z multipliers per quantile index (outer/inner with skew)
MULT = [OUTER * (1 + SKEW), INNER * (1 + SKEW), 0.0,
        INNER * (1 - SKEW), OUTER * (1 - SKEW)]


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
        infos.append({"id": m["id"], "x": x, "gap": gap, "sigma": sigma})

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
            s = inf["sigma"] * (h ** ALPHA)
            rows.append([price6(med + Z[j] * MULT[j] * s) for j in range(5)])
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
