# Polymarket NVIDIA CEO Forecast

Forecasts the Yes price of every market of Polymarket's "Next CEO of NVIDIA?"
event at every minute of the 4 hours after each checkpoint, as five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95). Entry for make.fun competition
`polymarket-nvidia-ceo-forecast-01`.

## Model

Per market, from its own 72-hour minute-price history:

```
q_tau(h) = clamp(x + z_tau * sigma * sqrt(h), 0, 1)
```

where `x` is the price at the checkpoint, `z_tau` the standard-normal quantile,
and `sigma` a jump-robust per-market 1-minute volatility from the last 6 hours:

```
sigma = 2.5 * (0.6 * MAD_sigma + 0.4 * IQR_sigma)
MAD_sigma = 1.4826 * median(|change - median(change)|)
IQR_sigma = (Q3 - Q1) / 1.349
```

Both estimators are robust to the jumps these prices make (e.g. Colette Kress
0.3535 → 0.59 → 0.4585 in two minutes), which would inflate a plain standard
deviation and over-spread the forecast. The median is persistence (the price
stays where it is); no drift term — short-window drift hurt on practice.

## Practice results (local scorer, 6 cases p00–p05)

| Variant | Mean pinball loss |
|---|---|
| House baseline | 0.363589 |
| **This bot** | **0.320587** |

Per case: p00 0.612563, p01 0.348537, p02 0.413799, p03 0.283486,
p04 0.231135, p05 0.118528. Beats the baseline on 4 of 6 cases (p01, p03,
p04, p05); loses on p00 (a jump-heavy window) and p02 (marginally).

## Run

```sh
python3 bot/main.py   # reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR (else /output)
```

Stdlib only, deterministic (byte-identical across runs), ~0.06 s per case
(limit 8 s). See `dev/REPORT.md` for the full variant sweep.
