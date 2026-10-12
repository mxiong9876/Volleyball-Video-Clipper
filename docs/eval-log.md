# Eval log

One line per detector run, scored with `volley_pipeline.eval` against data/labels (held-out
excluded). Miss buckets for a run: `volley_pipeline.diagnose --pred data/predictions/<run>`.

| Date | Run | What changed | Videos (labels) | P / R / F1 @ ±1.5s |
|---|---|---|---|---|
| 2026-10-08 | whistle_v0 | Baseline: whistle-only detector (2–4 kHz narrow-band, untuned defaults), consecutive whistles paired into rallies | 3 tune (196) | 0.032 / 0.066 / 0.043 |
| 2026-10-12 | motion_v0 | Motion-only detector (untuned defaults): 5 fps 160x90 gray frame-difference energy, cuts (hist + lone spike) masked, 1 s smoothing, per-video p20..p95 scaling, hysteresis 0.5/0.3, 1 s bridge, 1–45 s. Per video F1: 3fME 0.012, cWpK 0.000, n580 0.108 | 3 tune (196) | 0.023 / 0.061 / 0.034 |
