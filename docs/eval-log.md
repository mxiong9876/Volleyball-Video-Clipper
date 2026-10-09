# Eval log

One line per detector run, scored with `volley_pipeline.eval` against data/labels (held-out
excluded). Miss buckets for a run: `volley_pipeline.diagnose --pred data/predictions/<run>`.

| Date | Run | What changed | Videos (labels) | P / R / F1 @ ±1.5s |
|---|---|---|---|---|
| 2026-10-08 | whistle_v0 | Baseline: whistle-only detector (2–4 kHz narrow-band, untuned defaults), consecutive whistles paired into rallies | 3 tune (196) | 0.032 / 0.066 / 0.043 |
