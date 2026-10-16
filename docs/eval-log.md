# Eval log

One line per detector run, scored with `volley_pipeline.eval` against data/labels (held-out
excluded). Miss buckets for a run: `volley_pipeline.diagnose --pred data/predictions/<run>`.

## Tuning protocol

Back-of-court rec games are the primary footage (docs/BLUEPRINT.md), so:

- Select parameters on the back-court tune videos only (currently cWpKiSr1_9w, n580Vm1Oq2E).
  Use leave-one-video-out: pick on one, score on the other, and report both.
- Accept a change only if both back-court videos improve (F1 at ±1.5 s).
- Broadcast (3fME6LHzu0o) is a reported secondary row in every eval, never used to choose
  parameters.
- Small grid on the cached motion signal, scale-free parameters only (quantiles, ratios, seconds;
  no absolute energy levels). Order: start/end shifts → bridge/smooth → hi/lo → min_rally.
- Report ±3 s alongside ±1.5 s. One line below per run.
- Held-out videos only for the final check; label more back-court videos before calling
  anything tuned.

## Runs

| Date | Run | What changed | Videos (labels) | P / R / F1 @ ±1.5s |
|---|---|---|---|---|
| 2026-10-08 | whistle_v0 | Baseline: whistle-only detector (2–4 kHz narrow-band, untuned defaults), consecutive whistles paired into rallies | 3 tune (196) | 0.032 / 0.066 / 0.043 |
| 2026-10-12 | motion_v0 | Motion-only detector (untuned defaults): 5 fps 160x90 gray frame-difference energy, cuts (hist + lone spike) masked, 1 s smoothing, per-video p20..p95 scaling, hysteresis 0.5/0.3, 1 s bridge, 1–45 s. Per video F1: 3fME 0.012, cWpK 0.000, n580 0.108 | 3 tune (196) | 0.023 / 0.061 / 0.034 |
| 2026-10-15 | motion_v1 | motion_v0 + occlusion masking (a priori, untuned): changed-pixel fraction > 4× local median (±10 s, floored at video median) and ≥ 15% of frame, ≤ 4 s, not at a cut, ±2 pairs masked. 0/6/1 occlusions (3fME/cWpK/n580). Per video F1: 3fME 0.012, cWpK 0.154, n580 0.109; ±3 s overall 0.096 / 0.255 / 0.139 | 3 tune (196) | 0.031 / 0.082 / 0.045 |
| 2026-10-15 | motion_v1b | motion_v1 with the occlusion rule replaced: largest connected changed blob (3×3 dilation ×2, changed pixels counted) ≥ 0.175 of frame, ≤ 4 s, not at a cut; no local-level ratio. Threshold leave-one-out = max rally-time blob (confirmed walker 6:22.8 excluded): cWpK 0.091, n580 0.175. **Out-of-sample F1 ±1.5 / ±3: A (0.091 → n580) 0.143 / 0.247 (257 occlusions); B (0.175 → cWpK) 0.148 / 0.148.** Final 0.175 = larger max (still flags 6:22.8). In-sample: 530/6/5 occlusions (3fME/cWpK/n580); per video F1 3fME 0.005, cWpK 0.148, n580 0.122; ±3 s overall 0.100 / 0.214 / 0.136. Not accepted (cWpK down vs v1 0.154) | 3 tune (196) | in-sample 0.036 / 0.077 / 0.049 |
