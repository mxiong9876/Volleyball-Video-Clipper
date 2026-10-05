# Rally labeler

Single-page tool for hand-labeling rally boundaries. Writes
`data/labels/<youtube_id>.json` as `[{"start_sec": ..., "end_sec": ...}, ...]`
(sorted, 0.01 s precision), which `volley_pipeline.eval` reads.

Needs **Chrome or Edge** (File System Access API; Safari/Firefox don't have it).

## Run

```sh
open -a "Google Chrome" tools/labeler/index.html
```

If the folder picker doesn't appear or errors on `file://`, serve it from localhost instead:

```sh
python3 -m http.server 8765 -d tools/labeler   # then open http://localhost:8765
```

1. **Open data folder…** → select the repo's `data/` folder (the repo root also works) →
   allow Chrome to edit files.
2. Pick a video from the dropdown (every `data/raw/<id>/video.mp4`; the id is the folder name).
3. If `data/labels/<id>.json` exists it's loaded and the video jumps to the end of the last
   rally, so you can resume.
4. Every change autosaves. The top-right indicator shows `saved ✓`.

## Keys

| Key | Action |
| --- | --- |
| `S` | rally start at current time (press again to replace) |
| `E` | rally end (needs a pending start; rejects end ≤ start and overlaps) |
| `Z` | undo last change (marks, rally adds, deletes) |
| `←` / `→` | seek 2 s |
| `Shift` + `←` / `→` | seek 0.5 s |
| `1` / `2` | playback speed 1x / 2x |
| `Space` | play / pause |

Click a rally in the list to seek to its start; `×` deletes it.

If an existing label file is malformed, the page refuses to save over it. Fix or move the
file first.
