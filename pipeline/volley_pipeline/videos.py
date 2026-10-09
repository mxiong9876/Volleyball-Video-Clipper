"""The video list in docs/videos.md: which videos are held out from tuning.

docs/videos.md is a markdown table with an `ID` column and a `Split` column (`tune` or
`held-out`). Held-out videos are skipped by the detector CLI and excluded from eval scoring
unless explicitly included.
"""

import sys
from pathlib import Path

_HELD_OUT = {"held-out", "held out", "heldout"}


def default_videos_doc(data_subdir: str | Path) -> Path:
    """docs/videos.md for a repo-relative data dir such as data/labels or data/raw."""
    return Path(data_subdir).resolve().parent.parent / "docs" / "videos.md"


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def held_out_ids(path: str | Path) -> set[str]:
    """IDs whose Split column says held-out, from the first table in the file with ID + Split."""
    header: list[str] | None = None
    held_out: set[str] = set()
    for line in Path(path).read_text().splitlines():
        if not line.lstrip().startswith("|"):
            if header is not None:
                break  # end of the table
            continue
        cells = _cells(line)
        if header is None:
            lowered = [c.lower() for c in cells]
            if "id" in lowered and "split" in lowered:
                header = lowered
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue  # |---|---| separator row
        if len(cells) != len(header):
            raise ValueError(
                f"{path}: row has {len(cells)} cells, header has {len(header)}: {line}"
            )
        row = dict(zip(header, cells, strict=True))
        if row["split"].lower() in _HELD_OUT:
            held_out.add(row["id"])
    if header is None:
        raise ValueError(f"{path}: no table with ID and Split columns")
    return held_out


def load_held_out(
    videos_doc: str | Path | None, data_subdir: str | Path, include: bool
) -> set[str]:
    """Held-out ids for a CLI to skip.

    `include` disables skipping. An explicit `videos_doc` must exist; the default one
    (docs/videos.md next to `data_subdir`'s data dir) may be missing, meaning none held out.
    """
    if include:
        return set()
    if videos_doc is not None:
        return held_out_ids(videos_doc)
    default = default_videos_doc(data_subdir)
    if not default.exists():
        print(f"note: {default} not found; not skipping held-out videos", file=sys.stderr)
        return set()
    return held_out_ids(default)
