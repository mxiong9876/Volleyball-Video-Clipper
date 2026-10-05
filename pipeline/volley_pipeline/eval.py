"""Rally detection eval: score predicted rallies against hand labels.

A prediction matches a label when both its start and end are within `tolerance` seconds of
the label's. Matching is one-to-one and maximizes the number of matches. Rallies are
`(start_sec, end_sec)` tuples; files are the labeler's JSON (`[{start_sec, end_sec}, ...]`).
Compare a directory of predictions to the labels with:

    uv run python -m volley_pipeline.eval --pred <dir> --labels ../data/labels
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean

Rally = tuple[float, float]

DEFAULT_TOLERANCE = 1.5
# Absorbs float noise so a boundary exactly at the tolerance (e.g. 21.6 - 20.1) still counts.
_EPS = 1e-9


@dataclass
class Scores:
    tp: int
    fp: int
    fn: int
    # pred - label for each matched pair (positive = prediction late)
    start_offsets: list[float] = field(default_factory=list)
    end_offsets: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        return _ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    @property
    def mean_start_offset(self) -> float | None:
        return _mean(self.start_offsets)

    @property
    def mean_end_offset(self) -> float | None:
        return _mean(self.end_offsets)

    @property
    def mean_abs_start_offset(self) -> float | None:
        return _mean([abs(x) for x in self.start_offsets])

    @property
    def mean_abs_end_offset(self) -> float | None:
        return _mean([abs(x) for x in self.end_offsets])


@dataclass
class Report:
    per_video: dict[str, Scores]
    overall: Scores


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def _mean(xs: list[float]) -> float | None:
    return fmean(xs) if xs else None


def load_rallies(path: str | Path) -> list[Rally]:
    """Read a labeler-format JSON file into sorted `(start_sec, end_sec)` tuples."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of {{start_sec, end_sec}}")
    rallies = []
    for i, item in enumerate(data):
        try:
            start, end = float(item["start_sec"]), float(item["end_sec"])
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError(f"{path}: entry {i} needs numeric start_sec and end_sec") from exc
        if not end > start:
            raise ValueError(f"{path}: entry {i} has end_sec <= start_sec")
        rallies.append((start, end))
    return sorted(rallies)


def match_rallies(
    predicted: Sequence[Rally], labels: Sequence[Rally], tolerance: float = DEFAULT_TOLERANCE
) -> list[tuple[int, int]]:
    """Maximum one-to-one matching; returns (pred index, label index) pairs."""
    tol = tolerance + _EPS
    candidates = [
        [j for j, (ls, le) in enumerate(labels) if abs(ps - ls) <= tol and abs(pe - le) <= tol]
        for ps, pe in predicted
    ]
    label_owner: dict[int, int] = {}  # label index -> pred index

    # Kuhn's augmenting paths: try to give pred i a label, re-routing earlier preds if needed.
    def assign(i: int, seen: set[int]) -> bool:
        for j in candidates[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in label_owner or assign(label_owner[j], seen):
                label_owner[j] = i
                return True
        return False

    for i in range(len(predicted)):
        assign(i, set())
    return sorted((i, j) for j, i in label_owner.items())


def score_video(
    predicted: Sequence[Rally], labels: Sequence[Rally], tolerance: float = DEFAULT_TOLERANCE
) -> Scores:
    pairs = match_rallies(predicted, labels, tolerance)
    return Scores(
        tp=len(pairs),
        fp=len(predicted) - len(pairs),
        fn=len(labels) - len(pairs),
        start_offsets=[predicted[i][0] - labels[j][0] for i, j in pairs],
        end_offsets=[predicted[i][1] - labels[j][1] for i, j in pairs],
    )


def evaluate(
    predicted: Mapping[str, Sequence[Rally]],
    labels: Mapping[str, Sequence[Rally]],
    tolerance: float = DEFAULT_TOLERANCE,
) -> Report:
    """Score every labeled video (missing predictions = none found); overall is micro-averaged."""
    per_video = {
        vid: score_video(predicted.get(vid, []), labels[vid], tolerance) for vid in sorted(labels)
    }
    scores = per_video.values()
    overall = Scores(
        tp=sum(s.tp for s in scores),
        fp=sum(s.fp for s in scores),
        fn=sum(s.fn for s in scores),
        start_offsets=[x for s in scores for x in s.start_offsets],
        end_offsets=[x for s in scores for x in s.end_offsets],
    )
    return Report(per_video=per_video, overall=overall)


def _offset(x: float | None, signed: bool = True) -> str:
    if x is None:
        return "-"
    return f"{x:+.2f}" if signed else f"{x:.2f}"


def format_report(report: Report) -> str:
    header = (
        f"{'video':<14}{'tp':>5}{'fp':>5}{'fn':>5}{'precision':>11}{'recall':>8}{'f1':>7}"
        f"{'start Δ':>9}{'|start Δ|':>11}{'end Δ':>8}{'|end Δ|':>9}"
    )
    rows = [header, "-" * len(header)]
    for name, s in [*report.per_video.items(), ("overall", report.overall)]:
        if name == "overall":
            rows.append("-" * len(header))
        rows.append(
            f"{name:<14}{s.tp:>5}{s.fp:>5}{s.fn:>5}{s.precision:>11.3f}{s.recall:>8.3f}"
            f"{s.f1:>7.3f}{_offset(s.mean_start_offset):>9}"
            f"{_offset(s.mean_abs_start_offset, signed=False):>11}{_offset(s.mean_end_offset):>8}"
            f"{_offset(s.mean_abs_end_offset, signed=False):>9}"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score predicted rallies against labels.")
    parser.add_argument("--pred", required=True, help="dir of predicted <youtube_id>.json")
    parser.add_argument("--labels", default="data/labels", help="dir of labeled <youtube_id>.json")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="seconds")
    args = parser.parse_args(argv)

    pred_dir, label_dir = Path(args.pred), Path(args.labels)
    try:
        labels = {p.stem: load_rallies(p) for p in sorted(label_dir.glob("*.json"))}
        if not labels:
            raise ValueError(f"no label files in {label_dir}")
        predicted = {
            vid: load_rallies(pred_dir / f"{vid}.json")
            for vid in labels
            if (pred_dir / f"{vid}.json").exists()
        }
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for vid in labels.keys() - predicted.keys():
        print(
            f"warning: no predictions for {vid}; counting all its rallies as missed",
            file=sys.stderr,
        )

    print(f"tolerance ±{args.tolerance:g}s; offsets are pred − label in seconds (+ = late)")
    print(format_report(evaluate(predicted, labels, args.tolerance)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
