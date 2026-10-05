import json

import pytest

from volley_pipeline import eval as rally_eval
from volley_pipeline.eval import evaluate, load_rallies, score_video

LABELS = [(10.0, 20.0), (40.0, 55.0), (80.0, 92.0)]


def counts(s):
    return (s.tp, s.fp, s.fn)


# --- score_video -------------------------------------------------------------


def test_perfect_match():
    s = score_video(LABELS, LABELS)
    assert counts(s) == (3, 0, 0)
    assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)


def test_order_of_inputs_does_not_matter():
    assert counts(score_video(list(reversed(LABELS)), LABELS)) == (3, 0, 0)


def test_one_missed_rally():
    s = score_video(LABELS[:2], LABELS)
    assert counts(s) == (2, 0, 1)
    assert s.precision == 1.0
    assert s.recall == pytest.approx(2 / 3)
    assert s.f1 == pytest.approx(0.8)


def test_one_extra_rally():
    s = score_video(LABELS + [(120.0, 130.0)], LABELS)
    assert counts(s) == (3, 1, 0)
    assert s.precision == pytest.approx(0.75)
    assert s.recall == 1.0
    assert s.f1 == pytest.approx(6 / 7)


@pytest.mark.parametrize(
    "pred, matched",
    [
        ((11.4, 20.0), True),  # start 1.4s late
        ((10.0, 18.6), True),  # end 1.4s early
        ((8.6, 21.4), True),  # both 1.4s off, opposite directions
        ((11.5, 21.5), True),  # exactly at tolerance counts
        ((11.6, 20.0), False),  # start 1.6s late
        ((10.0, 21.6), False),  # end 1.6s late
        ((8.4, 18.4), False),  # whole rally shifted 1.6s early
    ],
)
def test_tolerance_boundaries(pred, matched):
    s = score_video([pred], [(10.0, 20.0)])
    assert s.tp == int(matched)


def test_custom_tolerance():
    assert score_video([(12.5, 20.0)], [(10.0, 20.0)]).tp == 0
    assert score_video([(12.5, 20.0)], [(10.0, 20.0)], tolerance=3.0).tp == 1


def test_two_predictions_near_one_label():
    s = score_video([(10.2, 19.9), (10.8, 20.5)], [(10.0, 20.0)])
    assert counts(s) == (1, 1, 0)


def test_one_prediction_near_two_labels():
    s = score_video([(10.5, 20.5)], [(10.0, 20.0), (11.0, 21.0)])
    assert counts(s) == (1, 0, 1)


def test_matching_maximizes_true_positives():
    # P1 fits both labels, P2 fits only A. Pairing P1-A would leave P2 unmatched.
    labels = [(10.0, 20.0), (12.0, 22.0)]  # A, B
    preds = [(11.0, 21.0), (9.0, 19.0)]  # P1, P2
    assert counts(score_video(preds, labels)) == (2, 0, 0)


def test_no_predictions():
    s = score_video([], LABELS)
    assert counts(s) == (0, 0, 3)
    assert (s.precision, s.recall, s.f1) == (0.0, 0.0, 0.0)
    assert s.mean_start_offset is None and s.mean_abs_end_offset is None


# --- boundary offsets (pred - label, matched pairs only) ---------------------


def test_offsets_signed_and_absolute():
    # Starts consistently 0.5s late; ends +1.0, -1.0, 0.0 (signed mean 0, abs mean 2/3).
    preds = [(10.5, 21.0), (40.5, 54.0), (80.5, 92.0), (120.0, 130.0)]  # last one unmatched
    s = score_video(preds, LABELS)
    assert counts(s) == (3, 1, 0)
    assert s.mean_start_offset == pytest.approx(0.5)
    assert s.mean_abs_start_offset == pytest.approx(0.5)
    assert s.mean_end_offset == pytest.approx(0.0)
    assert s.mean_abs_end_offset == pytest.approx(2 / 3)


def test_offsets_early_predictions_are_negative():
    s = score_video([(9.0, 19.5)], [(10.0, 20.0)])
    assert s.mean_start_offset == pytest.approx(-1.0)
    assert s.mean_abs_start_offset == pytest.approx(1.0)
    assert s.mean_end_offset == pytest.approx(-0.5)


def test_no_labels():
    s = score_video([(1.0, 2.0)], [])
    assert counts(s) == (0, 1, 0)
    assert (s.precision, s.recall, s.f1) == (0.0, 0.0, 0.0)


# --- evaluate (per video + overall) ------------------------------------------


def test_evaluate_per_video_and_micro_averaged_overall():
    labels = {"a": LABELS, "b": [(5.0, 15.0)]}
    preds = {"a": LABELS[:2], "b": [(5.0, 15.0), (30.0, 40.0)]}
    r = evaluate(preds, labels)
    assert counts(r.per_video["a"]) == (2, 0, 1)
    assert counts(r.per_video["b"]) == (1, 1, 0)
    assert counts(r.overall) == (3, 1, 1)
    assert r.overall.precision == pytest.approx(0.75)
    assert r.overall.recall == pytest.approx(0.75)
    assert r.overall.f1 == pytest.approx(0.75)


def test_evaluate_overall_offsets_pool_all_matched_pairs():
    labels = {"a": [(10.0, 20.0), (40.0, 50.0), (70.0, 80.0)], "b": [(5.0, 15.0)]}
    preds = {
        "a": [(11.0, 20.0), (41.0, 50.0), (71.0, 80.0)],  # starts +1.0
        "b": [(4.0, 15.0)],  # start -1.0
    }
    r = evaluate(preds, labels)
    assert r.per_video["a"].mean_start_offset == pytest.approx(1.0)
    assert r.per_video["b"].mean_start_offset == pytest.approx(-1.0)
    assert r.overall.mean_start_offset == pytest.approx(0.5)  # (3*1.0 - 1.0) / 4
    assert r.overall.mean_abs_start_offset == pytest.approx(1.0)
    assert r.overall.mean_end_offset == pytest.approx(0.0)


def test_evaluate_labeled_video_without_predictions_counts_all_missed():
    r = evaluate({}, {"a": LABELS})
    assert counts(r.per_video["a"]) == (0, 0, 3)


def test_evaluate_ignores_unlabeled_videos():
    r = evaluate({"a": LABELS, "unlabeled": [(1.0, 2.0)]}, {"a": LABELS})
    assert set(r.per_video) == {"a"}
    assert counts(r.overall) == (3, 0, 0)


# --- load_rallies + CLI ------------------------------------------------------


def test_load_rallies_reads_labeler_json_sorted(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(
        json.dumps([{"start_sec": 40, "end_sec": 55.5}, {"start_sec": 10.25, "end_sec": 20}])
    )
    assert load_rallies(p) == [(10.25, 20.0), (40.0, 55.5)]


@pytest.mark.parametrize(
    "content",
    [
        '{"start_sec": 1, "end_sec": 2}',  # not a list
        '[{"start_sec": 1}]',  # missing end
        '[{"start_sec": 5, "end_sec": 3}]',  # end before start
        '[{"start_sec": 5, "end_sec": 5}]',  # zero-length rally
    ],
)
def test_load_rallies_rejects_malformed(tmp_path, content):
    p = tmp_path / "x.json"
    p.write_text(content)
    with pytest.raises(ValueError):
        load_rallies(p)


def test_cli_prints_per_video_and_overall(tmp_path, capsys):
    rallies = [{"start_sec": s, "end_sec": e} for s, e in LABELS]
    (tmp_path / "labels").mkdir()
    (tmp_path / "pred").mkdir()
    (tmp_path / "labels" / "vid1.json").write_text(json.dumps(rallies))
    late = [{"start_sec": s + 0.5, "end_sec": e + 0.5} for s, e in LABELS[:2]]
    (tmp_path / "pred" / "vid1.json").write_text(json.dumps(late))
    code = rally_eval.main(["--pred", str(tmp_path / "pred"), "--labels", str(tmp_path / "labels")])
    out = capsys.readouterr().out
    assert code == 0
    assert "vid1" in out and "overall" in out
    assert "0.667" in out  # recall 2/3, printed to 3 decimals
    assert "+0.50" in out  # mean start/end offset, signed, 2 decimals
