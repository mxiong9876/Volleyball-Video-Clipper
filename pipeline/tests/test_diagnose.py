import json

import pytest

from volley_pipeline import diagnose


def test_cli_smoke(tmp_path, capsys):
    labels = [(10.0, 20.0), (40.0, 50.0), (70.0, 80.0), (100.0, 110.0)]
    # rally 1 matched, rally 2 off by the serve delay, rally 3 never predicted
    preds = [(10.2, 20.3), (35.0, 51.0), (200.0, 205.0)]
    whistles = [10.0, 21.0, 35.0, 51.0, 65.0, 200.0, 205.0]
    for d in ("labels", "pred"):
        (tmp_path / d).mkdir()
    rows = lambda rs: json.dumps([{"start_sec": s, "end_sec": e} for s, e in rs])  # noqa: E731
    (tmp_path / "labels" / "vid1.json").write_text(rows(labels))
    (tmp_path / "pred" / "vid1.json").write_text(rows(preds))
    (tmp_path / "pred" / "vid1.whistles.json").write_text(
        json.dumps([{"start_sec": t, "end_sec": t + 0.5} for t in whistles])
    )
    code = diagnose.main(["--pred", str(tmp_path / "pred"), "--labels", str(tmp_path / "labels")])
    out = capsys.readouterr().out
    assert code == 0
    assert "vid1" in out and "overall" in out

    d = diagnose.diagnose_video(labels, preds, whistles)
    assert d.labels == {"matched": 1, "offset_only": 1, "end_missed": 1, "no_whistle": 1}
    assert d.preds == {"overlap": 1, "dead_time": 1}  # (35, 51) covers rally 2


def test_boundary_buckets():
    labels = [(10.0, 20.0), (30.0, 40.0), (50.0, 55.0), (58.0, 62.0), (70.0, 80.0), (90.0, 95.0)]
    preds = [
        (10.3, 20.2),  # matched
        (28.0, 40.5),  # one label, start 2 s early -> boundary
        (49.0, 63.0),  # covers two labels -> both merged
        (70.0, 74.0),  # two preds inside one label -> split
        (75.0, 80.0),
        (120.0, 130.0),  # dead time, contains a cut
    ]
    d = diagnose.boundary_diagnosis(labels, preds, cuts=[125.0, 300.0])
    assert d.labels == {"matched": 1, "boundary": 1, "merged": 2, "split": 1, "missed": 1}
    assert d.unmatched_preds == 5 and d.unmatched_with_cut == 1
    assert d.start_offsets == [pytest.approx(0.3), pytest.approx(-2.0)]
    assert d.end_offsets == [pytest.approx(0.2), pytest.approx(0.5)]


def test_cli_without_whistles_prints_boundary_table(tmp_path, capsys):
    for d in ("labels", "pred"):
        (tmp_path / d).mkdir()
    rows = lambda rs: json.dumps([{"start_sec": s, "end_sec": e} for s, e in rs])  # noqa: E731
    (tmp_path / "labels" / "vid1.json").write_text(rows([(10.0, 20.0)]))
    (tmp_path / "pred" / "vid1.json").write_text(rows([(10.0, 20.0)]))
    (tmp_path / "pred" / "vid1.cuts.json").write_text(json.dumps([{"t_sec": 5.0}]))
    code = diagnose.main(["--pred", str(tmp_path / "pred"), "--labels", str(tmp_path / "labels")])
    out = capsys.readouterr().out
    assert code == 0
    assert "boundaries" in out and "whistles near labels" not in out
