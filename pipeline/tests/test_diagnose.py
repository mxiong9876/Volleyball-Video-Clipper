import json

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
