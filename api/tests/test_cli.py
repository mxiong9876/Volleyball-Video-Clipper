import json

import pytest
from sqlalchemy import select
from volley_pipeline import detectors, ingest, motion

from app import cli, config
from app.models import AnalysisJob, Rally, Video
from app.rallies import replace_rallies
from tests.conftest import add_video

KNOWN = "aaaaaaaaaaa"
NEW = "bbbbbbbbbbb"
HELD = "ccccccccccc"


def write_labels(labels_dir, youtube_id, rallies):
    labels_dir.mkdir(exist_ok=True)
    (labels_dir / f"{youtube_id}.json").write_text(
        json.dumps([{"start_sec": s, "end_sec": e} for s, e in rallies])
    )


@pytest.fixture
def labels_args(tmp_path):
    doc = tmp_path / "videos.md"
    doc.write_text(f"| ID | Split |\n|---|---|\n| {KNOWN} | tune |\n| {HELD} | held-out |\n")
    return ["import-labels", "--labels", str(tmp_path / "labels"), "--videos-doc", str(doc)]


@pytest.fixture
def lookup(monkeypatch):
    calls = []

    def fake(url, timeout=10):
        calls.append(url)
        return {"youtube_id": url[-11:], "title": "Looked up", "duration_sec": 600}

    monkeypatch.setattr(ingest, "fetch_metadata", fake)
    return calls


def rows(session_factory, youtube_id, source=None) -> list[tuple]:
    with session_factory() as s:
        query = (
            select(Rally)
            .join(Video, Rally.video_id == Video.id)
            .where(Video.youtube_id == youtube_id)
            .order_by(Rally.source, Rally.rally_index)
        )
        if source:
            query = query.where(Rally.source == source)
        return [(r.source, r.rally_index, r.start_sec, r.end_sec) for r in s.scalars(query)]


def test_import_labels_is_idempotent_and_leaves_predictions(
    session, session_factory, tmp_path, labels_args, lookup
):
    video = add_video(session, youtube_id=KNOWN)
    replace_rallies(session, video.id, "motion_v1", [(1.0, 2.0)])
    session.commit()
    write_labels(tmp_path / "labels", KNOWN, [(5.0, 6.0), (3.0, 4.0)])

    expected = [
        ("label", 0, 3.0, 4.0),
        ("label", 1, 5.0, 6.0),
        ("motion_v1", 0, 1.0, 2.0),
    ]
    assert cli.main(labels_args) == 0
    assert rows(session_factory, KNOWN) == expected
    assert cli.main(labels_args) == 0
    assert rows(session_factory, KNOWN) == expected

    write_labels(tmp_path / "labels", KNOWN, [(7.0, 8.0)])
    assert cli.main(labels_args) == 0
    assert rows(session_factory, KNOWN, "label") == [("label", 0, 7.0, 8.0)]
    assert lookup == []  # known video: no network


def test_import_labels_skips_held_out_unless_included(
    session_factory, tmp_path, labels_args, lookup, capsys
):
    write_labels(tmp_path / "labels", HELD, [(1.0, 2.0)])

    assert cli.main(labels_args) == 0
    assert f"{HELD}: held out, skipped" in capsys.readouterr().out
    assert rows(session_factory, HELD) == []
    assert lookup == []

    assert cli.main([*labels_args, "--include-held-out"]) == 0
    assert rows(session_factory, HELD) == [("label", 0, 1.0, 2.0)]


def test_import_labels_creates_missing_video_without_a_job(
    session_factory, tmp_path, labels_args, lookup
):
    write_labels(tmp_path / "labels", NEW, [(1.0, 2.0)])
    assert cli.main(labels_args) == 0
    assert lookup == [f"https://www.youtube.com/watch?v={NEW}"]
    with session_factory() as s:
        video = s.scalar(select(Video).where(Video.youtube_id == NEW))
        assert (video.title, video.duration_sec) == ("Looked up", 600)
        assert s.scalars(select(AnalysisJob)).all() == []


def test_import_labels_lookup_failure_continues_and_exits_1(
    session, session_factory, tmp_path, labels_args, monkeypatch, capsys
):
    add_video(session, youtube_id=NEW)  # sorts after KNOWN, whose lookup fails

    def fail(url, timeout=10):
        raise ingest.VideoUnavailableError("Video unavailable")

    monkeypatch.setattr(ingest, "fetch_metadata", fail)
    write_labels(tmp_path / "labels", KNOWN, [(1.0, 2.0)])
    write_labels(tmp_path / "labels", NEW, [(3.0, 4.0)])

    assert cli.main(labels_args) == 1
    assert f"error: {KNOWN}: Video unavailable" in capsys.readouterr().err
    assert rows(session_factory, NEW) == [("label", 0, 3.0, 4.0)]


def test_import_labels_bad_file_exits_1(session_factory, tmp_path, labels_args, capsys):
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / f"{KNOWN}.json").write_text('{"not": "a list"}')
    assert cli.main(labels_args) == 1
    assert "expected a list" in capsys.readouterr().err


def test_import_labels_no_files_exits_1(session_factory, labels_args, capsys):
    assert cli.main(labels_args) == 1
    assert "no label files" in capsys.readouterr().err


@pytest.fixture
def media(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path


def test_detect_replaces_current_source_only(
    session, session_factory, media, fake_detector, capsys
):
    video = add_video(session, youtube_id=KNOWN)
    (media / KNOWN).mkdir()
    replace_rallies(session, video.id, "label", [(100.0, 101.0)])
    replace_rallies(session, video.id, detectors.CURRENT, [(50.0, 51.0)])
    session.commit()
    add_video(session, youtube_id=NEW)  # no media

    assert cli.main(["detect"]) == 0

    assert fake_detector.calls == [(detectors.CURRENT, media / KNOWN)]
    assert rows(session_factory, KNOWN) == sorted(
        [
            ("label", 0, 100.0, 101.0),
            (detectors.CURRENT, 0, 1.0, 5.0),
            (detectors.CURRENT, 1, 10.12, 20.46),
        ]
    )
    assert f"{NEW}: not downloaded, skipped" in capsys.readouterr().out


def test_detect_named_video_only(session, session_factory, media, fake_detector):
    add_video(session, youtube_id=KNOWN)
    add_video(session, youtube_id=NEW)
    (media / KNOWN).mkdir()
    (media / NEW).mkdir()
    assert cli.main(["detect", NEW]) == 0
    assert fake_detector.calls == [(detectors.CURRENT, media / NEW)]


def test_detect_unknown_video_exits_1(session_factory, media, capsys):
    assert cli.main(["detect", KNOWN]) == 1
    assert f"error: {KNOWN}: not in the database" in capsys.readouterr().err


def test_detect_error_keeps_old_rows_and_exits_1(
    session, session_factory, media, fake_detector, capsys
):
    video = add_video(session, youtube_id=KNOWN)
    (media / KNOWN).mkdir()
    replace_rallies(session, video.id, detectors.CURRENT, [(1.0, 2.0)])
    session.commit()
    fake_detector.error = motion.MotionError("bad file")

    assert cli.main(["detect"]) == 1
    assert f"error: {KNOWN}: bad file" in capsys.readouterr().err
    assert rows(session_factory, KNOWN) == [(detectors.CURRENT, 0, 1.0, 2.0)]
