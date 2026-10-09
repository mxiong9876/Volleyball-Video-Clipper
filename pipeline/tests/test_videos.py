from pathlib import Path

import pytest

from volley_pipeline import videos

TABLE = """# Videos

Some intro text.

| ID | What | Split | Labels |
|---|---|---|---|
| aaaaaaaaaaa | match a | tune | done |
| bbbbbbbbbbb | match b | held-out | done |
| ccccccccccc | match c | Held out | not started |

Notes after the table | with a pipe.
"""


def test_held_out_ids(tmp_path):
    p = tmp_path / "videos.md"
    p.write_text(TABLE)
    assert videos.held_out_ids(p) == {"bbbbbbbbbbb", "ccccccccccc"}


def test_finds_split_column_by_header(tmp_path):
    p = tmp_path / "videos.md"
    p.write_text("| Split | ID |\n|---|---|\n| held-out | xxxxxxxxxxx |\n| tune | yyyyyyyyyyy |\n")
    assert videos.held_out_ids(p) == {"xxxxxxxxxxx"}


def test_no_held_out(tmp_path):
    p = tmp_path / "videos.md"
    p.write_text("| ID | Split |\n|---|---|\n| aaaaaaaaaaa | tune |\n")
    assert videos.held_out_ids(p) == set()


def test_rejects_glued_rows(tmp_path):
    p = tmp_path / "videos.md"
    p.write_text("| ID | Split |\n|---|---|\n| aaaaaaaaaaa | tune || bbbbbbbbbbb | held-out |\n")
    with pytest.raises(ValueError, match="cells"):
        videos.held_out_ids(p)


def test_rejects_missing_table(tmp_path):
    p = tmp_path / "videos.md"
    p.write_text("# Videos\n\nnothing here\n")
    with pytest.raises(ValueError, match="no table"):
        videos.held_out_ids(p)


def test_default_videos_doc_from_data_dir(tmp_path):
    assert videos.default_videos_doc(tmp_path / "data" / "labels") == (
        tmp_path.resolve() / "docs" / "videos.md"
    )


def test_repo_videos_doc_parses():
    doc = Path(__file__).resolve().parents[2] / "docs" / "videos.md"
    assert isinstance(videos.held_out_ids(doc), set)
