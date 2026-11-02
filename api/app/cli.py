"""Dev commands on the rallies table, from api/:

uv run python -m app.cli import-labels [--labels ../data/labels] [--include-held-out]
uv run python -m app.cli detect [youtube_id ...]
"""

import argparse
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session
from volley_pipeline import detectors, eval, ingest, motion, videos

from app import config, db
from app.models import Video
from app.rallies import replace_rallies

LABEL_SOURCE = "label"


def _get_or_create_video(session: Session, youtube_id: str) -> Video:
    """A new video gets no job; POST /videos starts one when it's submitted."""
    video = session.scalar(select(Video).where(Video.youtube_id == youtube_id))
    if video is None:
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        meta = ingest.fetch_metadata(url, config.METADATA_TIMEOUT)
        video = Video(youtube_id=youtube_id, title=meta["title"], duration_sec=meta["duration_sec"])
        session.add(video)
        session.flush()
    return video


def import_labels(labels_dir: Path, videos_doc: str | None, include_held_out: bool) -> int:
    held_out = videos.load_held_out(videos_doc, labels_dir, include_held_out)
    paths = sorted(labels_dir.glob("*.json"))
    if not paths:
        print(f"error: no label files in {labels_dir}", file=sys.stderr)
        return 1
    status = 0
    with db.SessionLocal() as session:
        for path in paths:
            youtube_id = path.stem
            if youtube_id in held_out:
                print(f"{youtube_id}: held out, skipped")
                continue
            try:
                rallies = eval.load_rallies(path)
                video = _get_or_create_video(session, youtube_id)
            except (ValueError, ingest.IngestError) as exc:
                session.rollback()
                print(f"error: {youtube_id}: {exc}", file=sys.stderr)
                status = 1
                continue
            count = replace_rallies(session, video.id, LABEL_SOURCE, rallies)
            session.commit()
            print(f"{youtube_id}: {count} labels")
    return status


def detect(youtube_ids: list[str]) -> int:
    source = detectors.CURRENT
    status = 0
    with db.SessionLocal() as session:
        query = select(Video).order_by(Video.id)
        if youtube_ids:
            query = query.where(Video.youtube_id.in_(youtube_ids))
        found = session.scalars(query).all()
        for missing in sorted(set(youtube_ids) - {v.youtube_id for v in found}):
            print(f"error: {missing}: not in the database", file=sys.stderr)
            status = 1
        for video in found:
            media_dir = config.DATA_DIR / video.youtube_id
            if not media_dir.is_dir():
                print(f"{video.youtube_id}: not downloaded, skipped")
                continue
            try:
                rallies = detectors.detect(source, media_dir)
            except (motion.MotionError, OSError) as exc:
                print(f"error: {video.youtube_id}: {exc}", file=sys.stderr)
                status = 1
                continue
            count = replace_rallies(session, video.id, source, rallies)
            session.commit()
            print(f"{video.youtube_id}: {count} rallies ({source})")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dev commands on the rallies table.")
    commands = parser.add_subparsers(dest="command", required=True)

    labels = commands.add_parser("import-labels", help=f'store label files as "{LABEL_SOURCE}"')
    labels.add_argument("--labels", default=str(config.DATA_DIR.parent / "labels"))
    labels.add_argument("--videos-doc", help="videos table (default: docs/videos.md)")
    labels.add_argument("--include-held-out", action="store_true", help="also import held-out")

    run = commands.add_parser("detect", help="run the current detector on downloaded videos")
    run.add_argument("youtube_ids", nargs="*", help="default: every video in the database")

    args = parser.parse_args(argv)
    if args.command == "detect":
        return detect(args.youtube_ids)
    try:
        return import_labels(Path(args.labels), args.videos_doc, args.include_held_out)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
