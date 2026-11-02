from collections.abc import Iterable

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models import Rally


def replace_rallies(
    session: Session, video_id: int, source: str, rallies: Iterable[tuple[float, float]]
) -> int:
    """Swap one source's rows for a video, leaving other sources alone. The caller commits."""
    session.execute(delete(Rally).where(Rally.video_id == video_id, Rally.source == source))
    rows = [
        Rally(
            video_id=video_id,
            rally_index=i,
            # 2 decimals, the same as the prediction and label files.
            start_sec=round(start, 2),
            end_sec=round(end, 2),
            source=source,
        )
        for i, (start, end) in enumerate(sorted(rallies))
    ]
    session.add_all(rows)
    return len(rows)
