"""rallies table and detecting_rallies job status

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-02 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _set_status_check(statuses: str) -> None:
    # Autogenerate can't see CHECK changes, so this part is written by hand.
    op.drop_constraint("ck_analysis_jobs_status", "analysis_jobs", type_="check")
    op.create_check_constraint(
        "ck_analysis_jobs_status", "analysis_jobs", f"status IN ({statuses})"
    )


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "rallies",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("rally_index", sa.Integer(), nullable=False),
        sa.Column("start_sec", sa.Float(), nullable=False),
        sa.Column("end_sec", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("video_id", "source", "rally_index"),
    )
    _set_status_check(
        "'queued', 'downloading', 'extracting_audio', 'detecting_rallies', 'done', 'failed'"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(
        "UPDATE analysis_jobs SET status = 'failed', error_msg = 'schema downgraded during "
        "rally detection', finished_at = now() WHERE status = 'detecting_rallies'"
    )
    _set_status_check("'queued', 'downloading', 'extracting_audio', 'done', 'failed'")
    op.drop_table("rallies")
