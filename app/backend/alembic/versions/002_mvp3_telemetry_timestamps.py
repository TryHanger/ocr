"""Add review details and lifecycle timing columns to documents.

Revision ID: 002_mvp3_telemetry
Revises: 
Create Date: 2026-09-28 13:08:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "002_mvp3_telemetry"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("review_details_json", sa.JSON(), nullable=True))
    op.add_column("documents", sa.Column("processing_started_at", sa.DateTime(), nullable=True))
    op.add_column("documents", sa.Column("processing_finished_at", sa.DateTime(), nullable=True))
    op.add_column("documents", sa.Column("review_started_at", sa.DateTime(), nullable=True))
    op.add_column("documents", sa.Column("review_finished_at", sa.DateTime(), nullable=True))
    op.add_column("documents", sa.Column("review_duration_ms", sa.Float(), nullable=True))
    op.add_column("documents", sa.Column("review_wait_duration_ms", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "review_wait_duration_ms")
    op.drop_column("documents", "review_duration_ms")
    op.drop_column("documents", "review_finished_at")
    op.drop_column("documents", "review_started_at")
    op.drop_column("documents", "processing_finished_at")
    op.drop_column("documents", "processing_started_at")
    op.drop_column("documents", "review_details_json")
