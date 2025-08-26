"""Add PageView table

Revision ID: d1f7b65e4b00
Revises: 8b2c343d2f34
Create Date: 2024-08-26 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'd1f7b65e4b00'
down_revision = '8b2c343d2f34'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'page_view',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('ip_address', sa.String(length=45), nullable=False),
        sa.Column('user_agent', sa.String(length=255), nullable=False),
        sa.Column('first_seen', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('ip_address', 'user_agent', name='uniq_page_view')
    )


def downgrade():
    op.drop_table('page_view')
