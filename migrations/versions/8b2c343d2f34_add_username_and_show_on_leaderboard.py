"""Add username and show_on_leaderboard to User

Revision ID: 8b2c343d2f34
Revises: de97489c4113
Create Date: 2025-08-25 16:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '8b2c343d2f34'
down_revision = 'de97489c4113'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.add_column(sa.Column('username', sa.String(length=80), nullable=True))
        batch_op.add_column(sa.Column('show_on_leaderboard', sa.Boolean(), nullable=True))


def downgrade():
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.drop_column('show_on_leaderboard')
        batch_op.drop_column('username')
