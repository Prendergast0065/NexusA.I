import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app, db, PageView
import os


def setup_module(module):
    os.makedirs('/data', exist_ok=True)
    with app.app_context():
        db.drop_all()
        db.create_all()


def test_unique_page_views():
    with app.app_context():
        db.session.query(PageView).delete()
        db.session.commit()
        client = app.test_client()

        client.get('/', headers={'User-Agent': 'UA1'}, environ_base={'REMOTE_ADDR': '1.1.1.1'})
        assert PageView.query.count() == 1

        client.get('/', headers={'User-Agent': 'UA1'}, environ_base={'REMOTE_ADDR': '1.1.1.1'})
        assert PageView.query.count() == 1

        client.get('/', headers={'User-Agent': 'UA2'}, environ_base={'REMOTE_ADDR': '2.2.2.2'})
        assert PageView.query.count() == 2
