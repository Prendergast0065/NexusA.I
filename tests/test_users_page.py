import os
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'

from app import app, db, User, hash_pw
import pytest

app.config['TESTING'] = True

@pytest.fixture(autouse=True)
def setup_db():
    with app.app_context():
        db.create_all()
        admin = User(email='prendergast307@gmail.com', pw_hash=hash_pw('pw'))
        user = User(email='other@example.com', pw_hash=hash_pw('pw'))
        db.session.add_all([admin, user])
        db.session.commit()
        yield
        db.session.remove()
        db.drop_all()

def login(client, email):
    return client.post('/login', data={'email': email, 'password': 'pw'}, follow_redirects=True)

def test_admin_can_access_users_page():
    with app.test_client() as client:
        login(client, 'prendergast307@gmail.com')
        resp = client.get('/users')
        assert resp.status_code == 200
        assert b'prendergast307@gmail.com' in resp.data

def test_normal_user_gets_403():
    with app.test_client() as client:
        login(client, 'other@example.com')
        resp = client.get('/users')
        assert resp.status_code == 403
