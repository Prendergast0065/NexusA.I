import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import hash_pw, verify_pw


def test_password_roundtrip():
    pw = "secret123"
    hashed = hash_pw(pw)
    assert hashed != pw
    assert verify_pw(pw, hashed) is True
    assert verify_pw("wrong", hashed) is False
