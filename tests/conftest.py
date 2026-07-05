import pytest

from tokentriage import db


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Give every test a fresh SQLite schema instead of relying on tokentriage.db."""
    monkeypatch.setattr(db, "settings", type("S", (), {"db_path": str(tmp_path / "tokentriage-test.db")})())
    db.init_db()
