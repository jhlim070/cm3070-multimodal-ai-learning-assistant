"""
Shared fixtures for the Experiment 3 (System Validation) test suite.

These fixtures isolate tests from the application's persistent state:
SQLite connections made by db.py are redirected to a temporary database, 
and rag.py's ChromaDB client is redirected to a temporary directory. 
Tests therefore never modify the real learner.db or ./chroma_db.

No application source files are modified by these fixtures.
"""
import sqlite3
import sys
import os
import pytest


def pytest_configure(config):
    """
    Register the custom pytest markers used by the Experiment 3 test suite.

    The `checklist` marker identifies the tests referenced by
    feature_validation_checklist.md. This allows the checklist to be run
    directly with ``pytest -m checklist -v`` without duplicating or relocating
    the underlying tests.
    """
    config.addinivalue_line(
        "markers", "checklist: the tests feature_validation_checklist.md's rows reference "
                  "(run with: pytest -m checklist -v)")

# Add the project root to sys.path so the test suite can import application
# modules such as app.py, db.py, engine.py, and rag.py.
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Once path is set, can safely import application module
import db  # noqa: E402


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """
    Redirect all SQLite connections made through db.py to a temporary
    database for the duration of the test.

    The connection function is patched rather than db.py itself, so the
    application's source code and configured database path remain unchanged.
    """
    test_db_path = str(tmp_path / "test_learner.db")
    real_connect = sqlite3.connect

    def fake_connect(*args, **kwargs):
        return real_connect(test_db_path)

    monkeypatch.setattr(db.sqlite3, "connect", fake_connect)
    db.init_db()
    yield test_db_path


@pytest.fixture(autouse=True)
def _isolate_chroma(tmp_path, monkeypatch):
    """
    Redirect rag.py's ChromaDB client to a temporary directory for every test.

    This fixture is autouse because application code may access ChromaDB
    indirectly through functions such as engine.generate_answer() or
    app.upload_and_index(). Keeping the isolation at the fixture level prevents
    tests from accidentally reading or modifying the real ./chroma_db.

    Tests that require a different ChromaDB location may override rag.client
    with their own fixture.
    """
    try:
        import chromadb
        import rag
    except Exception:
        yield
        return
    test_client = chromadb.PersistentClient(path=str(tmp_path / "chroma_autouse"))
    monkeypatch.setattr(rag, "client", test_client)
    yield