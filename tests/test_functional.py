"""
Functional Test Case

Covers: learner profile creation, interaction logging, weak-topic
detection, and the upload_and_index / respond handlers in app.py.

External services (Ollama, PaddleOCR, ChromaDB embeddings) are mocked
here on purpose -- this suite checks that YOUR logic wires things
together correctly, not that the third-party libraries work. That is
what Experiment 1/2 and the OCR robustness script are for.
"""
import pytest

import app  
import db

# Learner profile -----------------------------------------------------


@pytest.mark.checklist
def test_profile_create_and_retrieve(temp_db):
    db.save_profile("alice", "step-by-step", "beginner", ["python", "databases"])
    profile = db.get_profile("alice")
    assert profile is not None
    assert profile["learning_style"] == "step-by-step"
    assert profile["difficulty"] == "beginner"
    assert profile["topics"] == ["python", "databases"]


@pytest.mark.checklist
def test_profile_missing_user_returns_none(temp_db):
    assert db.get_profile("nobody_registered") is None


@pytest.mark.checklist
def test_profile_overwrite_on_resave(temp_db):
    db.save_profile("alice", "step-by-step", "beginner", ["python"])
    db.save_profile("alice", "visual examples", "advanced", ["machine learning"])
    profile = db.get_profile("alice")
    assert profile["learning_style"] == "visual examples"
    assert profile["difficulty"] == "advanced"



# Interaction logging + weak topic detection --------------------------

# NOTE: Weak topics are grouped by source document. A document is flagged
# after at least 3 low ratings (1-2). Tests must therefore provide `source`;
# rows without one are excluded from the aggregation.


@pytest.mark.checklist
def test_log_interaction_and_weak_topic_detected(temp_db):
    for _ in range(3):
        db.log_interaction("alice", "explain RAG", "some answer", "RAG", feedback=1, source="lecture1.pdf")
    weak = db.get_weak_topics("alice", threshold=3)
    assert len(weak) == 1
    assert weak[0][0] == "lecture1.pdf"
    assert weak[0][1] == 3          # number of low ratings (1-2)
    assert weak[0][2] == 1.0        # average feedback


@pytest.mark.checklist
def test_weak_topics_excludes_high_feedback_topics(temp_db):
    for _ in range(3):
        db.log_interaction("bob", "explain SQL joins", "answer", "SQL", feedback=5, source="lecture1.pdf")
    weak = db.get_weak_topics("bob", threshold=3)
    assert weak == []


@pytest.mark.checklist
def test_weak_topics_respects_threshold(temp_db):
    # Only 2 low ratings logged; the threshold is 3, so this document should NOT surface yet.
    for _ in range(2):
        db.log_interaction("carol", "explain pointers", "answer", "pointers", feedback=1, source="lecture1.pdf")
    weak = db.get_weak_topics("carol", threshold=3)
    assert weak == []


def test_weak_topics_ignores_null_feedback(temp_db):
    db.log_interaction("dave", "q1", "a1", "recursion", feedback=None, source="lecture1.pdf")
    db.log_interaction("dave", "q2", "a2", "recursion", feedback=None, source="lecture1.pdf")
    db.log_interaction("dave", "q3", "a3", "recursion", feedback=None, source="lecture1.pdf")
    weak = db.get_weak_topics("dave", threshold=3)
    # rows with NULL feedback are excluded by the WHERE clause in get_weak_topics
    assert weak == []



# app.py handlers ---------------------------------------------------

# NOTE: Importing app.py builds the Gradio UI and initialises learner.db.
# This is harmless because db.init_db() is idempotent, but makes imports
# slower and couples tests to application startup.

class _FakeFile:
    def __init__(self, name):
        self.name = name


def _real_file(tmp_path, name="fake.pdf", content=b"%PDF-1.4\n% test fixture\n"):
    """
    Create a real temporary file with a PDF signature so upload
    validation passes before the parser is mocked.
    """
    p = tmp_path / name
    p.write_bytes(content)
    return _FakeFile(str(p))


@pytest.mark.checklist
def test_upload_and_index_success(temp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(app, "process_file", lambda path: "Paragraph one.\n\nParagraph two.")
    monkeypatch.setattr(app, "chunk_text", lambda text, **kwargs: ["chunk1", "chunk2"])

    captured = {}

    # Accept `metadatas` because upload_and_index passes source 
    # metadata for every indexed chunk.
    def fake_add_documents(user_id, chunks, metadatas=None, **kw):
        captured["user_id"] = user_id
        captured["chunks"] = chunks
        captured["metadatas"] = metadatas

    monkeypatch.setattr(app, "add_documents", fake_add_documents)

    result = app.upload_and_index(_real_file(tmp_path, "fake.pdf"), "alice")

    assert "Indexed 2 chunks" in result
    assert captured["user_id"] == "alice"
    assert captured["chunks"] == ["chunk1", "chunk2"]
    assert captured["metadatas"] == [{"source": "fake.pdf"}, {"source": "fake.pdf"}]


@pytest.mark.checklist
def test_upload_and_index_no_file_selected(temp_db):
    result = app.upload_and_index(None, "alice")
    assert "No file selected" in result


@pytest.mark.checklist
def test_upload_and_index_no_extractable_text(temp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(app, "process_file", lambda path: "")
    monkeypatch.setattr(app, "chunk_text", lambda text, **kwargs: [])
    result = app.upload_and_index(_real_file(tmp_path, "blank.pdf"), "alice")
    # Match the user-facing message case-insensitively.
    assert "no text could be extracted" in result.lower()


@pytest.mark.checklist
def test_upload_and_index_handles_parser_failure(temp_db, tmp_path, monkeypatch):
    """
    A parser that raises on one file is reported for that file, not raised.
    """
    def raise_unsupported(path):
        raise ValueError("Unsupported file type")

    monkeypatch.setattr(app, "process_file", raise_unsupported)
    result = app.upload_and_index(_real_file(tmp_path, "file.pdf"), "alice")
    assert "Error" in result
    assert "Unsupported file type" in result


@pytest.mark.checklist
def test_respond_delegates_to_pipeline(monkeypatch):
    """
    Verify that respond() passes through the structured pipeline result.
    """
    monkeypatch.setattr(
        app, "process_image_and_text_detailed",
        lambda uid, msg, img: {"answer": f"echo:{uid}:{msg}", "kind": "answer",
                               "rateable": True, "sources": [], "model": "llama3.2"},
    )
    out = app.respond("what is a hash table?", [], "alice", None, False)
    assert out[0] == "echo:alice:what is a hash table?"


@pytest.mark.checklist
def test_get_recommendations_btn_delegates(monkeypatch):
    monkeypatch.setattr(app, "generate_recommendations", lambda uid: f"recs-for-{uid}")
    assert app.get_recommendations_btn("alice") == "recs-for-alice"