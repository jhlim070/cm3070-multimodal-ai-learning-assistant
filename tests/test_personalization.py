"""
Personalisation validation testing

The system should incorporate each learner's preferences into the prompt
used to generate an answer. These tests mock the LLM so that they verify
prompt construction and interaction logging without depending on model
inference or answer quality.

The test doubles accept ``**kw`` because the production code passes
additional keyword arguments to its retrieval and LLM collaborators.

Mocks target the collaborators actually used by ``generate_answer()``:
``engine._client.chat`` for LLM calls and ``engine.retrieve_with_sources``
for retrieval.
"""
import pytest

import db
import engine


@pytest.fixture(autouse=True)
def _no_real_source_lookup(monkeypatch):
    """
    Source matching is outside the scope of these tests, so disable the
    source lookup to keep the tests focused on personalisation and logging.
    """
    monkeypatch.setattr(engine, "list_sources", lambda user_id: [])


def _fake_single_source_results(text="some context chunk", source="doc.pdf"):
    """
    Return a single-source result so these tests do not trigger retrieval disambiguation.
    """
    return [{"text": text, "source": source, "distance": 0.1}]


@pytest.mark.checklist
def test_system_prompt_reflects_learning_style_and_difficulty(temp_db, monkeypatch):
    db.save_profile("alice", "visual examples", "beginner", ["biology"])
    db.save_profile("bob", "detailed technical", "advanced", ["biology"])

    captured_system_prompts = []

    def fake_chat(model, messages, **kw):
        # Capture the system prompt constructed by generate_answer().
        captured_system_prompts.append(messages[0]["content"])
        return {"message": {"content": "mock answer"}}

    monkeypatch.setattr(engine._client, "chat", fake_chat)
    monkeypatch.setattr(engine, "retrieve_with_sources", lambda user_id, query, top_k=5, **kw: _fake_single_source_results())

    engine.generate_answer("alice", "What is a cell?")
    engine.generate_answer("bob", "What is a cell?")

    alice_prompt, bob_prompt = captured_system_prompts

    assert "visual examples" in alice_prompt
    assert "beginner" in alice_prompt
    assert "detailed technical" in bob_prompt
    assert "advanced" in bob_prompt
    assert alice_prompt != bob_prompt


@pytest.mark.checklist
def test_missing_profile_falls_back_to_defaults(temp_db, monkeypatch):
    captured = []

    def fake_chat(model, messages, **kw):
        captured.append(messages[0]["content"])
        return {"message": {"content": "mock answer"}}

    monkeypatch.setattr(engine._client, "chat", fake_chat)
    monkeypatch.setattr(engine, "retrieve_with_sources", lambda user_id, query, top_k=5, **kw: _fake_single_source_results())

    # No profile exists for this user, so the default preferences should be used.
    engine.generate_answer("ghost", "What is a cell?")

    assert "clear and detailed" in captured[0]
    assert "intermediate" in captured[0]


@pytest.mark.checklist
def test_generate_answer_logs_the_interaction(temp_db, monkeypatch):
    monkeypatch.setattr(
        engine._client, "chat",
        lambda model, messages, **kw: {"message": {"content": "answer text"}},
    )
    monkeypatch.setattr(engine, "retrieve_with_sources", lambda user_id, query, top_k=5, **kw: _fake_single_source_results())

    engine.generate_answer("alice", "What is mitosis?")

    # Verify the persisted interaction rather than only the function's return value.
    import sqlite3
    conn = sqlite3.connect(temp_db)
    c = conn.cursor()
    c.execute("SELECT query, answer, topic FROM interactions WHERE user_id=?", ("alice",))
    rows = c.fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0][0] == "What is mitosis?"
    assert rows[0][1] == "answer text"
    # No profile topic matches "mitosis", so extract_topic() falls back to the keyword.
    assert rows[0][2] == "mitosis"


def test_ollama_failure_is_not_logged_as_an_interaction(temp_db, monkeypatch):
    """
    A failed LLM call should return an error without creating an interaction.
    Logging failed requests would contaminate interaction history and could
    cause error responses to be included in weak-topic feedback.
    """
    def raise_timeout(model, messages, **kw):
        raise TimeoutError("simulated ollama outage")

    monkeypatch.setattr(engine._client, "chat", raise_timeout)
    monkeypatch.setattr(engine, "retrieve_with_sources", lambda user_id, query, top_k=5, **kw: _fake_single_source_results())

    answer = engine.generate_answer("alice", "What is mitosis?")

    assert answer.startswith("⚠️")
    assert db.get_last_interaction_id("alice") is None