"""
Tests for multi-document disambiguation

These tests cover engine._check_ambiguous() and its integration with
generate_answer(). When retrieved evidence is closely split across multiple
uploaded documents, the system should ask the learner to identify the intended
document rather than choosing one without sufficient evidence.

A clarification is handled without an LLM call and is not recorded as a
normal interaction.

The ambiguity check is retrieval-based: it compares document sources and their
retrieval distances. It is therefore a heuristic rather than a full
understanding of the learner's intent. For example, it may not detect every
case where multiple documents are semantically plausible, and it cannot
resolve ambiguity within a single document.
"""
import db
import engine


def test_check_ambiguous_single_source_is_not_ambiguous():
    results = [
        {"text": "a", "source": "doc1.pdf", "distance": 0.10},
        {"text": "b", "source": "doc1.pdf", "distance": 0.15},
    ]
    assert engine._check_ambiguous(results) is None


def test_check_ambiguous_two_sources_with_clear_winner_is_not_ambiguous():
    results = [
        {"text": "a", "source": "doc1.pdf", "distance": 0.05},
        {"text": "b", "source": "doc2.pdf", "distance": 0.40},
    ]
    assert engine._check_ambiguous(results) is None


def test_check_ambiguous_two_close_sources_is_ambiguous():
    results = [
        {"text": "a", "source": "doc1.pdf", "distance": 0.10},
        {"text": "b", "source": "doc2.pdf", "distance": 0.12},
    ]
    assert engine._check_ambiguous(results) == ["doc1.pdf", "doc2.pdf"]


def test_check_ambiguous_three_close_sources_returns_top_three():
    results = [
        {"text": "a", "source": "doc1.pdf", "distance": 0.10},
        {"text": "b", "source": "doc2.pdf", "distance": 0.11},
        {"text": "c", "source": "doc3.pdf", "distance": 0.12},
        {"text": "d", "source": "doc4.pdf", "distance": 0.13},
    ]
    result = engine._check_ambiguous(results)
    assert len(result) == 3
    assert result == ["doc1.pdf", "doc2.pdf", "doc3.pdf"]


def test_check_ambiguous_handles_empty_results():
    assert engine._check_ambiguous([]) is None


def test_check_ambiguous_handles_missing_metadata_gracefully():
    """ 
    Older or incompletely indexed chunks may not contain a source value.
    Without a source, the result cannot be attributed to a document and
    therefore should not trigger document-level clarification.
    """
    results = [
        {"text": "a", "source": None, "distance": 0.1},
        {"text": "b", "source": None, "distance": 0.2},
    ]
    assert engine._check_ambiguous(results) is None


def test_generate_answer_returns_clarification_without_calling_llm(temp_db, monkeypatch):
    """
    Force retrieval to return similarly relevant chunks from two documents.
    The clarification should be generated directly from the retrieval result,
    so the LLM must not be called.
    """
    monkeypatch.setattr(engine, "list_sources", lambda user_id: [])
    llm_calls = []
    monkeypatch.setattr(engine._client, "chat", lambda model, messages, **kw: llm_calls.append(1) or {"message": {"content": "should not happen"}})
    monkeypatch.setattr(
        engine, "retrieve_with_sources",
        lambda user_id, query, top_k=5, **kw: [
            {"text": "about normalisation", "source": "CM3010.pdf", "distance": 0.10},
            {"text": "about routing", "source": "CM2040.pdf", "distance": 0.11},
        ],
    )

    answer = engine.generate_answer("alice", "what does it cover?")

    assert answer.startswith("🤔")
    assert "CM3010.pdf" in answer
    assert "CM2040.pdf" in answer
    assert llm_calls == [], "the LLM must not be called for an ambiguous query"


def test_generate_answer_clarification_is_not_logged(temp_db, monkeypatch):
    """
    A clarification is an intermediate prompt, not an answered learner
    question, so it must not create a normal interaction record.
    """
    monkeypatch.setattr(engine, "list_sources", lambda user_id: [])
    monkeypatch.setattr(engine._client, "chat", lambda model, messages, **kw: {"message": {"content": "unused"}})
    monkeypatch.setattr(
        engine, "retrieve_with_sources",
        lambda user_id, query, top_k=5, **kw: [
            {"text": "a", "source": "doc1.pdf", "distance": 0.10},
            {"text": "b", "source": "doc2.pdf", "distance": 0.11},
        ],
    )

    engine.generate_answer("alice", "what does it cover?")

    assert db.get_last_interaction_id("alice") is None


def test_generate_answer_proceeds_normally_when_unambiguous(temp_db, monkeypatch):
    """
    When the retrieved evidence points clearly to one document, normal
    answer generation and interaction logging should continue unchanged.
    """
    monkeypatch.setattr(engine, "list_sources", lambda user_id: [])
    monkeypatch.setattr(engine._client, "chat", lambda model, messages, **kw: {"message": {"content": "a real answer"}})
    monkeypatch.setattr(
        engine, "retrieve_with_sources",
        lambda user_id, query, top_k=5, **kw: [
            {"text": "about routing", "source": "CM2040.pdf", "distance": 0.05},
            {"text": "more about routing", "source": "CM2040.pdf", "distance": 0.08},
        ],
    )

    answer = engine.generate_answer("alice", "what is separation of concerns?")

    assert answer == "a real answer"
    assert db.get_last_interaction_id("alice") is not None