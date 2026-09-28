"""
Tests for document-level weakness detection and personalised recommendations.

The rule implemented by db.get_weak_topics() flags a source document when it
has at least three learner ratings of 1 or 2. The average rating is reported
for context but is not used as an additional filtering condition.

The implementation operates at document level rather than keyword-topic level.
In the current evaluation design, concepts are generally queried only once, so
there may be insufficient repeated observations for a per-topic threshold to
be reached reliably.

These tests verify the detection rule, document attribution, database schema,
and recommendation text.
"""
import sqlite3

import pytest

import db
import engine


def _rate(user, source, ratings, topic="t"):
    """
    Create interactions for a user with the specified feedback ratings.
    """
    for r in ratings:
        db.log_interaction(user, "q", "a", topic, feedback=r, source=source)


# Weak-document detection rules -------------------------------------------

def test_three_low_ratings_flag_a_document(temp_db):
    _rate("alice", "lecture1.pdf", [1, 2, 1])
    weak = db.get_weak_topics("alice")
    assert [(w[0], w[1]) for w in weak] == [("lecture1.pdf", 3)]
    assert weak[0][2] == pytest.approx(4 / 3)


def test_two_low_ratings_are_not_enough(temp_db):
    _rate("alice", "lecture1.pdf", [1, 1])
    assert db.get_weak_topics("alice") == []


@pytest.mark.parametrize("rating, counts_as_low", [(1, True), (2, True), (3, False), (4, False), (5, False)])
def test_only_ratings_of_1_and_2_are_low(temp_db, rating, counts_as_low):
    _rate("alice", "lecture1.pdf", [rating] * 3)
    assert bool(db.get_weak_topics("alice")) is counts_as_low


def test_the_boundary_is_2_not_3(temp_db):
    _rate("alice", "lecture1.pdf", [2, 2, 3])   # exactly two low ratings; below the threshold
    assert db.get_weak_topics("alice") == []
    _rate("alice", "lecture1.pdf", [2])
    assert len(db.get_weak_topics("alice")) == 1


def test_the_average_is_reported_but_is_not_a_filter(temp_db):
    """
    The flagging condition depends on the number of low ratings, not the overall
    average. A document can therefore be flagged even when other ratings raise
    its average above the low-rating range.

    """
    _rate("alice", "lecture1.pdf", [1, 1, 2] + [5] * 5) 
    weak = db.get_weak_topics("alice")
    assert len(weak) == 1
    name, n_low, avg = weak[0]
    assert (name, n_low) == ("lecture1.pdf", 3)
    assert avg == pytest.approx((1 + 1 + 2 + 25) / 8)


def test_documents_are_counted_separately(temp_db):
    _rate("alice", "a.pdf", [1, 1])
    _rate("alice", "b.pdf", [1, 1])
    assert db.get_weak_topics("alice") == [], "low ratings must be aggregated independently per source document"
    _rate("alice", "a.pdf", [2])
    assert [w[0] for w in db.get_weak_topics("alice")] == ["a.pdf"]


def test_results_are_ordered_by_number_of_low_ratings_then_worst_average(temp_db):
    _rate("alice", "few.pdf", [1, 1, 1])
    _rate("alice", "many.pdf", [2, 2, 2, 2, 2])
    _rate("alice", "few_but_worse.pdf", [1, 1, 1])
    _rate("alice", "few_but_worse.pdf", [5])        # same low-rating count, but a higher average
    assert [w[0] for w in db.get_weak_topics("alice")] == ["many.pdf", "few.pdf", "few_but_worse.pdf"]


def test_unrated_answers_never_count(temp_db):
    _rate("alice", "lecture1.pdf", [None, None, None, None]) # Missing feedback does not contribute 
                                                             # to the low-rating count.
    assert db.get_weak_topics("alice") == []


@pytest.mark.parametrize("source", [None, ""])
def test_rows_without_a_source_are_excluded(temp_db, source):
    """
    Interactions without a source document are excluded because they cannot be
    attributed to a document-level weakness.
    """
    _rate("alice", source, [1, 1, 1, 1])
    assert db.get_weak_topics("alice") == []


def test_users_are_isolated(temp_db):
    _rate("alice", "lecture1.pdf", [1, 1, 1])
    assert db.get_weak_topics("bob") == []
    assert db.get_weak_topics("Alice ")[0][0] == "lecture1.pdf", "user IDs should use the same normalisation across database operations"


def test_the_threshold_is_adjustable(temp_db):
    _rate("alice", "lecture1.pdf", [1, 1])
    assert db.get_weak_topics("alice", threshold=2)[0][1] == 2
    assert db.get_weak_topics("alice", threshold=3) == []


def test_the_default_threshold_is_three():
    import inspect
    assert inspect.signature(db.get_weak_topics).parameters["threshold"].default == 3


# Database schema and migration behaviour -----------------------------------


def test_log_interaction_source_defaults_to_none(temp_db):
    db.log_interaction("alice", "q", "a", "t")
    conn = sqlite3.connect(temp_db)
    assert conn.execute("SELECT source FROM interactions").fetchone() == (None,)
    conn.close()


def test_init_db_adds_the_source_column_to_an_older_database(temp_db):
    conn = sqlite3.connect(temp_db)
    conn.execute("DROP TABLE interactions")
    conn.execute("""CREATE TABLE interactions (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, query TEXT, answer TEXT, topic TEXT,
                    feedback INTEGER, rateable INTEGER DEFAULT 1)""")
    conn.execute("INSERT INTO interactions (user_id, query, answer, topic, feedback) VALUES ('old','q','a','t',1)")
    conn.commit()
    conn.close()
    db.init_db()
    db.init_db()            # repeated initialisation should be harmless
    for _ in range(3):
        db.log_interaction("old", "q", "a", "t", feedback=1, source="new.pdf")
    weak = db.get_weak_topics("old")
    assert [(w[0], w[1]) for w in weak] == [("new.pdf", 3)], "legacy interactions without a source must not contribute to document-level counts"


# Source attribution for generated answers ------------------------------------


@pytest.fixture
def world(temp_db, monkeypatch):
    """
    Provide controllable retrieval and LLM responses for engine tests.
    """
    w = {"results": [], "llm": []}
    monkeypatch.setattr(engine, "list_sources", lambda u: [])
    monkeypatch.setattr(engine, "retrieve_with_sources",
                        lambda u, q, top_k=5, source_filter=None, **kw: list(w["results"]))
    monkeypatch.setattr(engine._client, "chat",
                        lambda model, messages, **kw: (w["llm"].append(messages) or {"message": {"content": "an answer"}}))
    return w


def _chunks(*sources):
    return [{"text": f"t{i}", "source": s, "distance": 0.1 + i / 10} for i, s in enumerate(sources)]


def _last_source(temp_db):
    """
    Return the source stored on the most recently logged interaction.
    """
    conn = sqlite3.connect(temp_db)
    row = conn.execute("SELECT source FROM interactions ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return row[0]


def test_an_answer_is_attributed_to_its_highest_ranked_document(world, temp_db):
    world["results"] = _chunks("B.pdf", "A.pdf", "A.pdf")
    engine.generate_answer_detailed("alice", "what is x?")
    assert _last_source(temp_db) == "B.pdf"


def test_an_answer_spanning_two_documents_goes_to_the_first_only(world, temp_db):
    """
    A single interaction stores one source, so multi-document answers are
    attributed to the highest-ranked retrieved document.
    """
    world["results"] = [{"text": "x", "source": "A.pdf", "distance": 0.10},
                        {"text": "y", "source": "B.pdf", "distance": 0.30}]
    engine.generate_answer_detailed("alice", "what is x?")
    assert _last_source(temp_db) == "A.pdf"


def test_a_refusal_can_never_be_rated_so_it_can_never_flag_a_document(world, temp_db, monkeypatch):
    """
    Responses indicating that the answer is outside the available materials
    are not rateable and therefore cannot contribute to document weakness.
    """
    world["results"] = _chunks("A.pdf")
    monkeypatch.setattr(engine._client, "chat", lambda model, messages, **kw: {"message": {"content": "NOT_IN_MATERIALS"}})
    for i in range(3):
        r = engine.generate_answer_detailed("alice", f"a question the material does not cover {i}?")
        assert r["rateable"] is False
    assert db.get_last_rateable_interaction_id("alice") is None
    assert db.get_weak_topics("alice") == []


def test_end_to_end_three_low_ratings_on_answers_flag_the_document(world, temp_db):
    world["results"] = _chunks("CM3050 Topic 5.pptx")
    for i in range(3):
        engine.generate_answer_detailed("alice", f"question number {i} about testing frameworks?")
        db.set_feedback(db.get_last_rateable_interaction_id("alice"), 1 + (i % 2))  # produces the sequence 1, 2, 1
    weak = db.get_weak_topics("alice")
    assert [(w[0], w[1]) for w in weak] == [("CM3050 Topic 5.pptx", 3)]


def test_two_low_ratings_are_not_flagged_end_to_end(world, temp_db):
    world["results"] = _chunks("A.pdf")
    for i in range(2):
        engine.generate_answer_detailed("alice", f"a question about topic {i}?")
        db.set_feedback(db.get_last_rateable_interaction_id("alice"), 1)
    assert db.get_weak_topics("alice") == []



# Recommendation generation and fallback behaviour -------------------------------


def _capture_chat(monkeypatch):
    """
    Capture LLM prompts while returning a deterministic recommendation.
    """
    seen = []
    monkeypatch.setattr(engine._client, "chat",
                        lambda model, messages, **kw: (seen.append(messages[0]["content"]) or {"message": {"content": "revise X"}}))
    return seen


def test_no_recommendations_until_a_document_is_flagged(temp_db, monkeypatch):
    seen = _capture_chat(monkeypatch)
    _rate("alice", "lecture1.pdf", [1, 1])
    assert engine.generate_recommendations("alice") == "No weak topics identified yet. Keep studying!"
    assert seen == [], "the LLM should not be called when no weak documents are identified"


def test_the_recommendation_prompt_names_the_flagged_documents(temp_db, monkeypatch):
    seen = _capture_chat(monkeypatch)
    _rate("alice", "CM3010 Normalisation.pdf", [1, 2, 1])
    out = engine.generate_recommendations("alice")
    assert out == "revise X"
    prompt = seen[0]
    assert "CM3010 Normalisation.pdf" in prompt
    assert "3 question(s) rated confusing/struggling" in prompt
    assert "average understanding 1.3/5" in prompt
    assert "naming the lecture/document" in prompt


def test_the_recommendation_prompt_lists_every_flagged_document(temp_db, monkeypatch):
    seen = _capture_chat(monkeypatch)
    _rate("alice", "a.pdf", [1, 1, 1])
    _rate("alice", "b.pdf", [2, 2, 2])
    engine.generate_recommendations("alice")
    assert "a.pdf" in seen[0] and "b.pdf" in seen[0]


def test_a_failing_llm_still_reports_the_flagged_documents(temp_db, monkeypatch):
    def boom(model, messages, **kw):
        raise TimeoutError("ollama down") # default model used
    monkeypatch.setattr(engine._client, "chat", boom)
    _rate("alice", "lecture1.pdf", [1, 1, 1])
    out = engine.generate_recommendations("alice")
    assert out.startswith("⚠️") and "lecture1.pdf" in out
