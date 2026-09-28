"""
Mandatory rating-gate validation and limitation testing.

Covers app.py's respond()/rate_response() mechanism: after a rateable
answer, the learner must submit a 1-5 understanding rating before another
question is accepted.

The tests verify both the intended behaviour and known limitations of the
gate, including session-local state, recency-based rating assignment, and
the inability of a single rating to distinguish topic difficulty from
answer quality.

These tests require the same app.py dependency stubs used elsewhere in the
test suite; alternatively, they can be run in the project's full environment.
"""
import db
import app


def _res(answer="an answer", kind="answer", rateable=True):
    """
    Create a structured pipeline result for app.respond() tests.

    The rating gate uses the explicit `rateable` flag rather than inspecting the answer text.
    """
    return {"answer": answer, "kind": kind, "rateable": rateable, "sources": [], "model": "llama3.2"}


def test_gate_blocks_new_question_until_rated(monkeypatch):
    call_count = {"n": 0}
    monkeypatch.setattr(app, "process_image_and_text_detailed", lambda uid, msg, img: (call_count.__setitem__("n", call_count["n"] + 1), _res())[1])

    # A rateable answer puts the session into the awaiting-rating state.
    answer1, _, _, awaiting1 = app.respond("Q1", [], "alice", None, False)
    assert awaiting1 is True
    assert call_count["n"] == 1

    # A new question is rejected while rating is pending; the pipeline is not called.
    answer2, _, _, awaiting2 = app.respond("Q2", [], "alice", None, True)
    assert "rate" in answer2.lower()
    assert awaiting2 is True
    assert call_count["n"] == 1, "the pipeline must not run again while the gate is open"


def test_rating_releases_the_gate(temp_db, monkeypatch):
    monkeypatch.setattr(app, "process_image_and_text_detailed", lambda uid, msg, img: _res())
    app.respond("Q1", [], "alice", None, False)  # leaves the session awaiting a rating

    status, panel_update, awaiting = app.rate_response("alice", 4)
    assert awaiting is False
    assert panel_update["visible"] is False


def test_error_responses_do_not_open_the_gate(monkeypatch):
    monkeypatch.setattr(app, "process_image_and_text_detailed", lambda uid, msg, img: _res("⚠️ Couldn't get a response from 'llama3.2' via Ollama. Details: connection refused", kind="error", rateable=False))
    answer, _, panel_update, awaiting = app.respond("Q1", [], "alice", None, False)
    assert awaiting is False
    assert panel_update["visible"] is False


def test_clarification_responses_do_not_open_the_gate(monkeypatch):
    monkeypatch.setattr(app, "process_image_and_text_detailed", lambda uid, msg, img: _res('🤔 I found relevant content in more than one of your uploaded documents ("a.pdf", "b.pdf"). Could you clarify?', kind="clarification", rateable=False))
    answer, _, panel_update, awaiting = app.respond("Q1", [], "alice", None, False)
    assert awaiting is False
    assert panel_update["visible"] is False


# ---------------------------------------------------------------------------
# Known limitations
# These tests document limitations of the current rating-gate design rather
# than failures of its intended behaviour.
# ---------------------------------------------------------------------------

def test_LIMITATION_gate_state_is_not_shared_across_sessions(monkeypatch):
    """
    The awaiting-rating state is session-local rather than persisted per user.

    Therefore, another session for the same user can submit a new question
    without satisfying the first session's pending rating.

    """
    monkeypatch.setattr(app, "process_image_and_text_detailed", lambda uid, msg, img: _res())

    # Session A receives a rateable answer and becomes awaiting-rating.
    _, _, _, awaiting_tab_a = app.respond("Q1", [], "alice", None, False)
    assert awaiting_tab_a is True

    # A separate session has independent gate state and is therefore not blocked.
    answer_tab_b, _, _, awaiting_tab_b = app.respond("Q2", [], "alice", None, False)
    assert answer_tab_b == "an answer"  # went through, unaffected by Tab A


def test_LIMITATION_rating_targets_whichever_interaction_is_most_recent_by_id(temp_db):
    """
    Ratings are assigned using the user's most recent interaction ID rather
    than an interaction ID associated with the rating control.

    If multiple unrated interactions exist, the rating can therefore be assigned
    to a different interaction from the one the learner intended to rate.
    """
    db.log_interaction("alice", "first question", "answer 1", "topic_a", feedback=None)
    db.log_interaction("alice", "second question", "answer 2", "topic_b", feedback=None)

    # The rating targets the most recent interaction, so the first interaction remains unrated.
    interaction_id = db.get_last_interaction_id("alice")
    db.set_feedback(interaction_id, 5)

    import sqlite3
    conn = sqlite3.connect(temp_db)
    c = conn.cursor()
    c.execute("SELECT query, feedback FROM interactions WHERE user_id=? ORDER BY id", ("alice",))
    rows = c.fetchall()
    conn.close()
    assert rows[0] == ("first question", None)   # never rated
    assert rows[1] == ("second question", 5)      # got the rating instead


def test_LIMITATION_rating_scale_conflates_answer_quality_with_topic_mastery(temp_db, monkeypatch):
    """
    This is a measurement limitation rather than an implementation error.

    The 1-5 rating records perceived understanding but does not distinguish
    between difficulty caused by the topic and difficulty caused by the answer.
    Consequently, get_weak_topics() cannot infer the cause of a low rating from
    the stored feedback value alone.
    """
    db.log_interaction("alice", "q1", "a good answer to an easy topic", "easy_topic", feedback=1, source="easy_lecture.pdf")
    db.log_interaction("alice", "q2", "a good answer to an easy topic", "easy_topic", feedback=1, source="easy_lecture.pdf")
    db.log_interaction("alice", "q3", "a good answer to an easy topic", "easy_topic", feedback=1, source="easy_lecture.pdf")

    weak = db.get_weak_topics("alice", threshold=3)
    # The document is classified as weak from the low ratings alone; the stored
    # data contains no information about the cause of those ratings.
    assert len(weak) == 1
    assert weak[0][0] == "easy_lecture.pdf"


def test_LIMITATION_no_undo_for_a_misclicked_rating(temp_db):
    """
    A rating can only be updated through the normal rating flow while the
    corresponding interaction remains the most recent interaction for the user.

    After a newer interaction is logged, the previous interaction is no longer
    the target of the normal rating mechanism, so an incorrect rating cannot be
    corrected through that interface.
    """
    db.log_interaction("alice", "Q1", "answer 1", "topic_a", feedback=None)
    interaction_id = db.get_last_interaction_id("alice")
    db.set_feedback(interaction_id, 1)  # misclick -- meant to click 5

    db.log_interaction("alice", "Q2", "answer 2", "topic_b", feedback=None)

    # After Q2 is logged, the rating mechanism targets Q2 rather than Q1.
    import sqlite3
    conn = sqlite3.connect(temp_db)
    c = conn.cursor()
    c.execute("SELECT query, feedback FROM interactions WHERE user_id=? ORDER BY id", ("alice",))
    rows = c.fetchall()
    conn.close()
    assert rows[0] == ("Q1", 1)
    assert rows[1] == ("Q2", None)