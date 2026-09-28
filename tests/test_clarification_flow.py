"""
Regression tests for answering the original question after document clarification.

When retrieval matches multiple documents, the application asks the learner which
document they mean. The learner's reply should select a document, not become the
question sent to the LLM. These tests verify that the original question is
preserved throughout clarification and subsequent retrieval.

Retrieval and LLM calls are mocked and recorded so the tests can verify exactly
which query, document filter, and prompt the engine uses.
"""
import sqlite3
import pytest
import app
import db
import engine

SOURCES = [
    "CM3050 Topic 5 Developing a Mobile App Project.pptx",
    "CM3020 Artificial Intelligence_lecture 1.pdf",
    "CM3010 Topic 3 Normalisation.pdf",
]
QUESTION = "Beyond catching mistakes, what other benefits of testing does the deck give?"


# Test harness ----------------------------------------------------------------

@pytest.fixture
def world(temp_db, monkeypatch):
    """
    Create a controlled retrieval/LLM environment for clarification tests.

    Unrestricted retrieval returns three equally plausible documents, while
    source-filtered retrieval returns chunks from the selected document.
    Retrieval and LLM calls are recorded for assertions about engine behaviour.
    """
    w = {"retrieval": [], "llm": [], "reply": "a real answer"}

    def retrieve(user_id, query, top_k=5, source_filter=None, **kw):
        w["retrieval"].append({"query": query, "source_filter": source_filter})
        if source_filter and not w.get("ignore_filter"):
            return [{"text": f"chunk {i}", "source": source_filter, "distance": 0.30 + i / 100} for i in range(6)]
        return [{"text": "x", "source": SOURCES[0], "distance": 0.300},
                {"text": "y", "source": SOURCES[1], "distance": 0.320},
                {"text": "z", "source": SOURCES[2], "distance": 0.340}]

    def chat(model, messages, **kw):
        w["llm"].append(messages[1]["content"])
        return {"message": {"content": w["reply"]}}

    monkeypatch.setattr(engine, "retrieve_with_sources", retrieve)
    monkeypatch.setattr(engine, "list_sources", lambda uid: list(SOURCES))
    monkeypatch.setattr(engine._client, "chat", chat)
    w["ask"] = lambda text, user="jun": engine.generate_answer_detailed(user, text)
    w["question_sent_to_llm"] = lambda: w["llm"][-1].split("Question: ")[-1]
    return w


def _clarify(world, user="jun"):
    """
    Create a pending clarification for the shared test question.
    """
    r = world["ask"](QUESTION, user)
    assert r["kind"] == "clarification", r
    return r


# Database: pending clarification state ---------------------------------------

def test_pending_clarification_round_trip(temp_db):
    db.set_pending_clarification("jun", QUESTION, SOURCES)
    got = db.get_pending_clarification("jun")
    assert (got["query"], got["candidates"]) == (QUESTION, SOURCES)


def test_no_pending_clarification_returns_none(temp_db):
    assert db.get_pending_clarification("nobody") is None


def test_a_new_clarification_replaces_an_unanswered_one(temp_db):
    db.set_pending_clarification("jun", "first question", ["a.pdf", "b.pdf"])
    db.set_pending_clarification("jun", "second question", ["c.pdf", "d.pdf"])
    got = db.get_pending_clarification("jun")
    assert (got["query"], got["candidates"]) == ("second question", ["c.pdf", "d.pdf"])


def test_pending_clarification_is_per_user_and_normalises_the_id(temp_db):
    db.set_pending_clarification("Jun ", QUESTION, SOURCES)
    assert db.get_pending_clarification("jun") is not None
    assert db.get_pending_clarification("someone_else") is None


def test_clear_pending_clarification(temp_db):
    db.set_pending_clarification("jun", QUESTION, SOURCES)
    db.set_pending_clarification("other", QUESTION, SOURCES)
    db.clear_pending_clarification("jun")
    assert db.get_pending_clarification("jun") is None
    assert db.get_pending_clarification("other") is not None


def _age(temp_db, seconds):
    conn = sqlite3.connect(temp_db)
    conn.execute("UPDATE pending_clarifications SET created_at = datetime('now', ?)", (f"-{seconds} seconds",))
    conn.commit()
    conn.close()


def test_an_old_pending_clarification_expires_and_is_deleted(temp_db):
    db.set_pending_clarification("jun", QUESTION, SOURCES)
    _age(temp_db, db.PENDING_CLARIFICATION_MAX_AGE_SECONDS + 60)
    assert db.get_pending_clarification("jun") is None
    conn = sqlite3.connect(temp_db)
    assert conn.execute("SELECT COUNT(*) FROM pending_clarifications").fetchone()[0] == 0
    conn.close()


def test_a_recent_pending_clarification_does_not_expire(temp_db):
    db.set_pending_clarification("jun", QUESTION, SOURCES)
    _age(temp_db, db.PENDING_CLARIFICATION_MAX_AGE_SECONDS - 60)
    assert db.get_pending_clarification("jun") is not None


def test_expiry_age_is_overridable(temp_db):
    db.set_pending_clarification("jun", QUESTION, SOURCES)
    _age(temp_db, 120)
    assert db.get_pending_clarification("jun", max_age_seconds=60) is None


def test_corrupt_candidates_are_treated_as_no_pending(temp_db):
    conn = sqlite3.connect(temp_db)
    conn.execute("INSERT INTO pending_clarifications (user_id, query, candidates) VALUES ('jun','q','{not json')")
    conn.commit()
    conn.close()
    assert db.get_pending_clarification("jun") is None


def test_resetting_materials_drops_a_pending_clarification(temp_db):
    db.set_pending_clarification("jun", QUESTION, SOURCES)
    db.clear_indexed_files("jun")
    assert db.get_pending_clarification("jun") is None


def test_init_db_adds_the_table_to_a_database_that_predates_it(temp_db):
    conn = sqlite3.connect(temp_db)
    conn.execute("DROP TABLE pending_clarifications")
    conn.commit()
    conn.close()
    db.init_db()
    db.set_pending_clarification("jun", QUESTION, SOURCES)
    assert db.get_pending_clarification("jun")["query"] == QUESTION


def test_a_new_pending_clarification_starts_with_zero_failed_attempts(temp_db):
    db.set_pending_clarification("jun", QUESTION, SOURCES)
    assert db.get_pending_clarification("jun")["attempts"] == 0


def test_bump_pending_attempts_counts_and_persists(temp_db):
    db.set_pending_clarification("jun", QUESTION, SOURCES)
    assert db.bump_pending_attempts("jun") == 1
    assert db.bump_pending_attempts("jun") == 2
    assert db.get_pending_clarification("jun")["attempts"] == 2


def test_replacing_a_clarification_resets_the_attempt_count(temp_db):
    db.set_pending_clarification("jun", QUESTION, SOURCES)
    db.bump_pending_attempts("jun")
    db.set_pending_clarification("jun", "a different question", SOURCES)
    assert db.get_pending_clarification("jun")["attempts"] == 0


def test_bump_with_nothing_pending_is_harmless(temp_db):
    assert db.bump_pending_attempts("nobody") == 0


def test_init_db_adds_the_attempts_column_to_a_table_that_lacks_it(temp_db):
    conn = sqlite3.connect(temp_db)
    conn.execute("DROP TABLE pending_clarifications")
    conn.execute("CREATE TABLE pending_clarifications (user_id TEXT PRIMARY KEY, query TEXT, candidates TEXT, "
                 "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
    conn.execute("INSERT INTO pending_clarifications (user_id, query, candidates) VALUES ('jun','q','[\"a.pdf\"]')")
    conn.commit()
    conn.close()
    db.init_db()
    db.init_db()
    assert db.get_pending_clarification("jun") == {"query": "q", "candidates": ["a.pdf"], "attempts": 0}


# Resolving a clarification reply -----------------------------------------------

@pytest.mark.parametrize("reply, expected", [
    ("1", 1), ("2", 2), ("3", 3), ("option 2", 2), ("Number 3", 3), ("no 1", 1),
    ("first", 1), ("the second one", 2), ("3rd", 3), ("the first document", 1),
    ("4", None), ("0", None), ("12", None), ("", None), (None, None),
    ("first second", None), ("banana", None), ("the one", None),
])
def test_ordinal_choice(reply, expected):
    assert engine._ordinal_choice(reply, 3) == expected


@pytest.mark.parametrize("reply, expected", [
    (SOURCES[0], SOURCES[0]),                                   # exact file name
    (SOURCES[0].upper(), SOURCES[0]),                           # case-insensitive
    ("CM3050", SOURCES[0]),                                     # course code alone
    ("cm3020", SOURCES[1]),
    ("mobile", SOURCES[0]),                                     # a distinctive word
    ("lecture 1", SOURCES[1]),
    ("normalisation", SOURCES[2]),
    ("mobile app", SOURCES[0]),
    ("CM3050 Topic 5", SOURCES[0]),                             # partial, still unique
    ("2", SOURCES[1]),                                          # ordinal against the numbered options
    ("the first one", SOURCES[0]),
])
def test_resolve_choice_accepts_ways_of_naming_a_document(reply, expected):
    assert engine._resolve_choice(reply, SOURCES) == expected


@pytest.mark.parametrize("reply", [
    "topic",                  # appears in multiple file names
    "the",                    # contains no identifying term
    "banana",                 # matches nothing
    "the mobile one",         # does not uniquely identify a document
    "",
    "CM30",                   # not a complete token
    "mobile normalisation",   # matches different documents
])
def test_resolve_choice_refuses_to_guess(reply):
    assert engine._resolve_choice(reply, SOURCES) is None


def test_resolve_choice_can_name_a_document_that_was_not_offered():
    offered = SOURCES[:2]
    assert engine._resolve_choice("normalisation", offered, all_sources=SOURCES) == SOURCES[2]


def test_a_number_refers_to_the_offered_options_not_to_all_documents():
    offered = [SOURCES[2], SOURCES[0]]
    assert engine._resolve_choice("1", offered, all_sources=SOURCES) == SOURCES[2]


@pytest.mark.parametrize("text, expected", [
    ("What is BCNF?", True),
    ("what is bcnf", True),
    ("Explain normalisation", True),
    ("Define a functional dependency", True),
    ("Is this covered in lecture 2", True),
    ("this one is about mobile apps and their testing frameworks", True),
    ("the mobile one", False),
    ("lecture 1", False),
    ("CM3050", False),
    ("the second", False),
    ("", False),
])
def test_looks_like_new_question(text, expected):
    assert engine._looks_like_new_question(text) is expected


def test_clarification_message_is_numbered_and_no_longer_suggests_continue():
    msg = engine._clarification_message(SOURCES)
    assert msg.startswith("🤔")
    for i, name in enumerate(SOURCES, start=1):
        assert f'{i}. "{name}"' in msg
    assert "continue" not in msg.lower(), (
        "'continue' is reserved for continuing the previous answered question"
    )


# Engine: reported clarification regression --------------------------------------

def test_clarification_remembers_the_original_question(world):
    r = _clarify(world)
    got = db.get_pending_clarification("jun")
    assert (got["query"], got["candidates"]) == (QUESTION, SOURCES)
    assert r["sources"] == SOURCES
    assert world["llm"] == [], "no LLM call for a clarification"
    assert db.get_last_interaction("jun") is None, "a clarification is still not a logged interaction"


def test_naming_the_document_answers_the_ORIGINAL_question(world):
    """
    Regression test: selecting a document must answer the original question.
    """
    _clarify(world)
    r = world["ask"](SOURCES[0])
    assert r["kind"] == "answer" and r["resolved_source"] == SOURCES[0]
    assert world["question_sent_to_llm"]() == QUESTION
    assert SOURCES[0] not in world["llm"][-1].split("Question: ")[-1]
    assert world["retrieval"][-1] == {"query": QUESTION, "source_filter": SOURCES[0]}


def test_retrieval_uses_the_original_question_not_the_file_name(world):
    """
    Document selection changes the retrieval filter, not the search query.
    """
    _clarify(world)
    world["ask"](SOURCES[0])
    assert world["retrieval"][-1]["query"] == QUESTION


@pytest.mark.parametrize("reply, chosen", [
    ("1", SOURCES[0]), ("2", SOURCES[1]), ("3", SOURCES[2]),
    ("the second one", SOURCES[1]),
    ("CM3010", SOURCES[2]),
    ("normalisation", SOURCES[2]),
    ("lecture 1", SOURCES[1]),
])
def test_every_way_of_naming_a_document_answers_the_original_question(world, reply, chosen):
    _clarify(world)
    r = world["ask"](reply)
    assert r["kind"] == "answer" and r["resolved_source"] == chosen
    assert world["question_sent_to_llm"]() == QUESTION
    assert world["retrieval"][-1]["source_filter"] == chosen


def test_the_resolved_answer_is_logged_under_the_original_question_and_is_rateable(world, temp_db):
    _clarify(world)
    r = world["ask"](SOURCES[0])
    assert r["rateable"] is True
    last = db.get_last_interaction("jun", rateable_only=True)
    assert last["query"] == QUESTION, "the file name must not be stored as the question"
    assert last["topic"] != "cm3050", "the topic is derived from the question, not the file name"


def test_the_pending_clarification_is_consumed_by_answering_it(world):
    _clarify(world)
    world["ask"](SOURCES[0])
    assert db.get_pending_clarification("jun") is None


def test_the_answer_shows_which_document_it_came_from(world):
    _clarify(world)
    assert world["ask"](SOURCES[0])["sources"] == [SOURCES[0]]


def test_a_resolved_question_is_not_asked_to_clarify_again(world):
    """
    If filtered retrieval falls back to unfiltered retrieval, the selected document
    must still be treated as resolved rather than triggering another clarification.
    """
    world["ignore_filter"] = True
    _clarify(world)
    r = world["ask"](SOURCES[1])
    assert r["kind"] == "answer" and r["resolved_source"] == SOURCES[1]


def test_a_later_continue_continues_the_original_question(world):
    _clarify(world)
    world["ask"](SOURCES[0])
    world["ask"]("continue")
    assert world["retrieval"][-1]["query"] == QUESTION


def test_if_the_model_refuses_the_resolved_question_it_is_not_rateable(world):
    _clarify(world)
    world["reply"] = "NOT_IN_MATERIALS"
    r = world["ask"](SOURCES[0])
    assert r["kind"] == "not_in_materials" and r["rateable"] is False
    assert db.get_pending_clarification("jun") is None


# Engine: replies that must not be guessed ------------------------------------

def test_an_ambiguous_reply_asks_again_and_keeps_the_question(world):
    _clarify(world)
    calls = len(world["retrieval"])
    r = world["ask"]("topic")
    assert r["kind"] == "clarification" and r["rateable"] is False
    assert world["llm"] == [] and len(world["retrieval"]) == calls
    assert db.get_pending_clarification("jun")["query"] == QUESTION


def test_an_unrecognised_short_reply_is_not_answered_as_a_question(world):
    """
    An unrecognised clarification reply must not be sent to the LLM as a new question.
    """
    _clarify(world)
    r = world["ask"]("the mobile one")
    assert r["kind"] == "clarification" and world["llm"] == []
    assert '1. "' in r["answer"], "the document options are shown again"
    assert db.get_pending_clarification("jun")["query"] == QUESTION


def test_the_learner_can_still_answer_after_a_failed_attempt(world):
    _clarify(world)
    world["ask"]("the mobile one")
    r = world["ask"]("1")
    assert r["kind"] == "answer" and world["question_sent_to_llm"]() == QUESTION


def test_a_new_question_abandons_the_pending_one(world, monkeypatch):
    _clarify(world)
    # Make the new question unambiguous so it can proceed without clarification.
    monkeypatch.setattr(engine, "retrieve_with_sources",
                        lambda uid, q, top_k=5, source_filter=None, **kw:
                        world["retrieval"].append({"query": q, "source_filter": source_filter}) or
                        [{"text": "c", "source": SOURCES[2], "distance": 0.1}])
    r = world["ask"]("What is BCNF and why does it matter?")
    assert r["kind"] == "answer" and r["resolved_source"] is None
    assert world["question_sent_to_llm"]() == "What is BCNF and why does it matter?"
    assert db.get_pending_clarification("jun") is None


def test_a_new_ambiguous_question_replaces_the_pending_one(world):
    _clarify(world)
    newer = "What are the main advantages of automated testing frameworks in practice?"
    r = world["ask"](newer)
    assert r["kind"] == "clarification"
    assert db.get_pending_clarification("jun")["query"] == newer


def test_the_first_unrecognised_reply_is_counted(world):
    _clarify(world)
    world["ask"]("the mobile one")
    assert db.get_pending_clarification("jun")["attempts"] == 1


def test_a_second_unrecognised_reply_is_treated_as_a_new_question(world, monkeypatch):
    """
    After one failed clarification attempt, a second unrecognised message is
    Without this a short keyword-style NEW question ('bcnf definition'), no question
    treated as a new question so the learner cannot remain stuck in clarification.
    """
    _clarify(world)
    # Make the same text unambiguous on the second attempt so it can be answered.
    assert world["ask"]("bcnf definition")["kind"] == "clarification"
    monkeypatch.setattr(engine, "retrieve_with_sources",
                        lambda uid, q, top_k=5, source_filter=None, **kw:
                        world["retrieval"].append({"query": q, "source_filter": source_filter}) or
                        [{"text": "c", "source": SOURCES[2], "distance": 0.1}])
    r = world["ask"]("bcnf definition")
    assert r["kind"] == "answer"
    assert world["question_sent_to_llm"]() == "bcnf definition"
    assert db.get_pending_clarification("jun") is None


def test_giving_up_abandons_the_old_question_even_when_the_new_one_is_ambiguous_too(world):
    _clarify(world)
    world["ask"]("bcnf definition")
    world["ask"]("bcnf definition")
    assert db.get_pending_clarification("jun")["query"] == "bcnf definition"


def test_the_choice_still_works_after_one_failed_attempt_but_not_after_giving_up(world):
    _clarify(world)
    world["ask"]("the mobile one")
    assert world["ask"]("1")["kind"] == "answer"

    world["llm"].clear()
    _clarify(world)
    world["ask"]("the mobile one")
    world["ask"]("the mobile one")                       
    # After the second failed attempt, the old question is discarded. The 
    # next choice therefore belongs to the newly treated question.
    world["ask"]("1")
    assert world["question_sent_to_llm"]() == "the mobile one"
    assert QUESTION not in world["llm"][-1], "the original question must no longer be answered"


def test_a_bookkeeping_failure_while_counting_attempts_still_re_asks(world, monkeypatch):
    monkeypatch.setattr(engine, "bump_pending_attempts", lambda uid: (_ for _ in ()).throw(RuntimeError("db locked")))
    _clarify(world)
    r = world["ask"]("the mobile one")
    assert r["kind"] == "clarification" and world["llm"] == []


def test_at_least_one_re_ask_is_allowed():
    assert engine.MAX_CLARIFICATION_REASKS >= 1


def test_a_resolved_choice_is_never_counted_as_a_failed_attempt(world):
    _clarify(world)
    world["ask"]("1")
    assert db.get_pending_clarification("jun") is None


def test_continue_while_a_clarification_is_pending_means_the_previous_answer(world):
    db.log_interaction("jun", "What is a functional dependency, and how is it written?", "a", "functional dependency")
    _clarify(world)
    r = world["ask"]("continue")
    assert r["kind"] == "continuation"
    assert world["retrieval"][-1]["query"] == "What is a functional dependency, and how is it written?"
    assert db.get_pending_clarification("jun") is None


def test_an_expired_clarification_is_ignored(world, temp_db):
    _clarify(world)
    _age(temp_db, db.PENDING_CLARIFICATION_MAX_AGE_SECONDS + 60)
    r = world["ask"](SOURCES[0])
    assert r["resolved_source"] is None, "a stale clarification must not be answered"
    assert world["question_sent_to_llm"]() == SOURCES[0], "treated as an ordinary message, as before"


def test_a_pending_clarification_belongs_to_one_user(world):
    _clarify(world, user="alice")
    r = world["ask"](SOURCES[0], user="bob")
    assert r["resolved_source"] is None
    assert db.get_pending_clarification("alice") is not None


def test_without_any_pending_clarification_behaviour_is_unchanged(world):
    r = world["ask"]("What is a functional dependency?", user="fresh")
    assert r["kind"] == "clarification"
    assert db.get_pending_clarification("fresh")["query"] == "What is a functional dependency?"


def test_a_failing_pending_lookup_never_stops_answering(world, monkeypatch):
    def boom(user_id, *a, **k):
        raise RuntimeError("db locked")
    monkeypatch.setattr(engine, "get_pending_clarification", boom)
    assert world["ask"](SOURCES[0])["kind"] in ("answer", "clarification")


def test_a_failing_pending_write_never_stops_asking(world, monkeypatch):
    monkeypatch.setattr(engine, "set_pending_clarification", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db locked")))
    assert world["ask"](QUESTION)["kind"] == "clarification"


# App: conversation cleanup ----------------------------------------------------

def _pending(user="jun"):
    db.set_pending_clarification(user, QUESTION, SOURCES)


def test_clear_chat_empties_the_chat_releases_the_gate_and_drops_the_pending_question(temp_db):
    _pending()
    chat, panel, awaiting, status = app.clear_chat("jun")
    assert chat == [] and panel["visible"] is False and awaiting is False and status == ""
    assert db.get_pending_clarification("jun") is None


def test_the_trash_icon_hook_drops_the_pending_question_too(temp_db):
    _pending()
    panel, awaiting, status = app.release_gate("jun")
    assert awaiting is False and db.get_pending_clarification("jun") is None


def test_release_gate_still_works_with_no_user_argument(temp_db):
    _pending()
    panel, awaiting, status = app.release_gate()
    assert awaiting is False and db.get_pending_clarification("jun") is not None, \
        "with no user id there is nothing to clear, and it must not fail"


def test_switching_user_drops_the_pending_question(temp_db):
    _pending()
    app.go_back_to_login("jun")
    assert db.get_pending_clarification("jun") is None


def test_logging_in_starts_with_no_pending_question(temp_db):
    _pending()
    app.handle_login("jun")
    assert db.get_pending_clarification("jun") is None


def test_clearing_one_users_chat_leaves_another_users_pending_question(temp_db):
    _pending("jun")
    _pending("someone")
    app.clear_chat("jun")
    assert db.get_pending_clarification("someone") is not None


@pytest.mark.parametrize("bad", [None, "", "bad;id"])
def test_clean_up_tolerates_an_invalid_user_id(temp_db, bad):
    app.clear_chat(bad)
    app.release_gate(bad)
    app.go_back_to_login(bad)


def test_a_resolved_clarification_opens_the_rating_gate_like_any_answer(world, monkeypatch):
    """
    A resolved clarification produces a normal rateable answer.
    """
    _clarify(world)
    out = app.respond(SOURCES[0], [], "jun", None, False)
    assert out[0] == "a real answer" and out[3] is True and out[2]["visible"] is True


def test_the_clarification_prompt_itself_never_opens_the_gate(world):
    out = app.respond(QUESTION, [], "jun", None, False)
    assert out[0].startswith("🤔") and out[3] is False


# wiring ------------------------------------------------------------------

def _fn_name(bf):
    f = getattr(bf.fn, "func", bf.fn)
    return getattr(f, "__name__", None)


@pytest.mark.parametrize("name", ["clear_chat", "release_gate", "go_back_to_login"])
def test_cleanup_handlers_are_wired_to_receive_the_current_user(name):
    """
    Cleanup handlers need the current user ID to clear user-specific state.
    """
    fns = [bf for bf in app.demo.fns.values() if _fn_name(bf) == name]
    assert fns, f"{name} is not wired"
    assert all(len(bf.inputs) == 1 for bf in fns), f"{name} is wired without the user id"


def test_clear_chat_arity_matches_its_outputs():
    declared = {len(bf.outputs) for bf in app.demo.fns.values() if _fn_name(bf) == "clear_chat"}
    assert declared == {len(app.clear_chat("x"))} == {4}
