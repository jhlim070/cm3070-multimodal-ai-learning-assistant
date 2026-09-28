"""
Testing the following features: profile lock, duplicate 
rejection, rating-gate fixes, retrieval / prompt changes.

The Ollama client is mocked so these tests verify application logic
and integration behaviour rather than model-generated answer quality.
ChromaDB is exercised against a temporary test directory to verify real
collection and retrieval behaviour without affecting persistent data.

chromaDB tests are structural rather than retrieval-quality tests.
They verify collection naming, source filtering, and returned metadata,
while remaining independent of the embedding model's retrieval quality.
"""
import sqlite3

import pytest

import app
import db
import engine
import rag
import validation

# Helpers -------------------------------------------------------------------

class _FakeFile:
    def __init__(self, name):
        self.name = name


def _pdf(tmp_path, name="lecture.pdf", body=b"body"):
    """
    Create a minimal PDF-like file for upload validation tests.
    """
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4\n" + body)
    return _FakeFile(str(p))


def _res(answer="an answer", kind="answer", rateable=True, sources=None):
    return {"answer": answer, "kind": kind, "rateable": rateable,
            "sources": sources or [], "model": "llama3.2"}


def _rows(temp_db, columns="query, answer, feedback, rateable"):
    conn = sqlite3.connect(temp_db)
    rows = conn.execute(f"SELECT {columns} FROM interactions ORDER BY id").fetchall()
    conn.close()
    return rows


@pytest.fixture
def stub_indexing(monkeypatch):
    """
    Stub the expensive indexing operations while preserving 
    validation, hashing, and indexed-file registry behaviour.

    """
    calls = {"process": [], "index": []}

    def fake_process(path):
        calls["process"].append(path)
        return "text"

    def fake_add(user_id, chunks, metadatas=None, **kw):
        calls["index"].append((user_id, list(chunks), metadatas))

    monkeypatch.setattr(app, "process_file", fake_process)
    monkeypatch.setattr(app, "chunk_text", lambda text, **kwargs: ["c1", "c2", "c3"])
    monkeypatch.setattr(app, "add_documents", fake_add)
    return calls


@pytest.fixture
def no_source_lookup(monkeypatch):
    monkeypatch.setattr(engine, "list_sources", lambda user_id: [])


# db: indexed-file registry --------------------------------------------------
def test_registry_detects_duplicate_by_content_hash(temp_db):
    db.record_indexed_file("alice", "L1.pdf", "hash-a", 41)
    hit = db.is_file_indexed("alice", sha256="hash-a", filename="a_different_name.pdf")
    assert hit and hit["filename"] == "L1.pdf" and hit["chunks"] == 41


def test_registry_detects_duplicate_by_filename(temp_db):
    db.record_indexed_file("alice", "L1.pdf", "hash-a", 41)
    hit = db.is_file_indexed("alice", sha256="hash-b", filename="L1.pdf")
    assert hit and hit["sha256"] == "hash-a"


def test_registry_returns_none_for_a_genuinely_new_file(temp_db):
    db.record_indexed_file("alice", "L1.pdf", "hash-a", 41)
    assert db.is_file_indexed("alice", sha256="hash-z", filename="new.pdf") is None


def test_registry_is_per_user(temp_db):
    db.record_indexed_file("alice", "L1.pdf", "hash-a", 41)
    assert db.is_file_indexed("bob", sha256="hash-a", filename="L1.pdf") is None


def test_registry_normalises_user_id_like_the_rest_of_db(temp_db):
    db.record_indexed_file("Alice ", "L1.pdf", "hash-a", 41)
    assert db.is_file_indexed("alice", sha256="hash-a")


def test_registry_list_and_clear(temp_db):
    db.record_indexed_file("alice", "a.pdf", "h1", 1)
    db.record_indexed_file("alice", "b.pdf", "h2", 2)
    db.record_indexed_file("bob", "c.pdf", "h3", 3)
    assert [r["filename"] for r in db.list_indexed_files("alice")] == ["a.pdf", "b.pdf"]
    assert db.clear_indexed_files("alice") == 2
    assert db.list_indexed_files("alice") == []
    assert len(db.list_indexed_files("bob")) == 1, "clearing one user must not touch another"


def test_registry_is_not_matched_by_empty_defaults(temp_db):
    """
    A lookup without identifying fields must not match an existing row.
    """
    db.record_indexed_file("alice", "a.pdf", "h1", 1)
    assert db.is_file_indexed("alice") is None


# db: rateable interactions

def test_log_interaction_defaults_to_rateable(temp_db):
    db.log_interaction("alice", "q", "a", "t")
    assert _rows(temp_db)[0][3] == 1


def test_last_rateable_id_skips_non_rateable_rows(temp_db):
    db.log_interaction("alice", "real q", "real a", "t")               # id 1, rateable
    db.log_interaction("alice", "q2", "refusal", "t", rateable=0)      # id 2, not
    assert db.get_last_interaction_id("alice") == 2
    assert db.get_last_rateable_interaction_id("alice") == 1


def test_last_rateable_id_is_none_when_nothing_rateable(temp_db):
    db.log_interaction("alice", "q", "refusal", "t", rateable=0)
    assert db.get_last_rateable_interaction_id("alice") is None


def test_get_last_interaction_returns_dict_and_honours_rateable_only(temp_db):
    db.log_interaction("alice", "first", "a1", "t1")
    db.log_interaction("alice", "second", "a2", "t2", rateable=0)
    assert db.get_last_interaction("alice")["query"] == "second"
    assert db.get_last_interaction("alice", rateable_only=True)["query"] == "first"
    assert db.get_last_interaction("nobody") is None


def test_init_db_migrates_an_old_schema_without_losing_rows(temp_db):
    """
    Verify that databases created before the rateable field was introduced
    are migrated without losing existing interaction records.
    """
    conn = sqlite3.connect(temp_db)
    conn.execute("DROP TABLE interactions")
    conn.execute("""CREATE TABLE interactions
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                     query TEXT, answer TEXT, topic TEXT, feedback INTEGER)""")
    conn.execute("INSERT INTO interactions (user_id, query, answer, topic, feedback) "
                 "VALUES ('bob','old q','old a','t',4)")
    conn.commit()
    conn.close()

    db.init_db()
    db.init_db()  # Initialisation must be safe to repeat.

    old = db.get_last_interaction("bob")
    assert old["query"] == "old q" and old["feedback"] == 4
    assert db.get_last_rateable_interaction_id("bob") == old["id"], \
        "rows written before the migration must default to rateable"


def test_weak_topics_are_not_polluted_by_unrated_non_answers(temp_db):
    """
    Non-rateable responses must not contribute to weak-topic detection.
    """
    for _ in range(3):
        db.log_interaction("alice", "what is 5nf", "refusal", "5nf", rateable=0, source="lecture1.pdf")
    # Provide a source so the rows participate in weak-topic aggregation.
    assert db.get_weak_topics("alice", threshold=3) == []


# engine: continuation -------------------------------------------------------

@pytest.mark.parametrize("text", ["continue", "Continue", "CONTINUE.", "  go on  ", "keep going",
                                  "tell me more", "Please continue!", "carry on", "more"])
def test_is_continuation_true(text):
    assert engine._is_continuation(text)


@pytest.mark.parametrize("text", ["", "what is 5NF", "continue the normalisation lecture from slide 4",
                                  "I want to continue learning about SQL joins", "moreover", None])
def test_is_continuation_false(text):
    assert not engine._is_continuation(text)


# Engine: naming a document in the question ---------------------------------

SOURCES = [
    "CM3020 Artificial Intelligence_lecture 1.pdf",
    "CM3020 Artificial Intelligence_lecture 2.pdf",
    "CM3010 Normalisation.pdf",
]


def test_match_sources_picks_the_named_lecture():
    q = "For CM3020 Artificial intelligence lecture 1, What are the four approaches to AI?"
    assert engine._match_sources(q, SOURCES) == SOURCES[0]


def test_match_sources_distinguishes_lecture_2_from_lecture_1():
    assert engine._match_sources("In CM3020 lecture 2, what is a DQN agent?", SOURCES) == SOURCES[1]


def test_match_sources_does_nothing_when_no_document_is_named():
    assert engine._match_sources("What are the four approaches to AI?", SOURCES) is None


def test_match_sources_refuses_to_guess_between_tied_documents():
    """
    A course-level match is ambiguous when multiple lectures match; the
    implementation must request clarification rather than select one.
    """
    assert engine._match_sources("For CM3020 Artificial Intelligence, what are the approaches?", SOURCES) is None


def test_match_sources_needs_more_than_one_matching_token():
    assert engine._match_sources("tell me about artificial things", SOURCES) is None


def test_match_sources_handles_empty_inputs():
    assert engine._match_sources("", SOURCES) is None
    assert engine._match_sources("anything at all", []) is None
    assert engine._match_sources(None, SOURCES) is None


# engine: non-answer detection ------------------------------------------------

@pytest.mark.parametrize("text", [
    "NOT_IN_MATERIALS",
    "  not_in_materials  ",
    "Unfortunately, the provided context does not contain the answer to your question.",
    "I couldn't find any information about that in the context.",
    "I don't see the provided context for Lecture 1.",
    "",
])
def test_looks_like_non_answer_true(text):
    assert engine._looks_like_non_answer(text)


@pytest.mark.parametrize("text", [
    "The four approaches are thinking humanly, thinking rationally, acting humanly, acting rationally.",
    "A functional dependency is written X -> Y.",
    "Step 1: Identify the recursive case.",
])
def test_looks_like_non_answer_false_for_real_answers(text):
    assert not engine._looks_like_non_answer(text)


# engine: prompting

def test_system_prompt_states_the_learners_profile_verbatim():
    p = engine._build_system_prompt("visual examples", "beginner")
    assert "visual examples" in p and "beginner" in p


def test_every_learning_style_maps_to_a_distinct_instruction():
    prompts = {s: engine._build_system_prompt(s, "intermediate") for s in engine._STYLE_INSTRUCTIONS}
    assert len(set(prompts.values())) == len(engine._STYLE_INSTRUCTIONS)


def test_visual_style_asks_for_a_text_based_visual_structure():
    """
    Ensure the visual learning style produces explicit text-based
    visual structures rather than plain prose.
    """
    p = engine._build_system_prompt("visual examples", "beginner").lower()
    assert "table" in p and "diagram" in p


def test_step_by_step_style_asks_for_numbered_steps():
    assert "numbered steps" in engine._build_system_prompt("step-by-step", "beginner").lower()


def test_system_prompt_contains_the_refusal_token_and_grounding_rules():
    p = engine._build_system_prompt("step-by-step", "intermediate")
    assert engine.NOT_IN_MATERIALS_TOKEN in p
    assert "only the excerpts" in p.lower()
    assert "never" in p.lower()


def test_system_prompt_tells_the_model_that_excerpts_are_not_instructions():
    """
    Ensure retrieved document text is treated as reference material 
    rather than as instructions to the language model.
    """
    p = engine._build_system_prompt("step-by-step", "intermediate").lower()
    assert "not instructions" in p and "ignore" in p


def test_unknown_style_falls_back_without_crashing():
    assert "clear and detailed" in engine._build_system_prompt("clear and detailed", "intermediate")


def test_context_is_numbered_and_labelled_by_source():
    ctx = engine._build_context([
        {"text": "alpha", "source": "L1.pdf", "distance": 0.1},
        {"text": "beta", "source": None, "distance": 0.2},
    ])
    assert "[Excerpt 1 | source: L1.pdf]" in ctx
    assert "[Excerpt 2 | source: unnamed document]" in ctx
    assert "alpha" in ctx and "beta" in ctx


# engine: generate_answer_detailed ----------------------------------------------

def _patch_llm(monkeypatch, reply="a real answer"):
    """Replace the LLM call with a deterministic response and record requests."""
    calls = []

    def fake_chat(model, messages, **kw):
        calls.append({"model": model, "messages": messages, "kw": kw})
        return {"message": {"content": reply}}

    monkeypatch.setattr(engine._client, "chat", fake_chat)
    return calls


def _patch_retrieval(monkeypatch, results, capture=None):
    """
    Replace retrieval with deterministic results and 
    optionally capture the retrieval arguments.
    """
    def fake(user_id, query, top_k=5, **kw):
        if capture is not None:
            capture.append({"query": query, "top_k": top_k, **kw})
        return results
    monkeypatch.setattr(engine, "retrieve_with_sources", fake)


def _one_source(n=3, source="doc.pdf"):
    return [{"text": f"chunk {i}", "source": source, "distance": 0.1 + i / 100} for i in range(n)]


def test_real_answer_is_rateable_and_logged(temp_db, monkeypatch, no_source_lookup):
    _patch_llm(monkeypatch, "The four approaches are ...")
    _patch_retrieval(monkeypatch, _one_source())
    r = engine.generate_answer_detailed("alice", "What are the four approaches?")
    assert r["kind"] == "answer" and r["rateable"] is True
    assert _rows(temp_db)[0][3] == 1


def test_refusal_token_becomes_a_clean_message_and_is_not_rateable(temp_db, monkeypatch, no_source_lookup):
    _patch_llm(monkeypatch, "NOT_IN_MATERIALS")
    _patch_retrieval(monkeypatch, _one_source())
    r = engine.generate_answer_detailed("alice", "What is 5NF?")
    assert r["kind"] == "not_in_materials" and r["rateable"] is False
    assert r["answer"] == engine.NOT_IN_MATERIALS_MESSAGE
    assert _rows(temp_db)[0][3] == 0


def test_free_text_refusal_is_caught_by_the_fallback_detector(temp_db, monkeypatch, no_source_lookup):
    _patch_llm(monkeypatch, "Unfortunately, the provided context does not contain the answer.")
    _patch_retrieval(monkeypatch, _one_source())
    r = engine.generate_answer_detailed("alice", "anything")
    assert r["kind"] == "not_in_materials" and r["rateable"] is False


def test_raw_model_output_is_what_gets_logged_not_the_substituted_message(temp_db, monkeypatch, no_source_lookup):
    """
    The raw model response must be retained in the database even when the
    user-facing response is replaced by the standard refusal message.
    """
    raw = "The context does not contain 5NF. However 5NF is like 2NF (a wrong claim)."
    _patch_llm(monkeypatch, raw)
    _patch_retrieval(monkeypatch, _one_source())
    engine.generate_answer_detailed("alice", "What is 5NF?")
    assert _rows(temp_db)[0][1] == raw


def test_empty_index_gives_a_helpful_non_rateable_message(temp_db, monkeypatch, no_source_lookup):
    llm = _patch_llm(monkeypatch)
    _patch_retrieval(monkeypatch, [])
    r = engine.generate_answer_detailed("alice", "anything")
    assert r["rateable"] is False and "Nothing is indexed" in r["answer"]
    assert llm == [], "Empty retrieval results must not trigger an LLM call."
    assert _rows(temp_db) == []


def test_ollama_error_is_reported_not_logged_and_not_rateable(temp_db, monkeypatch, no_source_lookup):
    def boom(model, messages, **kw):
        raise TimeoutError("simulated outage")
    monkeypatch.setattr(engine._client, "chat", boom)
    _patch_retrieval(monkeypatch, _one_source())
    r = engine.generate_answer_detailed("alice", "q")
    assert r["kind"] == "error" and r["rateable"] is False
    assert "simulated outage" in r["answer"]
    assert _rows(temp_db) == []


@pytest.mark.parametrize("bad", ["", "   ", "x" * (engine.ENGINE_QUERY_CHAR_LIMIT + 1)])
def test_invalid_queries_never_reach_retrieval_or_the_llm(temp_db, monkeypatch, no_source_lookup, bad):
    llm = _patch_llm(monkeypatch)
    seen = []
    _patch_retrieval(monkeypatch, _one_source(), capture=seen)
    r = engine.generate_answer_detailed("alice", bad)
    assert r["kind"] == "invalid" and r["rateable"] is False
    assert llm == [] and seen == [] and _rows(temp_db) == []


def test_llm_is_called_with_pinned_generation_options(temp_db, monkeypatch, no_source_lookup):
    """
    Pin the context window to prevent long prompts from being 
    truncated before all selected excerpts reach the model.
    """
    llm = _patch_llm(monkeypatch)
    _patch_retrieval(monkeypatch, _one_source())
    engine.generate_answer_detailed("alice", "q")
    opts = llm[0]["kw"]["options"]
    assert opts["num_ctx"] >= 8192
    assert opts["temperature"] <= 0.2
    assert opts == engine.GENERATION_OPTIONS


def test_retrieval_is_wide_but_only_context_chunks_reach_the_prompt(temp_db, monkeypatch, no_source_lookup):
    llm = _patch_llm(monkeypatch)
    seen = []
    _patch_retrieval(monkeypatch, _one_source(n=engine.RETRIEVAL_TOP_K), capture=seen)
    engine.generate_answer_detailed("alice", "q")
    assert seen[0]["top_k"] == engine.RETRIEVAL_TOP_K
    user_msg = llm[0]["messages"][1]["content"]
    assert user_msg.count("[Excerpt ") == engine.CONTEXT_CHUNKS


def test_context_window_was_three_chunks(temp_db):
    """
    Ensure the context window remains larger than the previous three-chunk
    limit and does not exceed the retrieval candidate count.
    """
    assert engine.CONTEXT_CHUNKS > 3
    assert engine.RETRIEVAL_TOP_K >= engine.CONTEXT_CHUNKS


def test_a_named_document_restricts_retrieval_to_that_source(temp_db, monkeypatch):
    _patch_llm(monkeypatch)
    monkeypatch.setattr(engine, "list_sources", lambda user_id: SOURCES)
    seen = []
    # Similar results from multiple lectures would otherwise be ambiguous.
    _patch_retrieval(monkeypatch, [
        {"text": "a", "source": SOURCES[0], "distance": 0.10},
        {"text": "b", "source": SOURCES[1], "distance": 0.11},
    ], capture=seen)
    r = engine.generate_answer_detailed(
        "alice", "For CM3020 Artificial intelligence lecture 1, what are the four approaches to AI?")
    assert seen[0]["source_filter"] == SOURCES[0]
    assert r["kind"] == "answer", "An explicit document reference should resolve the ambiguity."


def test_no_named_document_means_no_source_filter(temp_db, monkeypatch):
    _patch_llm(monkeypatch)
    monkeypatch.setattr(engine, "list_sources", lambda user_id: SOURCES)
    seen = []
    _patch_retrieval(monkeypatch, _one_source(), capture=seen)
    engine.generate_answer_detailed("alice", "What is a functional dependency?")
    assert seen[0]["source_filter"] is None


def test_source_lookup_failure_never_breaks_answering(temp_db, monkeypatch):
    _patch_llm(monkeypatch)

    def boom(user_id):
        raise RuntimeError("chroma exploded")

    monkeypatch.setattr(engine, "list_sources", boom)
    _patch_retrieval(monkeypatch, _one_source())
    assert engine.generate_answer_detailed("alice", "q")["kind"] == "answer"


#  Continuation, end to end through the engine -------------------------------

def test_continue_with_no_history_asks_for_a_full_question(temp_db, monkeypatch, no_source_lookup):
    llm = _patch_llm(monkeypatch)
    _patch_retrieval(monkeypatch, _one_source())
    r = engine.generate_answer_detailed("alice", "continue")
    assert r["kind"] == "clarification" and r["rateable"] is False
    assert llm == []


def test_continue_retrieves_on_the_previous_question_not_the_word_continue(temp_db, monkeypatch, no_source_lookup):
    _patch_llm(monkeypatch)
    seen = []
    _patch_retrieval(monkeypatch, _one_source(), capture=seen)
    engine.generate_answer_detailed("alice", "What is separation of concerns?")
    engine.generate_answer_detailed("alice", "continue")
    assert seen[0]["query"] == "What is separation of concerns?"
    assert seen[1]["query"] == "What is separation of concerns?", \
        "'continue' provides no topical query terms, so retrieval must use the previous question."


def test_answer_to_continue_is_not_rateable_and_does_not_gate(temp_db, monkeypatch, no_source_lookup):
    """
    Continuation responses are informational and must not require feedback.
    """
    _patch_llm(monkeypatch, "more detail")
    _patch_retrieval(monkeypatch, _one_source())
    engine.generate_answer_detailed("alice", "What is separation of concerns?")
    r = engine.generate_answer_detailed("alice", "continue")
    assert r["kind"] == "continuation" and r["rateable"] is False
    assert [row[3] for row in _rows(temp_db)] == [1, 0]


def test_continue_skips_the_ambiguity_check_so_it_cannot_loop(temp_db, monkeypatch, no_source_lookup):
    """
    A continuation request must not repeat the ambiguity clarification,
    otherwise the continuation flow could enter a loop.
    """
    llm = _patch_llm(monkeypatch)
    _patch_retrieval(monkeypatch, _one_source())
    engine.generate_answer_detailed("alice", "What is normalisation?")

    _patch_retrieval(monkeypatch, [
        {"text": "a", "source": "x.pdf", "distance": 0.10},
        {"text": "b", "source": "y.pdf", "distance": 0.11},
    ])
    r = engine.generate_answer_detailed("alice", "continue")
    assert r["kind"] == "continuation"
    assert len(llm) == 2


def test_a_second_continue_still_resolves_to_the_original_question(temp_db, monkeypatch, no_source_lookup):
    _patch_llm(monkeypatch)
    seen = []
    _patch_retrieval(monkeypatch, _one_source(), capture=seen)
    engine.generate_answer_detailed("alice", "What is BCNF?")
    engine.generate_answer_detailed("alice", "continue")
    engine.generate_answer_detailed("alice", "continue")
    assert [s["query"] for s in seen] == ["What is BCNF?"] * 3


# app: rating gate

def test_only_a_real_answer_opens_the_gate(monkeypatch):
    for kind, rateable, expect_gate in [
        ("answer", True, True),
        ("continuation", False, False),
        ("not_in_materials", False, False),
        ("clarification", False, False),
        ("error", False, False),
        ("invalid", False, False),
    ]:
        monkeypatch.setattr(app, "process_image_and_text_detailed",
                            lambda u, m, i, k=kind, r=rateable: _res(kind=k, rateable=r))
        _, _, panel, awaiting = app.respond("a question", [], "alice", None, False)
        assert awaiting is expect_gate, kind
        assert panel["visible"] is expect_gate, kind


def test_gate_does_not_depend_on_emoji_prefixes_any_more(monkeypatch):
    """
    Gate behaviour must depend on the response classification rather 
    than presentation characters such as an emoji prefix.
    """
    monkeypatch.setattr(app, "process_image_and_text_detailed",
                        lambda u, m, i: _res("⚠️ Warning: this is a REAL answer about warnings", kind="answer", rateable=True))
    _, _, _, awaiting = app.respond("q", [], "alice", None, False)
    assert awaiting is True


def test_sources_are_shown_for_answers_and_continuations_only(monkeypatch):
    for kind, expect in [("answer", True), ("continuation", True), ("not_in_materials", False)]:
        monkeypatch.setattr(app, "process_image_and_text_detailed",
                            lambda u, m, i, k=kind: _res(kind=k, rateable=(k == "answer"), sources=["L1.pdf"]))
        _, status, _, _ = app.respond("q", [], "alice", None, False)
        assert ("L1.pdf" in status) is expect, kind


def test_rating_lands_on_the_answer_not_on_a_later_non_answer(temp_db):
    # Rateable interaction.
    db.log_interaction("alice", "real q", "real a", "t") 
    # Non-rateable response must not receive the rating.
    db.log_interaction("alice", "next q", "refusal", "t", rateable=0)
    assert "Logged" in app.submit_feedback("alice", 4)
    assert [r[2] for r in _rows(temp_db)] == [4, None]


def test_rating_with_nothing_rateable_is_refused(temp_db):
    db.log_interaction("alice", "q", "refusal", "t", rateable=0)
    assert app.submit_feedback("alice", 5).startswith("!!!")
    assert _rows(temp_db)[0][2] is None


@pytest.mark.parametrize("bad", [0, 6, -1, "5", None, 4.5])
def test_invalid_rating_values_are_rejected(temp_db, bad):
    db.log_interaction("alice", "q", "a", "t")
    assert app.submit_feedback("alice", bad).startswith("!!!")
    assert _rows(temp_db)[0][2] is None


def test_rating_with_a_malformed_user_id_is_rejected(temp_db):
    assert app.submit_feedback("bad;id", 3).startswith("!!!")


def test_release_gate_clears_the_gate_without_recording_a_rating(temp_db):
    db.log_interaction("alice", "q", "a", "t")
    panel, awaiting, status = app.release_gate()
    assert awaiting is False and panel["visible"] is False and status == ""
    assert _rows(temp_db)[0][2] is None, "Releasing the gate must not create feedback."


def test_gate_message_mentions_clear_chat_as_an_escape(monkeypatch):
    called = []
    monkeypatch.setattr(app, "process_image_and_text_detailed", lambda u, m, i: called.append(1) or _res())
    msg, _, panel, awaiting = app.respond("q", [], "alice", None, True)
    assert "rate" in msg.lower() and "clear chat" in msg.lower()
    assert awaiting is True and panel["visible"] is True and called == []


# app: chat input validation -----------------------------------------------

def _boom(*a, **k):
    raise AssertionError("pipeline must not run for invalid input")


def test_overlong_question_is_rejected_before_the_pipeline(monkeypatch):
    monkeypatch.setattr(app, "process_image_and_text_detailed", _boom)
    msg, _, panel, awaiting = app.respond("x" * 3000, [], "alice", None, False)
    assert msg.startswith("⚠️") and awaiting is False and panel["visible"] is False


def test_blank_question_is_rejected_before_the_pipeline(monkeypatch):
    monkeypatch.setattr(app, "process_image_and_text_detailed", _boom)
    assert app.respond("   ", [], "alice", None, False)[0].startswith("⚠️")


def test_malformed_user_id_is_rejected_in_chat(monkeypatch):
    monkeypatch.setattr(app, "process_image_and_text_detailed", _boom)
    assert app.respond("hi", [], "bad;user", None, False)[0].startswith("⚠️")


def test_invalid_input_does_not_release_a_pending_gate(monkeypatch):
    """
    Invalid input must not alter an existing pending rating gate.
    """
    monkeypatch.setattr(app, "process_image_and_text_detailed", _boom)
    _, _, panel, awaiting = app.respond("x" * 3000, [], "alice", None, True)
    assert awaiting is True and panel["visible"] is True


def test_nonexistent_or_fake_chat_image_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "process_image_and_text_detailed", _boom)
    assert app.respond("hi", [], "alice", str(tmp_path / "ghost.png"), False)[0].startswith("⚠️")
    fake = tmp_path / "fake.png"
    fake.write_bytes(b"not a png")
    assert app.respond("hi", [], "alice", str(fake), False)[0].startswith("⚠️")


def test_recommendations_reject_a_malformed_user_id(monkeypatch):
    monkeypatch.setattr(app, "generate_recommendations", _boom)
    assert app.get_recommendations_btn("bad;id").startswith("!!!")


# app: login and profile lock --------------------------------------------

def test_login_rejects_a_malformed_user_id_and_keeps_the_login_screen(temp_db):
    out = app.handle_login("bad;id")
    assert out[0]["visible"] is True and out[1]["visible"] is False
    assert out[5].startswith("!!!")
    assert out[8] is False


def test_new_user_logs_in_unlocked_with_defaults(temp_db):
    out = app.handle_login("newcomer")
    style, difficulty, topics = out[2], out[3], out[4]
    assert (style["value"], difficulty["value"], topics["value"]) == ("step-by-step", "intermediate", "")
    assert style["interactive"] and difficulty["interactive"] and topics["interactive"]
    assert out[8] is False and out[7]["value"].endswith("Save Profile")


def test_returning_user_logs_in_with_the_profile_loaded_and_locked(temp_db):
    db.save_profile("alice", "visual examples", "advanced", ["robots", "ai"])
    out = app.handle_login("Alice")
    style, difficulty, topics = out[2], out[3], out[4]
    assert (style["value"], difficulty["value"], topics["value"]) == ("visual examples", "advanced", "robots, ai")
    assert not (style["interactive"] or difficulty["interactive"] or topics["interactive"])
    assert out[8] is True and out[7]["value"].endswith("Edit Profile")


def test_save_profile_persists_cleans_and_locks(temp_db):
    out = app.toggle_profile("alice", "visual examples", "beginner", " robots ,, ai ,", False)
    assert out[4] is True and "locked" in out[5].lower()
    assert not out[0]["interactive"] and not out[1]["interactive"] and not out[2]["interactive"]
    assert out[3]["value"].endswith("Edit Profile")
    saved = db.get_profile("alice")
    assert saved == {"learning_style": "visual examples", "difficulty": "beginner", "topics": ["robots", "ai"]}


def test_edit_profile_unlocks_without_changing_the_saved_profile(temp_db):
    db.save_profile("alice", "step-by-step", "beginner", ["ai"])
    out = app.toggle_profile("alice", "visual examples", "advanced", "ai", True)
    assert out[4] is False
    assert out[0]["interactive"] and out[1]["interactive"] and out[2]["interactive"]
    assert out[3]["value"].endswith("Save Profile")
    assert db.get_profile("alice")["learning_style"] == "step-by-step", "unlocking alone must not save"


def test_save_edit_save_cycle_works(temp_db):
    locked = False
    out = app.toggle_profile("alice", "step-by-step", "beginner", "ai", locked); locked = out[4]
    assert locked is True
    out = app.toggle_profile("alice", "step-by-step", "beginner", "ai", locked); locked = out[4]
    assert locked is False
    out = app.toggle_profile("alice", "visual examples", "advanced", "databases", locked); locked = out[4]
    assert locked is True
    assert db.get_profile("alice") == {"learning_style": "visual examples", "difficulty": "advanced", "topics": ["databases"]}


def test_invalid_topics_are_not_saved_and_the_profile_stays_unlocked(temp_db):
    out = app.toggle_profile("alice", "step-by-step", "beginner", "z" * 500, False)
    assert out[4] is False and out[5].startswith("!!!")
    assert db.get_profile("alice") is None


def test_saving_a_profile_for_a_malformed_user_id_is_rejected(temp_db):
    out = app.toggle_profile("bad;id", "step-by-step", "beginner", "ai", False)
    assert out[4] is False and out[5].startswith("!!!")


def test_back_to_login_resets_the_gate_for_the_next_user():
    out = app.go_back_to_login()
    assert out[0]["visible"] is True and out[1]["visible"] is False
    assert out[3]["visible"] is False and out[4] is False


# app: Gradio wiring ------------------------------------------------------

# Verify handler/output arity and the Clear-chat gate-release event.

def _fn_name(block_fn):
    f = block_fn.fn
    f = getattr(f, "func", f)
    return getattr(f, "__name__", None)


def _declared_outputs(name):
    return {len(bf.outputs) for bf in app.demo.fns.values() if _fn_name(bf) == name}


@pytest.mark.parametrize("name, call", [
    ("handle_login",     lambda: app.handle_login("someone")),
    ("go_back_to_login", lambda: app.go_back_to_login()),
    ("toggle_profile",   lambda: app.toggle_profile("someone", "step-by-step", "beginner", "ai", False)),
    ("rate_response",    lambda: app.rate_response("someone", 3)),
    ("release_gate",     lambda: app.release_gate()),
])
def test_handler_return_arity_matches_its_wired_outputs(temp_db, name, call):
    declared = _declared_outputs(name)
    assert declared, f"{name} is not wired to any event"
    assert declared == {len(call())}, (
        f"{name} returns {len(call())} values but is wired to {declared} outputs; "
        f"Gradio raises this only when the button is clicked")


def test_respond_returns_the_message_plus_three_additional_outputs(monkeypatch):
    monkeypatch.setattr(app, "process_image_and_text_detailed", lambda u, m, i: _res())
    assert len(app.respond("q", [], "alice", None, False)) == 4


def test_the_clear_chat_gate_release_is_actually_wired():
    """
    Verify that the built-in Chatbot clear event releases the rating 
    gate when supported by the installed Gradio version.
    """
    # Registration alone is insufficient, verify the specific triggering event.
    events = [event for bf in app.demo.fns.values() if _fn_name(bf) == "release_gate"
              for (_block_id, event) in bf.targets]
    assert "clear" in events, (
        "Chatbot.clear did not register the expected event; "
        f"release_gate is triggered by {events or 'nothing'}. "
        "'Verify that the explicit Clear chat control remains available.")


# app: upload / duplicate rejection ---------------------------------------

def test_first_upload_is_indexed_and_registered(temp_db, tmp_path, stub_indexing):
    result = app.upload_and_index([_pdf(tmp_path, "L1.pdf")], "alice")
    assert "✅ L1.pdf: 3 chunks" in result and "Indexed 3 chunks from 1 of 1" in result
    assert stub_indexing["index"][0][2] == [{"source": "L1.pdf"}] * 3
    assert [r["filename"] for r in db.list_indexed_files("alice")] == ["L1.pdf"]


def test_reuploading_the_same_file_is_rejected_and_not_reindexed(temp_db, tmp_path, stub_indexing):
    f = _pdf(tmp_path, "L1.pdf")
    app.upload_and_index([f], "alice")
    result = app.upload_and_index([f], "alice")
    assert "already indexed" in result and "✅" not in result
    assert len(stub_indexing["index"]) == 1


def test_same_content_under_a_new_name_is_rejected(temp_db, tmp_path, stub_indexing):
    app.upload_and_index([_pdf(tmp_path, "L1.pdf", b"same body")], "alice")
    result = app.upload_and_index([_pdf(tmp_path, "renamed copy.pdf", b"same body")], "alice")
    assert "already indexed" in result and len(stub_indexing["index"]) == 1


def test_different_content_under_an_existing_name_is_rejected(temp_db, tmp_path, stub_indexing):
    app.upload_and_index([_pdf(tmp_path, "L1.pdf", b"version one")], "alice")
    other = tmp_path / "elsewhere"
    other.mkdir()
    result = app.upload_and_index([_pdf(other, "L1.pdf", b"version two")], "alice")
    assert "different file is already indexed" in result and len(stub_indexing["index"]) == 1


def test_the_same_file_selected_twice_in_one_batch_is_indexed_once(temp_db, tmp_path, stub_indexing):
    f = _pdf(tmp_path, "L1.pdf")
    result = app.upload_and_index([f, f], "alice")
    assert "selected twice in this batch" in result and len(stub_indexing["index"]) == 1


def test_two_files_with_identical_content_in_one_batch_are_indexed_once(temp_db, tmp_path, stub_indexing):
    a = _pdf(tmp_path, "a.pdf", b"identical")
    b = _pdf(tmp_path, "b.pdf", b"identical")
    app.upload_and_index([a, b], "alice")
    assert len(stub_indexing["index"]) == 1


def test_one_bad_file_does_not_stop_the_rest_of_the_batch(temp_db, tmp_path, stub_indexing):
    good = _pdf(tmp_path, "good.pdf", b"good")
    evil = tmp_path / "evil.pdf"
    evil.write_bytes(b"MZ\x90\x00 executable")
    result = app.upload_and_index([_FakeFile(str(evil)), good], "alice")
    assert "❌ evil.pdf: rejected" in result and "✅ good.pdf" in result
    assert stub_indexing["process"] == [good.name], "rejected files must not reach the parser"


@pytest.mark.checklist
def test_unsupported_extension_never_reaches_a_parser(temp_db, tmp_path, stub_indexing):
    exe = tmp_path / "setup.exe"
    exe.write_bytes(b"MZ")
    result = app.upload_and_index([_FakeFile(str(exe))], "alice")
    assert "unsupported" in result and stub_indexing["process"] == []


def test_a_file_that_fails_to_parse_is_not_registered_so_it_can_be_retried(temp_db, tmp_path, monkeypatch):
    f = _pdf(tmp_path, "flaky.pdf")
    monkeypatch.setattr(app, "process_file", lambda p: (_ for _ in ()).throw(ValueError("parse failure")))
    result = app.upload_and_index([f], "alice")
    assert "parse failure" in result
    assert db.list_indexed_files("alice") == []

    monkeypatch.setattr(app, "process_file", lambda p: "text")
    monkeypatch.setattr(app, "chunk_text", lambda t, **kwargs: ["c"])
    monkeypatch.setattr(app, "add_documents", lambda *a, **k: None)
    assert "✅" in app.upload_and_index([f], "alice"), "failed indexing attempts must not prevent a later retry."


def test_a_file_with_no_extractable_text_is_not_registered(temp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(app, "process_file", lambda p: "")
    monkeypatch.setattr(app, "chunk_text", lambda t: [])
    app.upload_and_index([_pdf(tmp_path, "blank.pdf")], "alice")
    assert db.list_indexed_files("alice") == []


def test_only_five_files_per_batch_are_processed(temp_db, tmp_path, stub_indexing):
    files = [_pdf(tmp_path, f"f{i}.pdf", f"body {i}".encode()) for i in range(7)]
    result = app.upload_and_index(files, "alice")
    assert len(stub_indexing["index"]) == app.MAX_FILES_PER_UPLOAD == 5
    assert "Only the first 5 of the 7" in result


def test_uploads_are_scoped_per_user(temp_db, tmp_path, stub_indexing):
    f = _pdf(tmp_path, "L1.pdf")
    app.upload_and_index([f], "alice")
    assert "✅" in app.upload_and_index([f], "bob"), "Indexed files must be scoped to the owning user."


def test_upload_rejects_a_malformed_user_id(temp_db, tmp_path, stub_indexing):
    assert app.upload_and_index([_pdf(tmp_path)], "bad;id").startswith("!!!")
    assert stub_indexing["process"] == []


def test_reset_materials_clears_the_registry_and_the_vector_collection(temp_db, monkeypatch):
    db.record_indexed_file("alice", "L1.pdf", "h", 3)
    reset = []
    monkeypatch.setattr(app, "reset_collection", lambda uid: reset.append(uid))
    msg = app.reset_materials("Alice")
    assert db.list_indexed_files("alice") == []
    assert reset == ["Alice"] or reset == ["alice"]
    assert "Cleared 1" in msg


def test_after_a_reset_the_same_file_can_be_indexed_again(temp_db, tmp_path, stub_indexing, monkeypatch):
    monkeypatch.setattr(app, "reset_collection", lambda uid: None)
    f = _pdf(tmp_path, "L1.pdf")
    app.upload_and_index([f], "alice")
    app.reset_materials("alice")
    assert "✅" in app.upload_and_index([f], "alice")


def test_show_indexed_files_lists_what_is_registered(temp_db):
    assert app.show_indexed_files("alice") == "Nothing indexed yet."
    db.record_indexed_file("alice", "L1.pdf", "h", 41)
    out = app.show_indexed_files("alice")
    assert "L1.pdf" in out and "41 chunks" in out


# rag: real ChromaDB (temporary directory), structural checks ------------

def _names():
    return [getattr(c, "name", c) for c in rag.client.list_collections()]


@pytest.mark.parametrize("user_id", ["Jun Hui", "a@b.com", "x.y-z", "  Padded  ", "demo"])
def test_collection_names_are_always_accepted_by_chromadb(user_id):
    """
    User identifiers must be normalised into ChromaDB-compatible collection names.
    """
    assert rag.get_collection(user_id) is not None


def test_collection_naming_rules():
    assert rag._collection_name("Jun Hui") == "user_jun_hui"
    assert rag._collection_name("demo") == "user_demo"
    assert rag._collection_name("demo", "minilm") == "user_demo__minilm"
    assert rag._collection_name("Alice ") == rag._collection_name("alice")


def test_list_sources_for_an_unknown_user_is_empty_and_creates_nothing():
    """
    Listing sources must not create a collection for an unknown user.
    """
    before = set(_names())
    assert rag.list_sources("never_seen_before") == []
    assert set(_names()) == before


def test_list_sources_returns_distinct_filenames_in_order():
    rag.add_documents("srcuser", ["a1", "a2"], metadatas=[{"source": "A.pdf"}] * 2)
    rag.add_documents("srcuser", ["b1"], metadatas=[{"source": "B.pdf"}])
    assert rag.list_sources("srcuser") == ["A.pdf", "B.pdf"]


def test_source_filter_restricts_results_to_one_document():
    rag.add_documents("filt", ["alpha one", "alpha two", "alpha three"], metadatas=[{"source": "A.pdf"}] * 3)
    rag.add_documents("filt", ["beta one", "beta two", "beta three"], metadatas=[{"source": "B.pdf"}] * 3)
    got = rag.retrieve_with_sources("filt", "one", top_k=6, source_filter="B.pdf")
    assert got and {r["source"] for r in got} == {"B.pdf"}


def test_source_filter_accepts_a_list_of_documents():
    rag.add_documents("filt2", ["a"], metadatas=[{"source": "A.pdf"}])
    rag.add_documents("filt2", ["b"], metadatas=[{"source": "B.pdf"}])
    rag.add_documents("filt2", ["c"], metadatas=[{"source": "C.pdf"}])
    got = rag.retrieve_with_sources("filt2", "x", top_k=6, source_filter=["A.pdf", "C.pdf"])
    assert {r["source"] for r in got} == {"A.pdf", "C.pdf"}


def test_source_filter_naming_a_nonexistent_document_returns_nothing():
    rag.add_documents("filt3", ["a"], metadatas=[{"source": "A.pdf"}])
    assert rag.retrieve_with_sources("filt3", "x", top_k=5, source_filter="ghost.pdf") == []


def test_no_source_filter_searches_every_document():
    rag.add_documents("filt4", ["a"], metadatas=[{"source": "A.pdf"}])
    rag.add_documents("filt4", ["b"], metadatas=[{"source": "B.pdf"}])
    assert {r["source"] for r in rag.retrieve_with_sources("filt4", "x", top_k=5)} == {"A.pdf", "B.pdf"}


def test_reset_collection_removes_the_index_and_is_safe_to_repeat():
    rag.add_documents("resetme", ["a"], metadatas=[{"source": "A.pdf"}])
    rag.reset_collection("resetme")
    assert rag.list_sources("resetme") == []
    rag.reset_collection("resetme")   # Repeated resets must be safe.