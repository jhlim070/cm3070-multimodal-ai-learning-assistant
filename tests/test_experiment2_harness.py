"""
Experiment 2 harness tests

The suite compares generation models while keeping retrieval and prompts fixed.
It also verifies three important regression points:

    1. Every model receives a byte-identical prompt for the same question. 
    
    2. Resource sampling reuses primed psutil.Process objects, so CPU readings are
    based on real samples rather than fresh-object baselines. 
    
    3. Ollama model tags such as "phi3:mini" are safely converted to filesystem
    names without collisions.

Most end-to-end tests use deterministic fake embeddings and model responses.
The resource-sampler tests intentionally use real psutil objects and CPU load,
because those behaviors cannot be meaningfully validated with mocks alone.
"""
import csv
import hashlib
import json
import os
import re
import sys
import time

import pytest

import db
import engine
import experiment2_harness as h
import rag


# timing extraction ---------------------------------------------------------

def test_extract_timing_reads_all_ollama_fields():
    response = {"total_duration": 2_000_000_000, "load_duration": 300_000_000,
                "prompt_eval_count": 500, "prompt_eval_duration": 200_000_000,
                "eval_count": 100, "eval_duration": 1_500_000_000}
    t = h.extract_timing(response, wall_s=2.1)
    assert t["total_duration_s"] == 2.0 and t["load_duration_s"] == 0.3
    assert t["gen_tokens_per_s"] == pytest.approx(100 / 1.5)
    assert t["prompt_tokens_per_s"] == pytest.approx(500 / 0.2)
    assert t["wall_s"] == 2.1


@pytest.mark.parametrize("response", [
    {}, {"total_duration": "not a number"}, {"eval_count": 100},   # no eval_duration -> no rate
    {"eval_duration": 0}, None,
])
def test_extract_timing_never_raises_on_missing_or_malformed_fields(response):
    if response is None:
        response = {}
    t = h.extract_timing(response, wall_s=1.0)
    assert t["wall_s"] == 1.0
    assert t["gen_tokens_per_s"] is None or isinstance(t["gen_tokens_per_s"], float)


def test_extract_timing_reads_a_real_ollama_chatresponse_not_just_a_dict():
    """
    Regression guard for Ollama's real ChatResponse type.
    
    Ollama returns a ChatResponse object rather than a plain dict. The timing
    extractor must therefore support attribute access as well as dict access;
    otherwise all timing fields would silently be missing during real runs.
    """
    ollama = pytest.importorskip("ollama")
    response = ollama.ChatResponse(
        model="llama3.2", message={"role": "assistant", "content": "ok"},
        total_duration=2_000_000_000, load_duration=300_000_000,
        prompt_eval_count=500, prompt_eval_duration=200_000_000,
        eval_count=100, eval_duration=1_500_000_000)
    assert not isinstance(response, dict)          # ChatResponse is an object, not a dict
    t = h.extract_timing(response, wall_s=2.1)
    assert t["total_duration_s"] == 2.0 and t["load_duration_s"] == 0.3
    assert t["prompt_eval_count"] == 500 and t["eval_count"] == 100
    assert t["prompt_tokens_per_s"] == pytest.approx(2500.0)
    assert t["gen_tokens_per_s"] == pytest.approx(100 / 1.5)


def test_extract_timing_reads_attribute_only_responses():
    """A response object exposing fields only as attributes (no .get) is also read."""
    class AttrOnly:
        total_duration = 1_000_000_000
        eval_count = 20
        eval_duration = 500_000_000
    t = h.extract_timing(AttrOnly(), wall_s=1.0)
    assert t["total_duration_s"] == 1.0 and t["gen_tokens_per_s"] == 40.0


def test_extract_timing_zero_duration_does_not_divide_by_zero():
    t = h.extract_timing({"eval_count": 50, "eval_duration": 0}, wall_s=1.0)
    assert t["gen_tokens_per_s"] is None


# filename sanitisation ---------------------------------------------------------

@pytest.mark.parametrize("model, expected", [
    ("llama3.2", "llama3.2"),
    ("phi3:mini", "phi3_mini"),
    ("gemma3:4b", "gemma3_4b"),
    ('weird<>:"/\\|?*name', "weird_________name"),
])
def test_sanitize_filename(model, expected):
    assert h._sanitize_filename(model) == expected


def test_sanitized_filenames_stay_distinct_for_different_tags():
    """Two different real tags must never collide into the same cell file."""
    tags = ["phi3:mini", "phi3:medium", "gemma3:4b", "gemma3:12b", "llama3.2"]
    assert len({h._sanitize_filename(t) for t in tags}) == len(tags)


# Resource sampler, real psutil, real CPU load ----------------------------------

def _busy_for(seconds):
    t0 = time.time()
    x = 0
    while time.time() - t0 < seconds:
        x += 1


def test_sampler_system_wide_collects_real_samples_under_load():
    s = h.ResourceSampler(interval=0.05)
    s.start()
    _busy_for(0.6)
    s.stop()
    summary = s.summary()
    assert summary["n_samples"] >= 5
    assert summary["matched_any_process"] is True
    assert summary["cpu_pct"]["mean"] is not None and summary["ram_mb"]["mean"] > 0


def test_sampler_process_specific_matches_a_real_process_and_reads_nonzero():
    """
    Regression guard for psutil.Process priming
    psutil.cpu_percent() reports CPU usage relative to the previous call on
    the same Process object. Recreating Process objects on every sampling tick
    resets that baseline and can make a busy process appear to use ~0% CPU.
    
    Run the current process under load and require a non-zero mean CPU reading
    to catch that regression.
    """
    import psutil
    this_name = psutil.Process(os.getpid()).name()
    s = h.ResourceSampler(interval=0.05, process_names=[this_name])
    s.start()
    _busy_for(0.6)
    s.stop()
    summary = s.summary()
    assert summary["matched_any_process"] is True
    assert summary["n_samples"] >= 5
    assert summary["ram_mb"]["mean"] > 0
    assert summary["cpu_pct"]["mean"] > 0, (
        "busy current process reported 0% CPU; Process objects may not be reused"
    )


def test_sampler_reuses_the_same_process_objects_across_ticks():
    """
    Verify that process-mode sampling reuses the primed Process objects.
    
    Reusing the objects is required for psutil's cpu_percent() measurements
    to have a meaningful previous sample.
    """
    import psutil
    this_name = psutil.Process(os.getpid()).name()
    s = h.ResourceSampler(interval=0.05, process_names=[this_name])
    s.start()
    time.sleep(0.12)
    procs_after_start = list(s._procs)
    time.sleep(0.12)
    s.stop()
    assert s._procs and s._procs == procs_after_start, "process objects changed identity between ticks"


def test_sampler_with_no_matching_process_reports_zeros_not_system_wide_fallback():
    """
    A process-name filter must not silently fall back to system-wide metrics.
    
    If no requested process exists, report zero process-specific usage instead
    of returning measurements for unrelated processes.
    """
    s = h.ResourceSampler(interval=0.05, process_names=["definitely_not_a_real_process_xyz_123"])
    s.start()
    time.sleep(0.2)
    s.stop()
    summary = s.summary()
    assert summary["matched_any_process"] is False
    assert summary["cpu_pct"]["mean"] == 0.0 and summary["ram_mb"]["mean"] == 0.0


def test_sampler_survives_zero_duration():
    s = h.ResourceSampler(interval=0.05)
    s.start()
    s.stop()
    summary = s.summary()
    assert summary["cpu_pct"]["mean"] is None and summary["n_samples"] == 0


def test_sampler_process_mode_starts_with_matched_any_process_false_before_start():
    s = h.ResourceSampler(process_names=["whatever"])
    assert s.matched_any_process is False
    s2 = h.ResourceSampler()   # system-wide: trivially "matched" from the start
    assert s2.matched_any_process is True


# Aggregation -----------------------------------------------------------------

def _row(total=2.0, load=0.3, gen_tps=50.0, prompt_tps=1000.0, refused=False, error=None):
    return {"id": 1, "type": "factual", "answerable": True, "question": "q", "answer": "a",
            "reference_answer": "", "sources": [], "refused": refused, "error": error,
            "wall_s": total, "total_duration_s": total, "load_duration_s": load,
            "prompt_eval_count": 100, "prompt_eval_s": 100 / prompt_tps if prompt_tps else None,
            "prompt_tokens_per_s": prompt_tps,
            "eval_count": 50, "eval_s": 50 / gen_tps if gen_tps else None, "gen_tokens_per_s": gen_tps}


def _resources(cpu_mean=10.0, cpu_peak=20.0, ram_mean=100.0, ram_peak=150.0, n=5, matched=True):
    return {"cpu_pct": {"mean": cpu_mean, "peak": cpu_peak}, "ram_mb": {"mean": ram_mean, "peak": ram_peak},
            "n_samples": n, "matched_any_process": matched}


def test_summarise_computes_mean_and_median_and_counts_refusals():
    cell = {"model": "m", "resources": _resources(),
            "rows": [_row(total=1.0, refused=True), _row(total=3.0), _row(total=2.0, error="boom")]}
    s = h.summarise([cell])[0]
    assert s["n_questions"] == 3 and s["n_errors"] == 1
    assert s["mean_total_duration_s"] == 2.0            # mean of the two OK rows: (1+3)/2
    assert s["median_total_duration_s"] == 2.0
    assert s["n_refused"] == 1


def test_summarise_reports_a_fully_failed_model_instead_of_dropping_it():
    cell = {"model": "ghost:1b", "rows": [_row(error="not found")], "resources": _resources(n=0, matched=True),
            "error": "not found"}
    s = h.summarise([cell])[0]
    assert s == {"model": "ghost:1b", "error": "not found"}


def test_summarise_carries_resource_numbers_through():
    cell = {"model": "m", "resources": _resources(cpu_mean=55.5, ram_peak=888.0), "rows": [_row()]}
    s = h.summarise([cell])[0]
    assert s["cpu_pct_mean"] == 55.5 and s["ram_mb_peak"] == 888.0


def test_markdown_sorts_by_latency_and_flags_unmatched_process():
    fast = {"model": "fast", "n_questions": 31, "n_errors": 0, "n_refused": 0,
            "mean_total_duration_s": 1.0, "median_total_duration_s": 1.0, "mean_load_duration_s": 0.1,
            "mean_gen_tokens_per_s": 90.0, **{k: 0.0 for k in ("cpu_pct_mean", "cpu_pct_peak", "ram_mb_mean", "ram_mb_peak")},
            "resource_process_matched": True, "error": ""}
    slow_unmatched = {**fast, "model": "slow", "mean_total_duration_s": 5.0, "median_total_duration_s": 5.0,
                      "resource_process_matched": False}
    md = h.render_markdown([slow_unmatched, fast], sample_interval=0.2)
    assert md.index("| fast |") < md.index("| slow |")
    assert "Warning" in md and "matched no process" in md


def test_markdown_lists_failed_models_separately():
    md = h.render_markdown([{"model": "ghost:1b", "error": "not found"}], sample_interval=0.2)
    assert "ghost:1b: not found" in md and "Models that failed" in md


# Prompt fairness: byte-identical prompts across models -----------------------

SOURCES = ["A.pdf", "B.pdf"]
QUESTIONS = [
    {"id": 1, "type": "factual", "answerable": True, "question": "What is X?", "evidence": [], "reference_answer": ""},
    {"id": 2, "type": "conceptual", "answerable": True, "question": "Why does Y happen?", "evidence": [], "reference_answer": ""},
]


def test_build_fixed_prompts_matches_the_live_engines_prompt_exactly(temp_db, monkeypatch):
    """
    Regression guard against prompt drift between Engine and Experiment 2
    
    Experiment 2 must reproduce the live Engine prompt exactly so that models
    differ only in generation, not in the prompt they receive.
    """
    results = [{"text": f"chunk {i}", "source": SOURCES[0], "distance": 0.1 * i} for i in range(8)]
    captured = []
    monkeypatch.setattr(engine._client, "chat",
                        lambda model, messages, **kw: captured.append((model, messages)) or {"message": {"content": "x"}})
    monkeypatch.setattr(engine, "list_sources", lambda uid: [])
    monkeypatch.setattr(engine, "retrieve_with_sources", lambda uid, q, top_k=5, **kw: results)

    engine.generate_answer_detailed("nobody", QUESTIONS[0]["question"])   # the live path
    live_system, live_user = captured[0][1][0]["content"], captured[0][1][1]["content"]

    prompts = h.build_fixed_prompts(engine, rag_module_stub(results), "nobody", [QUESTIONS[0]], "bge-small")
    p = prompts[1]
    assert p["system"] == live_system
    assert p["user"] == live_user


class rag_module_stub:
    """
    Minimal retrieval stub used to test prompt construction without indexing.
    """
    def __init__(self, results):
        self._results = results

    def retrieve_with_sources(self, user, query, top_k=5, **kw):
        return self._results


def test_every_model_receives_the_identical_prompt_for_the_same_question(temp_db, monkeypatch):
    """
    Every model must receive byte-identical messages for the same question.
    
    Retrieval is performed once and the resulting prompt is reused across the
    model grid. This prevents differences in retrieval or prompt construction
    from contaminating the model comparison.
    """
    results = [{"text": "shared chunk", "source": "A.pdf", "distance": 0.1}]
    monkeypatch.setattr(engine, "list_sources", lambda uid: [])
    monkeypatch.setattr(engine, "retrieve_with_sources", lambda uid, q, top_k=5, **kw: results)
    prompts = h.build_fixed_prompts(engine, engine, "u", QUESTIONS, "bge-small")

    captured = []
    monkeypatch.setattr(engine._client, "chat",
                        lambda model, messages, **kw: captured.append((model, messages)) or {"message": {"content": "ok"}})

    for model in ("llama3.2", "phi3:mini", "gemma3:4b"):
        h.run_model(engine, model, prompts, QUESTIONS, lambda: h.ResourceSampler(interval=10))

    # Each model is run over QUESTIONS in the same order, so 
    # corresponding calls can be compared directly.
    first_calls = [msgs for i, (m, msgs) in enumerate(captured) if i % len(QUESTIONS) == 0]
    assert len(first_calls) == 3
    assert all(c == first_calls[0] for c in first_calls), "the three models were not asked the same prompt"


# Run_model / per-question rows ----------------------------------------------

def _fixed_prompts():
    return {q["id"]: {"system": "SYS", "user": f"USER for {q['id']}", "sources": ["A.pdf"]} for q in QUESTIONS}


def test_run_model_records_timing_answer_and_refusal(monkeypatch):
    monkeypatch.setattr(engine._client, "chat", lambda model, messages, **kw: {
        "message": {"content": "a real answer"},
        "total_duration": 1_000_000_000, "eval_count": 20, "eval_duration": 500_000_000,
    })
    cell = h.run_model(engine, "llama3.2", _fixed_prompts(), QUESTIONS, lambda: h.ResourceSampler(interval=10))
    assert cell["model"] == "llama3.2" and cell["error"] is None
    assert len(cell["rows"]) == 2
    assert cell["rows"][0]["total_duration_s"] == 1.0
    assert cell["rows"][0]["refused"] is False


def test_run_model_flags_a_refusal(monkeypatch):
    monkeypatch.setattr(engine._client, "chat", lambda model, messages, **kw: {"message": {"content": "NOT_IN_MATERIALS"}})
    cell = h.run_model(engine, "llama3.2", _fixed_prompts(), QUESTIONS, lambda: h.ResourceSampler(interval=10))
    assert all(r["refused"] is True for r in cell["rows"])


def test_run_model_records_per_row_errors_without_stopping_the_other_questions(monkeypatch):
    calls = {"n": 0}
    def chat(model, messages, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("slow model")
        return {"message": {"content": "ok"}, "total_duration": 1_000_000_000}
    monkeypatch.setattr(engine._client, "chat", chat)
    cell = h.run_model(engine, "llama3.2", _fixed_prompts(), QUESTIONS, lambda: h.ResourceSampler(interval=10))
    assert cell["rows"][0]["error"] is not None and cell["rows"][1]["error"] is None
    assert cell["error"] is None, "one bad row among several good ones is not a fully-failed cell"


def test_run_model_wraps_the_whole_run_in_the_sampler(monkeypatch):
    started = {"v": False}
    stopped = {"v": False}

    class SpySampler(h.ResourceSampler):
        def start(self_inner):
            started["v"] = True
        def stop(self_inner):
            stopped["v"] = True
        def summary(self_inner):
            return h.ResourceSampler().summary()

    monkeypatch.setattr(engine._client, "chat", lambda model, messages, **kw: {"message": {"content": "ok"}})
    h.run_model(engine, "llama3.2", _fixed_prompts(), QUESTIONS, lambda: SpySampler())
    assert started["v"] and stopped["v"]


def test_sampler_is_stopped_even_if_every_question_raises(monkeypatch):
    """
    The resource sampler must be stopped even when every model call fails.
    """
    stopped = {"v": False}

    class SpySampler(h.ResourceSampler):
        def start(self_inner):
            pass
        def stop(self_inner):
            stopped["v"] = True
        def summary(self_inner):
            return h.ResourceSampler().summary()

    monkeypatch.setattr(engine._client, "chat", lambda model, messages, **kw: (_ for _ in ()).throw(RuntimeError("down")))
    h.run_model(engine, "llama3.2", _fixed_prompts(), QUESTIONS, lambda: SpySampler())
    assert stopped["v"], "the sampler must be stopped (in a finally) even when every call raises"


# Full grid, synthetic corpus, deterministic fake embedder + fake models ------

DOCS = {
    "CM1001 Marsupials.docx": [
        "The quokka is a small marsupial found mainly on Rottnest Island off the coast of Western Australia.",
        "Wombats produce cube shaped droppings which they use to mark territory near their burrows.",
        "A joey remains inside its mother's pouch for several months while it continues to develop.",
    ],
    "CM1002 Chemistry.docx": [
        "Water molecules attract one another through hydrogen bonds which explain its unusually high boiling point.",
        "An acid donates protons while a base accepts them according to the Bronsted Lowry definition.",
        "Catalysts lower the activation energy of a reaction without being consumed in the process.",
    ],
}
GRID_QUESTIONS = [
    [1, "CM1001", "Factual", "What shape are wombat droppings?", "yes", "cube shaped droppings", "ref"],
    [2, "CM1002", "Conceptual", "Why does water boil at a high temperature?", "yes", "hydrogen bonds", ""],
]


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        import numpy as np
        out = []
        for t in texts:
            v = np.zeros(512, dtype="float32")
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 512] += 1.0
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.array(out)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    from docx import Document
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, paras in DOCS.items():
        d = Document()
        for p in paras:
            d.add_paragraph(p)
        d.save(str(docs / name))
    qpath = tmp_path / "q.csv"
    with open(qpath, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "doc", "type", "question", "answerable", "evidence", "reference_answer"])
        w.writerows(GRID_QUESTIONS)
    monkeypatch.setattr(rag, "get_embedder", lambda name=rag.DEFAULT_EMBEDDING_MODEL: FakeEmbedder())
    return {"docs": str(docs), "questions": str(qpath), "out": str(tmp_path / "out")}


def _fake_chat_factory(reply="an answer", fail_for=()):
    def chat(model, messages, **kw):
        if model in fail_for:
            raise RuntimeError(f"model '{model}' not found, try pulling it first")
        return {"message": {"content": reply}, "total_duration": 1_000_000_000,
                "eval_count": 20, "eval_duration": 500_000_000}
    return chat


def _run(corpus, models, *extra, chat_fn=None, monkeypatch=None):
    if monkeypatch and chat_fn:
        monkeypatch.setattr(engine._client, "chat", chat_fn)
    return h.main(["--docs", corpus["docs"], "--questions", corpus["questions"], "--out", corpus["out"],
                   "--models", *models, "--sample-interval", "10", *extra])


def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def test_full_grid_runs_and_writes_every_output(corpus, monkeypatch):
    rc = _run(corpus, ["llama3.2", "phi3:mini"], chat_fn=_fake_chat_factory(), monkeypatch=monkeypatch)
    assert rc == 0
    out = corpus["out"]
    for f in ("config.json", "summary.csv", "per_question.csv", "generations.csv", "summary.md"):
        assert os.path.exists(os.path.join(out, f)), f
    assert len(os.listdir(os.path.join(out, "cells"))) == 2


def test_per_question_has_one_row_per_model_per_question(corpus, monkeypatch):
    _run(corpus, ["llama3.2", "phi3:mini"], chat_fn=_fake_chat_factory(), monkeypatch=monkeypatch)
    rows = _read_csv(os.path.join(corpus["out"], "per_question.csv"))
    assert len(rows) == 2 * len(GRID_QUESTIONS)
    assert {r["model"] for r in rows} == {"llama3.2", "phi3:mini"}


def test_generations_csv_has_blank_manual_columns_for_every_row(corpus, monkeypatch):
    _run(corpus, ["llama3.2"], chat_fn=_fake_chat_factory(), monkeypatch=monkeypatch)
    rows = _read_csv(os.path.join(corpus["out"], "generations.csv"))
    assert len(rows) == len(GRID_QUESTIONS)
    assert all(r["manual_relevance"] == "" and r["manual_quality"] == "" for r in rows)
    assert all(r["answer"] == "an answer" for r in rows)


def test_config_json_pins_every_fairness_setting(corpus, monkeypatch):
    _run(corpus, ["llama3.2"], "--embedding-model", "minilm", "--chunking", "fixed",
        chat_fn=_fake_chat_factory(), monkeypatch=monkeypatch)
    with open(os.path.join(corpus["out"], "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    assert cfg["embedding_model"] == "minilm" and cfg["chunking"] == "fixed"
    assert cfg["generation_options"] == engine.GENERATION_OPTIONS
    assert cfg["retrieval_top_k"] == engine.RETRIEVAL_TOP_K and cfg["context_chunks"] == engine.CONTEXT_CHUNKS
    assert set(cfg["documents"]) == set(DOCS)


def test_one_failing_model_does_not_lose_the_others(corpus, monkeypatch):
    rc = _run(corpus, ["llama3.2", "ghost:1b", "phi3:mini"],
             chat_fn=_fake_chat_factory(fail_for={"ghost:1b"}), monkeypatch=monkeypatch)
    assert rc == 1
    summary = {r["model"]: r for r in _read_csv(os.path.join(corpus["out"], "summary.csv"))}
    assert summary["llama3.2"]["error"] == "" and summary["phi3:mini"]["error"] == ""
    assert "not found" in summary["ghost:1b"]["error"] and "ollama pull ghost:1b" in summary["ghost:1b"]["error"]


def test_resume_skips_completed_cells(corpus, monkeypatch):
    calls = {"n": 0}
    def chat(model, messages, **kw):
        calls["n"] += 1
        return {"message": {"content": "ok"}, "total_duration": 1_000_000_000}
    monkeypatch.setattr(engine._client, "chat", chat)
    _run(corpus, ["llama3.2", "phi3:mini"])
    after_first = calls["n"]
    assert after_first == 2 * len(GRID_QUESTIONS)
    _run(corpus, ["llama3.2", "phi3:mini"], "--resume")
    assert calls["n"] == after_first, "resume must not re-call the model for completed cells"


def test_resume_still_runs_a_newly_added_model(corpus, monkeypatch):
    calls = {"n": 0}
    def chat(model, messages, **kw):
        calls["n"] += 1
        return {"message": {"content": "ok"}, "total_duration": 1_000_000_000}
    monkeypatch.setattr(engine._client, "chat", chat)
    _run(corpus, ["llama3.2"])
    after_first = calls["n"]
    _run(corpus, ["llama3.2", "phi3:mini"], "--resume")
    assert calls["n"] == after_first + len(GRID_QUESTIONS)


def test_the_harness_never_touches_the_default_chroma_directory(corpus, monkeypatch):
    _run(corpus, ["llama3.2"], chat_fn=_fake_chat_factory(), monkeypatch=monkeypatch)
    assert os.path.isdir(os.path.join(corpus["out"], "chroma_exp2"))


def test_no_stray_chroma_db_when_run_standalone_as_documented(tmp_path, corpus):
    """
    A standalone harness run must not create the default Chroma directory.
    
    rag.py creates its Chroma client when it is first imported. Therefore the 
    harness must configure CHROMA_DB_PATH before importing rag/engine; otherwise
    importing the harness can create ./chroma_db in the caller's working directory.

    This test runs the harness in a fresh subprocess because this test module has
    already imported rag/engine, which would hide an import-order bug. The subprocess
    uses stub Ollama and sentence-transformers modules so the test does not require 
    an Ollama server or downloaded embedding model.
    
    The assertion is intentionally scoped to standalone harness invocation: code that
    imports rag.py before importing the harness is outside the harness's control.
    """
    import subprocess
    stub_dir = tmp_path / "stub_pkgs"
    stub_dir.mkdir()
    (stub_dir / "ollama.py").write_text(
        "class Client:\n"
        "    def __init__(self, *a, **k): pass\n"
        "    def chat(self, model, messages, **kw):\n"
        "        return {'message': {'content': f'A stub answer from {model}.'},\n"
        "                'total_duration': 500_000_000, 'load_duration': 50_000_000,\n"
        "                'prompt_eval_count': 50, 'prompt_eval_duration': 100_000_000,\n"
        "                'eval_count': 20, 'eval_duration': 300_000_000}\n"
        "    def generate(self, *a, **k): return {'response': 'a stub description'}\n"
    )
    (stub_dir / "sentence_transformers.py").write_text(
        "import hashlib, re\n"
        "import numpy as np\n"
        "class SentenceTransformer:\n"
        "    def __init__(self, *a, **k): pass\n"
        "    def encode(self, texts, **kw):\n"
        # rag.py expects the real API's numpy-array-like result.
        "        out = []\n"                            
        "        for t in texts:\n"                     
        "            v = np.zeros(64, dtype='float32')\n"
        "            for tok in re.findall(r'[a-z0-9]+', t.lower()):\n"
        "                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 64] += 1.0\n"
        "            n = np.linalg.norm(v)\n"
        "            out.append(v / n if n else v)\n"
        "        return np.array(out)\n"
    )
    cwd = tmp_path / "launch_dir"
    cwd.mkdir()
    project_root = os.path.dirname(os.path.abspath(h.__file__))
    script = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "import experiment2_harness as h2\n"    # Import order is part of the regression
        "rc = h2.main(['--docs', %r, '--questions', %r, '--out', %r, '--models', 'llama3.2'])\n"
        "sys.exit(rc)\n"
    ) % (project_root, corpus["docs"], corpus["questions"], str(tmp_path / "out"))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(stub_dir), project_root, os.environ.get("PYTHONPATH", "")]))
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(cwd), env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert not (cwd / "chroma_db").exists(), "a stray ./chroma_db was created in the launch directory"
    assert os.path.isdir(str(tmp_path / "out" / "chroma_exp2")), "the harness's own index was not created"


# CLI validation ----------------------------------------------------------------

def test_dry_run_validates_but_asks_no_model(corpus, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(engine._client, "chat", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or {"message": {"content": "x"}})
    rc = _run(corpus, ["llama3.2", "phi3:mini"], "--dry-run", monkeypatch=monkeypatch)
    assert rc == 0 and calls["n"] == 0
    assert not os.path.exists(os.path.join(corpus["out"], "chroma_exp2"))


def test_unknown_embedding_model_is_rejected(corpus, monkeypatch):
    rc = _run(corpus, ["llama3.2"], "--embedding-model", "not-a-model", chat_fn=_fake_chat_factory(), monkeypatch=monkeypatch)
    assert rc == 2


def test_bad_docs_folder_is_rejected(corpus):
    rc = h.main(["--docs", "/no/such/folder", "--questions", corpus["questions"], "--out", corpus["out"],
                "--models", "llama3.2"])
    assert rc == 2


def test_course_code_with_no_document_is_rejected(tmp_path, corpus):
    bad = tmp_path / "bad.csv"
    with open(bad, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "doc", "type", "question", "answerable", "evidence", "reference_answer"])
        w.writerow([1, "CM9999", "Factual", "q?", "yes", "", ""])
    rc = h.main(["--docs", corpus["docs"], "--questions", str(bad), "--out", corpus["out"], "--models", "llama3.2"])
    assert rc == 2


def test_models_argument_is_required():
    with pytest.raises(SystemExit):
        h.main(["--docs", "somewhere"])


def test_unscored_evidence_warning_is_suppressed_for_experiment_2(tmp_path, corpus, capsys):
    """
    Experiment 2 asks every question regardless of the evidence column -- the UNSCORED warning
    is Experiment 1's concern, not this one's, and would be noise here.
    """
    _run(corpus, ["llama3.2"], "--dry-run")
    assert "UNSCORED" not in capsys.readouterr().out