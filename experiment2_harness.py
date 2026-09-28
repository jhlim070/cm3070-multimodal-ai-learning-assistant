"""
Experiment 2 harness: Llama 3.2 vs Phi-3 Mini vs Gemma 3,
comparing latency, throughput, CPU/RAM, and a manual quality rubric -- with RETRIEVAL HELD
FIXED so the only thing that varies between cells is the generation model.

HOW TO RUN (If no file names are changed):
    Dry run (ensure requirements are met):
    python experiment2_harness.py --docs assets --models llama3.2 phi3:mini gemma3:4b --out results/exp2 --dry-run

    Full run (only after `Dry run OK: 31 questions, 3 models. Nothing was indexed, no model was called.`):
    python experiment2_harness.py --docs assets --models llama3.2 phi3:mini gemma3:4b --out results/exp2
    
    NOTE: add `--resume` to continue from previous run, skipping 
        results that have already been collected 

WHY THIS IS A DIFFERENT DESIGN FROM EXPERIMENT 1
    Experiment 1 varied retrieval (embedding model x chunking strategy) and measured retrieval
    quality; the LLM was incidental. Experiment 2 varies the LLM and measures ITS behaviour, so
    retrieval has to be the one thing that does NOT vary -- otherwise a latency or quality
    difference between models could really be a difference in what context they were given.

    This harness retrieves ONCE per question (a single fixed embedding model + chunking
    strategy, matching engine.py's real retrieval width) and builds ONE system or user prompt pair
    per question. Every model in --models is then asked the retrieval has to be the one thing that
    does NOT vary -- The prompts are byte-identical across models; a test verifies that the messages
    captured by a mock client are identical for every model call. Only the model name differs in 
    the API call.
    

WHAT IT MEASURES
    Per model, aggregated over every question in the configured question file:
      total_duration_s, load_duration_s      -- Ollama-reported timing metadata
      
      gen_tokens_per_s, prompt_tokens_per_s  -- Token-processing rates calculated from 
                                                Ollama's response metadata
                                                
      cpu_pct (mean, peak), ram_mb (mean, peak) -- Background resource samples collected 
                                                   throughout the model's run
                                                   
      refused (bool)                          -- Result of engine._looks_like_non_answer(), 
                                                 using the same detector as Experiment 1
                                                 
    Outputs: 
        config.json       -- Pinned settings, document hashes, and version information.
        summary.csv/.md,  -- Aggregated performance results
        per_question.csv  -- Per-question results
        generations.csv   -- Generated answers for manual quality/relevance assessment.

CPU/RAM MEASUREMENT SCOPE
    Default: SYSTEM-WIDE deltas (psutil.cpu_percent() / psutil.virtual_memory()). Simple, needs no
    process name, but includes whatever else the machine is doing during the run -- close other
    heavy applications before running this for real.
    Optional: --process-name NAME [NAME ...] restricts sampling to processes with that exact name
    (case-insensitive), e.g. --process-name ollama ollama_llama_server. Find the right name in
    Task Manager (Windows) while a model is loaded and answering. This gives a tighter number but
    only if the name is right -- if nothing matches, the harness warns and reports zeros rather
    than silently falling back to system-wide (which would look like a real but wrong number).

FAIRNESS
    embedding_model, chunking strategy, chunk_size, overlap, RETRIEVAL_TOP_K, CONTEXT_CHUNKS,
    the system/user prompt template, and GENERATION_OPTIONS are identical for every model and
    are written to config.json so the run is reproducible and the pinning is checkable.
    GENERATION_OPTIONS include num_ctx=8192, which is deliberately the SAME for every model even
    though the three model families may have different native context windows -- this is a
    fairness choice (equal context budget), not a limit any one model naturally has.

WHAT THIS DOES NOT MEASURE
    Answer quality. Performance metrics such as CPU utilisation and generation throughput are 
    quantitative measurements; answer quality requires human assessment. generations.csv is 
    the input to your own reading, not the output of one.

SAFETY
    Uses its own ChromaDB directory inside --out (via CHROMA_DB_PATH, same mechanism as
    experiment1_harness.py) and never touches ./chroma_db or learner.db.
"""
import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse Experiment 1's document/question utilities to keep validation 
# and experiment setup consistent across both harnesses.
import experiment1_harness as exp1   
import validation

DEFAULT_EMBEDDING_MODEL = "bge-small"
# Fixed retrieval configuration selected from Experiment 1.
DEFAULT_CHUNKING = "sliding_window"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100
DEFAULT_SAMPLE_INTERVAL = 0.2
NS_PER_SECOND = 1_000_000_000


def _sanitize_filename(name):
    """
    Convert an Ollama model tag into a filesystem-safe cell filename.
    
    The original model tag is retained for API calls, CSV output, and reports.
    Sanitisation is used only for filenames because tags such as ``phi3:mini``
    contain characters that are invalid on Windows.
    """
    return re.sub(r'[<>:"/\\|?*]', "_", name)


# resource sampling ---------------------------------------------------------
class ResourceSampler:
    """
    Sample CPU utilisation and RAM usage in a background thread.

    Sampling can be performed either system-wide or for processes whose names match
    the configured process-name set. Statistics are computed after sampling stops.

    For process-level CPU measurements, each Process object must first be primed with
    cpu_percent(interval=None). Subsequent readings then measure CPU utilisation since
    the previous reading. The same Process objects are retained between samples so
    that this baseline is not reset on every iteration.
    """

    def __init__(self, interval=DEFAULT_SAMPLE_INTERVAL, process_names=None):
        self.interval = interval
        self.process_names = {n.lower() for n in (process_names or [])}
        self._stop = threading.Event()
        self._thread = None
        self._procs = []
        self.cpu_samples = []
        self.ram_samples_mb = []
        self.matched_any_process = not self.process_names  # trivially true in system-wide mode

    def _discover_and_prime(self, psutil):
        """
        Find matching processes and initialise their CPU measurement baselines.
        """
        found = []
        for p in psutil.process_iter(["name"]):
            try:
                if (p.info.get("name") or "").lower() in self.process_names:
                    p.cpu_percent(interval=None)  # prime; first reading is meaningless, discarded
                    found.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return found

    def start(self):
        import psutil
        self.cpu_samples, self.ram_samples_mb = [], []
        self._stop.clear()
        if self.process_names:
            self._procs = self._discover_and_prime(psutil)
            self.matched_any_process = bool(self._procs)
        else:
            # Initialise the system-wide CPU measurement baseline
            psutil.cpu_percent(interval=None)  
        self._thread = threading.Thread(target=self._run, args=(psutil,), daemon=True)
        self._thread.start()

    def _run(self, psutil):
        while not self._stop.wait(self.interval):
            if self.process_names:
                cpu, ram, alive = 0.0, 0.0, []
                for p in self._procs:
                    try:
                        cpu += p.cpu_percent(interval=None)
                        ram += p.memory_info().rss / (1024 * 1024)
                        alive.append(p)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                if not alive:
                    # A monitored process may have exited and been replaced during model loading.
                    alive = self._discover_and_prime(psutil)
                    if alive:
                        # The newly discovered processes were only primed, so sample them
                        # on the next interval.
                        continue 
                self._procs = alive
                self.cpu_samples.append(cpu)
                self.ram_samples_mb.append(ram)
            else:
                self.cpu_samples.append(psutil.cpu_percent(interval=None))
                self.ram_samples_mb.append(psutil.virtual_memory().used / (1024 * 1024))

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self):
        def stats(xs):
            return {"mean": (sum(xs) / len(xs)) if xs else None, "peak": max(xs) if xs else None}
        return {"cpu_pct": stats(self.cpu_samples), "ram_mb": stats(self.ram_samples_mb),
                "n_samples": len(self.cpu_samples), "matched_any_process": self.matched_any_process}


# building the fixed prompt (once per question, shared by every model) --------------

def build_fixed_prompts(engine, rag, user, questions, embedding_model):
    """
    Retrieve context once for each question and construct the prompts shared by
    every generation model.

    The prompt construction mirrors the non-image path of
    engine.generate_answer_detailed(). A fixed learner profile is used so that
    personalisation settings do not vary across experimental conditions.

    Each question is retrieved exactly once. The resulting context, system prompt,
    user prompt, and question are stored and reused for every model, ensuring that
    retrieval differences cannot contribute to differences between generation models.
    """
    system_msg = engine._build_system_prompt("clear and detailed", "intermediate")
    prompts = {}
    for q in questions:
        results = rag.retrieve_with_sources(user, q["question"], top_k=engine.RETRIEVAL_TOP_K,
                                            embedding_model=embedding_model)
        used = results[:engine.CONTEXT_CHUNKS]
        context = engine._build_context(used) if used else "(no related excerpts were found)"
        user_msg = f"""Excerpts from the student's uploaded materials:

{context}

Question: {q['question']}"""
        prompts[q["id"]] = {"system": system_msg, "user": user_msg,
                            "sources": [r.get("source") for r in used]}
    return prompts


# timing extraction ---------------------------------------------------------

def _is_number(v):
    """
    # Exclude bool because bool is a subclass of int in Python, but True/False
    # is not a valid numeric timing or token-count measurement.
    """
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _response_field(response, key):
    """
    Read a response field from either a mapping or an Ollama response object.

    Ollama's Python client may return a ChatResponse object rather than a plain
    dictionary. Supporting both attribute and mapping-style access keeps timing
    extraction compatible with the client response type and with test doubles.
    """
    if response is None:
        return None
    getter = getattr(response, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:
            pass
    return getattr(response, key, None)


def extract_timing(response, wall_s):
    """
    Extract timing and token-count metadata reported by Ollama.

    Ollama reports duration fields in nanoseconds and token-processing rates can
    therefore be calculated from the corresponding duration and token-count fields.
    Fields that are unavailable are represented as None.

    wall_s is measured independently around the Python API call and is retained as
    a wall-clock reference and fallback measurement.
    """
    def ns(key):
        v = _response_field(response, key)
        return (v / NS_PER_SECOND) if _is_number(v) else None

    prompt_count = _response_field(response, "prompt_eval_count")
    prompt_count = prompt_count if _is_number(prompt_count) else None
    eval_count = _response_field(response, "eval_count")
    eval_count = eval_count if _is_number(eval_count) else None
    prompt_eval_s = ns("prompt_eval_duration")
    eval_s = ns("eval_duration")

    return {
        "wall_s": wall_s,
        "total_duration_s": ns("total_duration"),
        "load_duration_s": ns("load_duration"),
        "prompt_eval_count": prompt_count,
        "prompt_eval_s": prompt_eval_s,
        "prompt_tokens_per_s": (prompt_count / prompt_eval_s)
                               if (prompt_count and prompt_eval_s) else None,
        "eval_count": eval_count,
        "eval_s": eval_s,
        "gen_tokens_per_s": (eval_count / eval_s) if (eval_count and eval_s) else None,
    }


# running one model --------------------------------------------------------------------

def run_model(engine, model, prompts, questions, sampler_factory):
    """
    Run every experiment question through one generation model.

    The questions are processed in a fixed order. Resource utilisation is sampled for
    the duration of the complete model run. Per-question responses, timing metadata,
    and errors are returned together with aggregate resource measurements.
    """
    sampler = sampler_factory()
    sampler.start()
    rows = []
    try:
        for q in questions:
            p = prompts[q["id"]]
            started = time.time()
            answer, error = "", None
            timing = extract_timing({}, 0.0)
            try:
                response = engine._client.chat(
                    model=model,
                    messages=[{"role": "system", "content": p["system"]},
                             {"role": "user", "content": p["user"]}],
                    options=engine.GENERATION_OPTIONS,
                )
                wall_s = time.time() - started
                answer = (response.get("message", {}).get("content") or "").strip()
                timing = extract_timing(response, wall_s)
            except Exception as exc:
                error = str(exc)
            rows.append({
                "id": q["id"], "type": q["type"], "answerable": q["answerable"],
                "question": q["question"], "answer": answer,
                "reference_answer": q.get("reference_answer", ""),
                "sources": p["sources"],
                "refused": (engine._looks_like_non_answer(answer) if answer else None),
                "error": error,
                **timing,
            })
    finally:
        sampler.stop()
    resources = sampler.summary()
    error = None
    if not any(r["error"] is None for r in rows):
        error = rows[0]["error"] if rows else "no questions were asked"
        # Model-specific failures are normally returned by the API for individual questions
        # rather than raised by run_model(). Attach the installation hint at this level so
        # an all-question failure is reported consistently.
        if error and "not found" in error.lower() and "ollama pull" not in error.lower():
            error += f" (try: ollama pull {model})"
    return {"model": model, "rows": rows, "resources": resources, "error": error}


# aggregation and output ------------------------------------------------

def _mean(values):
    values = [v for v in values if v is not None]
    return (sum(values) / len(values)) if values else None


def _median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    n = len(values)
    mid = n // 2
    return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2


def summarise(cells):
    summary = []
    for cell in cells:
        base = {"model": cell["model"]}
        if cell.get("error") and not any(r["error"] is None for r in cell["rows"]):
            summary.append({**base, "error": cell["error"]})
            continue
        ok_rows = [r for r in cell["rows"] if r["error"] is None]
        n_errors = len(cell["rows"]) - len(ok_rows)
        res = cell["resources"]
        summary.append({
            **base, "n_questions": len(cell["rows"]), "n_errors": n_errors,
            "n_refused": sum(1 for r in ok_rows if r["refused"]),
            "mean_total_duration_s": _mean([r["total_duration_s"] for r in ok_rows]),
            "median_total_duration_s": _median([r["total_duration_s"] for r in ok_rows]),
            "mean_wall_s": _mean([r["wall_s"] for r in ok_rows]),
            "mean_load_duration_s": _mean([r["load_duration_s"] for r in ok_rows]),
            "mean_gen_tokens_per_s": _mean([r["gen_tokens_per_s"] for r in ok_rows]),
            "mean_prompt_tokens_per_s": _mean([r["prompt_tokens_per_s"] for r in ok_rows]),
            "cpu_pct_mean": res["cpu_pct"]["mean"], "cpu_pct_peak": res["cpu_pct"]["peak"],
            "ram_mb_mean": res["ram_mb"]["mean"], "ram_mb_peak": res["ram_mb"]["peak"],
            "resource_samples": res["n_samples"],
            "resource_process_matched": res["matched_any_process"],
            "error": cell.get("error") or "",
        })
    return summary


def write_csv(path, rows, first_columns=()):
    columns = list(first_columns)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v, suffix="", nd=2):
    return "-" if v is None else f"{v:.{nd}f}{suffix}"


def render_markdown(summary, sample_interval):
    ok = [r for r in summary if not r.get("error")]
    failed = [r for r in summary if r.get("error")]
    ok.sort(key=lambda r: (r.get("mean_total_duration_s") is None, r.get("mean_total_duration_s", 0)))
    lines = ["# Experiment 2 -- generation model comparison", "",
             "Retrieval (embedding model, chunking, prompt, GENERATION_OPTIONS) is IDENTICAL for "
             "every row below; only the model differs. Timing comes from Ollama's own response "
             "metadata, not Python wall-clock. CPU/RAM are sampled every "
             f"{sample_interval}s across each model's full run -- see config.json for whether "
             "sampling was system-wide or restricted to named processes.", "",
             "| Model | mean latency (s) | median latency (s) | gen tok/s | load (s) | "
             "CPU % (mean/peak) | RAM MB (mean/peak) | refused | errors |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in ok:
        cpu = f"{_fmt(r['cpu_pct_mean'], nd=1)} / {_fmt(r['cpu_pct_peak'], nd=1)}"
        ram = f"{_fmt(r['ram_mb_mean'], nd=0)} / {_fmt(r['ram_mb_peak'], nd=0)}"
        lines.append(
            f"| {r['model']} | {_fmt(r['mean_total_duration_s'])} | {_fmt(r['median_total_duration_s'])} | "
            f"{_fmt(r['mean_gen_tokens_per_s'], nd=1)} | {_fmt(r['mean_load_duration_s'])} | "
            f"{cpu} | {ram} | {r['n_refused']}/{r['n_questions']} | {r['n_errors']} |")
    if failed:
        lines += ["", "## Models that failed", ""]
        lines += [f"- {r['model']}: {r['error']}" for r in failed]
    if ok and not all(r["resource_process_matched"] for r in ok):
        lines += ["", "**Warning:** --process-name matched no process for at least one model; its "
                      "CPU/RAM are meaningless zeros, not a real measurement. Check the process "
                      "name in Task Manager and re-run, or drop --process-name for a system-wide "
                      "(less precise, but never silently wrong) measurement."]
    lines += ["", "Read `generations.csv` and fill in `manual_relevance` / `manual_quality` "
                  "(no automatic score exists for either) to complete the rubric half of this "
                  "experiment."]
    return "\n".join(lines)


# CLI -----------------------------------------------------------------------
def _cell_complete(path):
    """
    Return True when a saved cell contains at least one successful question.

    Cells with partial failures can be reused because successful results are retained.
    A cell in which every question failed is treated as incomplete.
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            cell = json.load(fh)
    except (OSError, ValueError):
        return False
    return not cell.get("error") or any(r.get("error") is None for r in cell.get("rows", []))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Experiment 2: compare LLM generation models with retrieval held fixed")
    ap.add_argument("--docs", required=True, help="folder containing the experiment documents")
    ap.add_argument("--questions", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "experiment1_questions.csv"),
                    help="reuses Experiment 1's question file; evidence is not required here")
    ap.add_argument("--out", default="results/exp2")
    ap.add_argument("--models", nargs="+", required=True,
                    help="exact Ollama tags, e.g. llama3.2 phi3:mini gemma3:4b -- run 'ollama list' "
                         "to get the exact strings you have pulled; use 'ollama list' to verify the installed tags")
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ap.add_argument("--chunking", default=DEFAULT_CHUNKING)
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    ap.add_argument("--process-name", nargs="+", default=None,
                    help="restrict CPU/RAM sampling to processes with this exact name (case-"
                         "insensitive), e.g. --process-name ollama ollama_llama_server. Omit for "
                         "system-wide sampling (the default).")
    ap.add_argument("--sample-interval", type=float, default=DEFAULT_SAMPLE_INTERVAL)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    try:
        questions, warnings = exp1.load_questions(args.questions)
    except (exp1.QuestionFileError, OSError) as exc:
        print(f"ERROR reading {args.questions}:\n{exc}")
        return 2
    for w in warnings:
        if "UNSCORED" not in w:   # Experiment 2 asks every question regardless of evidence
            print(f"WARNING: {w}")

    if not os.path.isdir(args.docs):
        print(f"ERROR: --docs '{args.docs}' is not a folder")
        return 2
    paths, problems = exp1.discover_docs(args.docs)
    if problems:
        print("ERROR: these files failed validation and would not be accepted by the app either:")
        for p in problems:
            print(f"  - {p}")
        return 2
    if not paths:
        print("ERROR: no supported documents found in --docs")
        return 2
    uncovered = exp1.check_doc_coverage(questions, [os.path.basename(p) for p in paths])
    if uncovered:
        print(f"ERROR: no document filename contains the course code(s): {', '.join(uncovered)}")
        return 2
    print("Documents (every one is indexed):")
    for p in paths:
        print(f"  - {os.path.basename(p)}")
    print(f"Models to compare: {', '.join(args.models)}")
    print(f"Fixed retrieval: embedding={args.embedding_model}, chunking={args.chunking}")

    os.makedirs(args.out, exist_ok=True)
    print("Extracting text (cached after the first run)...")
    doc_texts = exp1.extract_all(paths, os.path.join(args.out, "extracted"))
    for name, text in doc_texts.items():
        print(f"  {name}: {len(text):,} characters")
        if not text.strip():
            print(f"WARNING: no text extracted from {name}")

    if args.dry_run:
        print(f"\nDry run OK: {len(questions)} questions, {len(args.models)} models. Nothing was indexed, "
              f"no model was called.")
        return 0

    # Configure the experiment-specific ChromaDB path before importing rag.
    # rag initialises its persistent client at import time.
    chroma_path = os.path.abspath(os.path.join(args.out, "chroma_exp2"))
    os.environ["CHROMA_DB_PATH"] = chroma_path
    import chromadb
    import rag
    import engine
    # Reassign the client as an additional safeguard if rag was imported previously.
    rag.client = chromadb.PersistentClient(path=chroma_path)

    if args.embedding_model not in rag.EMBEDDING_MODELS:
        print(f"ERROR: unknown embedding model '{args.embedding_model}' (choose from {', '.join(rag.EMBEDDING_MODELS)})")
        return 2
    if args.chunking not in rag.CHUNKING_STRATEGIES and args.chunking != "sliding_window":
        print(f"ERROR: unknown chunking strategy '{args.chunking}'")
        return 2

    user = "exp2"
    rag.reset_collection(user, args.embedding_model)
    for name, text in doc_texts.items():
        chunks = rag.chunk_text(text, max_chunk_size=args.chunk_size, strategy=args.chunking, overlap=args.overlap)
        if chunks:
            rag.add_documents(user, chunks, metadatas=[{"source": name} for _ in chunks],
                              embedding_model=args.embedding_model)

    print("Retrieving and building the fixed prompt for every question (shared by all models)...")
    prompts = build_fixed_prompts(engine, rag, user, questions, args.embedding_model)

    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "started": datetime.now().isoformat(timespec="seconds"),
            "models": args.models, "embedding_model": args.embedding_model,
            "chunking": args.chunking, "chunk_size": args.chunk_size, "overlap": args.overlap,
            "retrieval_top_k": engine.RETRIEVAL_TOP_K, "context_chunks": engine.CONTEXT_CHUNKS,
            "generation_options": engine.GENERATION_OPTIONS,
            "process_name": args.process_name, "sample_interval": args.sample_interval,
            "documents": {os.path.basename(p): validation.sha256_file(p) for p in paths},
            "n_questions": len(questions), "versions": exp1._versions(),
        }, fh, indent=2)

    def sampler_factory():
        return ResourceSampler(interval=args.sample_interval, process_names=args.process_name)

    cells_dir = os.path.join(args.out, "cells")
    os.makedirs(cells_dir, exist_ok=True)
    cells = []
    for n, model in enumerate(args.models, start=1):
        label = f"[{n}/{len(args.models)}] {model}"
        path = os.path.join(cells_dir, f"{_sanitize_filename(model)}.json")
        if args.resume and _cell_complete(path):
            with open(path, encoding="utf-8") as fh:
                cells.append(json.load(fh))
            print(f"{label}: reused from a previous run")
            continue
        print(f"{label}: asking {len(questions)} questions...")
        try:
            cell = run_model(engine, model, prompts, questions, sampler_factory)
        except Exception as exc:
            hint = f" (try: ollama pull {model})" if "not found" in str(exc).lower() else ""
            cell = {"model": model, "rows": [], "resources": ResourceSampler().summary(),
                    "error": f"{type(exc).__name__}: {exc}{hint}"}
        if cell.get("error") and not any(r.get("error") is None for r in cell.get("rows", [])):
            print(f"{label}: FAILED -- {cell['error']}")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cell, fh, indent=1, default=str)
        cells.append(cell)

    summary = summarise(cells)
    write_csv(os.path.join(args.out, "summary.csv"), summary, first_columns=("model",))
    per_question = [{"model": c["model"], **r} for c in cells for r in c["rows"]]
    write_csv(os.path.join(args.out, "per_question.csv"), per_question, first_columns=("model", "id"))

    generations = []
    for c in cells:
        for r in c["rows"]:
            generations.append({
                "model": c["model"], "id": r["id"], "type": r["type"], "answerable": r["answerable"],
                "question": r["question"], "answer": r["answer"], "reference_answer": r["reference_answer"],
                "refused": r["refused"], "sources": r["sources"], "error": r["error"],
                "manual_relevance": "", "manual_quality": "", "notes": "",
            })
    write_csv(os.path.join(args.out, "generations.csv"), generations, first_columns=("model", "id"))

    md = render_markdown(summary, args.sample_interval)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    print(f"Results written to {os.path.abspath(args.out)}")
    # A model is considered failed only when every question failed.
    # Partial per-question failures remain in the output and do not invalidate the cell.
    return 1 if any(c.get("error") and not any(r.get("error") is None for r in c.get("rows", [])) for c in cells) else 0


if __name__ == "__main__":
    sys.exit(main())