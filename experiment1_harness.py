"""
Experiment 1 harness: embedding models x chunking strategy evaluation

HOW TO RUN (If no file names are changed):
    Dry Run (ensure requirements are met):
    python experiment1_harness.py --docs assets --questions experiment1_questions.csv --out results/exp1 --dry-run
    
    Full Run (`Dry run OK: 31 questions, 9 cells. Nothing was indexed.`):
    python experiment1_harness.py --docs assets --questions experiment1_questions.csv --out results/exp1
    
    NOTE: add `--resume` to continue from previous run, skipping 
    results that have already been collected 

WHAT IT MEASURES (primary evaluation is retrieval-based; LLM generation is optional)

    For each of the 9 (embedding model, chunking strategy) cells it indexes every
    document in --docs into one collection (so cross-document confusion counts, as
    in the live app), asks every question, and scores the retrieved chunks against
    per-question EVIDENCE PHRASES: short strings that must appear in the text for
    the question to be answerable from it (e.g. "thinking humanly").

    Evidence phrases are used instead of reference chunks because different chunking
    strategies produce different chunk boundaries. Phrase-based evaluation therefore
    provides a chunk-independent retrieval criterion: each configuration is evaluated
    on whether its retrieved context contains the required information, regardless
    of chunk boundaries.

    Metrics, per cell, over the answerable questions that have evidence:
      any_hit@k     at least one evidence phrase occurs in the top-k retrieved chunks (lenient)
      full_hit@k    all evidence phrases occur in the top-k retrieved chunks          (strict)
      coverage@k    mean fraction of evidence phrases found in the top-k
      source_hit@k  a chunk from the correct document is in the top-k
      mrr           mean reciprocal rank of the first chunk with evidence (0 if none in top-10)
    
    k values are 1, 3, engine.CONTEXT_CHUNKS (what the LLM actually sees) and
    engine.RETRIEVAL_TOP_K.
    
    The 95% CI for any_hit@CONTEXT_CHUNKS is estimated by bootstrapping questions.
    Because the evaluation set is small, differences between configurations should
    be interpreted together with their confidence intervals.

WHAT IT DELIBERATELY DOES NOT DO
    * Answer quality is evaluated separately through manual rubric scoring; automatic 
      generation output is provided only as supporting evidence. --generate dumps 
      each cell's LLM answers to generations.csv with blank columns for the rubrics; 
      The only automatic generation metric is evidence-phrase overlap, which is retained
      as a diagnostic proxy rather than treated as a measure of answer quality.
      The only automatic generation metric is evidence-phrase overlap. It is retained
      as a diagnostic proxy and is not treated as a measure of answer quality.
      
    * The harness does not generate or infer reference evidence. Questions without valid 
      evidence phrases are excluded from retrieval scoring.
    
    * The unanswerable question (Q31) has no retrieval metric. Its top-1 retrieval 
      distance is recorded for descriptive analysis only and is excluded from 
      retrieval metrics. (Retrieval distances are embedding-model-specific and should 
      thus only be compared within the same embedding configuration, not across models).

FAIRNESS
    To isolate the effects of embedding model and chunking strategy, all other experimental 
    parameters are held constant across configurations, including chunk size, overlap, 
    retrieval depth, context size, prompt, and generation options.

EXPERIMENT ISOLATION
    The experiment uses an isolated ChromaDB directory under --out and configures 
    CHROMA_DB_PATH before importing rag, preventing the experiment from modifying 
    the application's default vector store.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
import unicodedata
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validation  

EXPERIMENT_USER = "exp1"
DEFAULT_MODELS = ("bge-small", "minilm", "nomic")
DEFAULT_STRATEGIES = ("paragraph", "fixed", "sliding_window")
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100

EVIDENCE_SEPARATOR = "||"
REQUIRED_COLUMNS = ["id", "doc", "type", "question", "answerable", "evidence"]
OPTIONAL_COLUMNS = ["reference_answer"]
VALID_TYPES = {"definitional", "factual", "conceptual", "unanswerable"}
MIN_EVIDENCE_CHARS = 4   # shorter phrases match by accident ("ai" is inside "said")
_TRUE = {"yes", "y", "true", "1"}
_FALSE = {"no", "n", "false", "0"}


class QuestionFileError(ValueError):
    """The questions CSV is structurally wrong (every problem is listed at once)."""


# text normalisation and evidence matching ---------------------------------------


def normalise(text):
    """
    Reduces text to lowercase ASCII alphanumerics separated by single spaces, so
    an evidence phrase matches regardless of case, punctuation, line breaks,
    hyphenation across lines, or PDF ligatures.

    NFKC normalisation converts compatibility characters such as typographic
    ligatures into their canonical text representation, improving matching against
    PDF-extracted text. Words split by a line-break hyphen are also rejoined so
    that PDF line wrapping does not prevent evidence matching.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"(\w)-[ \t]*\r?\n[ \t]*(\w)", r"\1\2", text)
    text = re.sub(r"[^0-9a-z]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _squash(text):
    """
    Removes non-alphanumeric characters for document-code matching
    """
    return re.sub(r"[^0-9a-z]+", "", (text or "").lower())


def doc_matches(source, doc_code):
    """
    Returns True if the normalised course code occurs in the 
    normalised document filename
    """
    code = _squash(doc_code)
    return bool(code) and code in _squash(source)


def phrases_found(norm_text, norm_phrases):
    # Returns the indices of normalised evidence phrases that occur in norm_text
    return {i for i, p in enumerate(norm_phrases) if p and p in norm_text}


# scoring -------------------------------------------------------------------------

def score_retrieval(results, evidence, doc_code, ks):
    """
    Compute retrieval metrics for a ranked result list.

    Args:
        results: Retrieved chunks ordered from highest to lowest relevance.
        evidence: Evidence phrases required by the question.
        doc_code: Expected source document code.
        ks: Retrieval depths at which metrics are calculated.
    """
    norm_phrases = [p for p in (normalise(e) for e in evidence) if p]
    per_chunk = [phrases_found(normalise(r["text"]), norm_phrases) for r in results]

    first_hit_rank = next((rank for rank, found in enumerate(per_chunk, 1) if found), None)
    source_ranks = [rank for rank, r in enumerate(results, 1)
                    if doc_matches(r.get("source"), doc_code)]

    out = {
        "first_hit_rank": first_hit_rank,
        "reciprocal_rank": (1.0 / first_hit_rank) if first_hit_rank else 0.0,
        "first_source_rank": source_ranks[0] if source_ranks else None,
    }
    for k in ks:
        union = set()
        for found in per_chunk[:k]:
            union |= found
        n = len(norm_phrases)
        out[f"coverage@{k}"] = (len(union) / n) if n else None
        out[f"any_hit@{k}"] = int(bool(union))
        out[f"full_hit@{k}"] = int(n > 0 and len(union) == n)
        out[f"source_hit@{k}"] = int(any(r <= k for r in source_ranks))
    return out


def bootstrap_ci(values, n_boot=1000, seed=0, alpha=0.05):
    """
    Returns a percentile bootstrap confidence interval for the sample mean.
    The result is reproducible when a fixed seed is provided.
    """
    values = list(values)
    if not values:
        return (None, None)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def _mean(values):
    values = [v for v in values if v is not None]
    return (sum(values) / len(values)) if values else None


# questions and documents --------------------------------------------------------------

def load_questions(path):
    """
    Loads and validates the experiment question file. All structural validation  
    errors are collected and reported together so that multiple input problems
    can be corrected in a single run.
    """
    errors, warnings, questions, seen_ids = [], [], [], set()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise QuestionFileError(f"questions file is missing column(s): {', '.join(missing)}")
        rows = list(reader)

    for line_no, raw in enumerate(rows, start=2):
        row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
        where = f"row {line_no}"
        if not any(row.values()):
            continue

        qid_raw = row.get("id", "")
        if not qid_raw.isdigit():
            errors.append(f"{where}: id '{qid_raw}' is not a whole number")
            continue
        qid = int(qid_raw)
        if qid in seen_ids:
            errors.append(f"{where}: duplicate id {qid}")
        seen_ids.add(qid)

        for col in ("doc", "question"):
            if not row.get(col):
                errors.append(f"{where} (id {qid}): '{col}' is empty")

        ans_raw = row.get("answerable", "").lower()
        if ans_raw in _TRUE:
            answerable = True
        elif ans_raw in _FALSE:
            answerable = False
        else:
            errors.append(f"{where} (id {qid}): answerable must be yes/no, got '{row.get('answerable', '')}'")
            continue

        qtype = row.get("type", "").lower()
        if qtype not in VALID_TYPES:
            errors.append(f"{where} (id {qid}): type '{row.get('type', '')}' must be one of "
                          f"{', '.join(sorted(VALID_TYPES))}")
            continue
        if answerable and qtype == "unanswerable":
            errors.append(f"{where} (id {qid}): type 'unanswerable' contradicts answerable=yes")
        if not answerable and qtype != "unanswerable":
            errors.append(f"{where} (id {qid}): answerable=no requires type 'unanswerable'")

        evidence = [e.strip() for e in row.get("evidence", "").split(EVIDENCE_SEPARATOR) if e.strip()]
        for e in evidence:
            if not normalise(e):
                errors.append(f"{where} (id {qid}): evidence phrase '{e}' has no letters or digits")
            elif len(normalise(e)) < MIN_EVIDENCE_CHARS:
                warnings.append(f"id {qid}: evidence phrase '{e}' is very short and may produce false-positive substring matches")
        if not answerable and evidence:
            warnings.append(f"id {qid}: evidence on an unanswerable question is ignored")
            evidence = []

        questions.append({
            "id": qid, "doc": row.get("doc", ""), "type": qtype,
            "question": row.get("question", ""), "answerable": answerable,
            "evidence": evidence, "reference_answer": row.get("reference_answer", ""),
        })

    if errors:
        raise QuestionFileError("\n".join(errors))

    unscored = [q["id"] for q in questions if q["answerable"] and not q["evidence"]]
    if unscored:
        warnings.append(f"{len(unscored)} answerable question(s) have no evidence and will be "
                        f"UNSCORED (excluded from every metric): ids {unscored}")
    return questions, warnings


def discover_docs(folder):
    """ 
    Discovers supported documents directly within folder and validates 
    them using the same upload-validation rules as the application
    """
    paths, problems = [], []
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if not os.path.isfile(full):
            continue
        if os.path.splitext(name)[1].lower() not in validation.ALLOWED_DOC_EXTENSIONS:
            continue
        ok, _, why = validation.validate_upload(full)
        if ok:
            paths.append(full)
        else:
            problems.append(f"{name}: {why}")
    return paths, problems


def check_doc_coverage(questions, doc_names):
    """
    Course codes used in the questions that match no document
    """
    codes = sorted({q["doc"] for q in questions})
    return [c for c in codes if not any(doc_matches(n, c) for n in doc_names)]


def extract_all(paths, cache_dir, extractor=None):
    """
    Extracts each document once and caches the result using the document hash. 
    All experimental cells reuse the same extracted text, ensuring that document
    extraction does not become a configuration-dependent variable.
    """
    if extractor is None:
        from material_processor import process_file as extractor
    os.makedirs(cache_dir, exist_ok=True)
    texts = {}
    for path in paths:
        name = os.path.basename(path)
        digest = validation.sha256_file(path)
        txt_path = os.path.join(cache_dir, name + ".txt")
        sha_path = txt_path + ".sha256"
        if os.path.exists(txt_path) and os.path.exists(sha_path):
            with open(sha_path, encoding="utf-8") as fh:
                if fh.read().strip() == digest:
                    with open(txt_path, encoding="utf-8") as tf:
                        texts[name] = tf.read()
                    continue
        text = extractor(path)
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        with open(sha_path, "w", encoding="utf-8") as fh:
            fh.write(digest)
        texts[name] = text
    return texts


def validate_evidence(questions, doc_texts):
    """
    Evidence phrases that do NOT occur in their document's extracted text.
    
    Evidence phrases absent from the extracted source text cannot be retrieved by
    any experimental configuration. They would therefore introduce a systematic
    scoring error unrelated to retrieval performance and are reported before the
    experiment begins.
    """
    normalised_docs = {name: normalise(text) for name, text in doc_texts.items()}
    missing = []
    for q in questions:
        if not (q["answerable"] and q["evidence"]):
            continue
        haystack = " ".join(t for n, t in normalised_docs.items() if doc_matches(n, q["doc"]))
        for phrase in q["evidence"]:
            if normalise(phrase) not in haystack:
                missing.append((q["id"], q["doc"], phrase))
    return missing


# running one cell ----------------------------------------------------------------

def run_cell(rag, model, strategy, doc_texts, questions, ks, chunk_size, overlap,
             generate_fn=None, user=EXPERIMENT_USER):
    """
    Indexes the corpus for one embedding-model/chunking-strategy 
    configuration and evaluates all experiment questions
    """
    rag.reset_collection(user, model)

    t0 = time.time()
    n_chunks, chunk_chars = 0, []
    for name, text in doc_texts.items():
        chunks = rag.chunk_text(text, max_chunk_size=chunk_size, strategy=strategy, overlap=overlap)
        if not chunks:
            continue
        rag.add_documents(user, chunks, metadatas=[{"source": name} for _ in chunks],
                          embedding_model=model)
        n_chunks += len(chunks)
        chunk_chars.extend(len(c) for c in chunks)
    index_seconds = time.time() - t0

    rows, generations = [], []
    for q in questions:
        t = time.time()
        results = rag.retrieve_with_sources(user, q["question"], top_k=max(ks), embedding_model=model)
        retrieval_ms = (time.time() - t) * 1000

        row = {
            "id": q["id"], "doc": q["doc"], "type": q["type"], "answerable": q["answerable"],
            "n_evidence": len(q["evidence"]), "retrieval_ms": retrieval_ms,
            "top1_distance": results[0]["distance"] if results else None,
            "top1_source": results[0]["source"] if results else None,
            "scored": bool(q["answerable"] and q["evidence"]),
        }
        if row["scored"]:
            row.update(score_retrieval(results, q["evidence"], q["doc"], ks))
        rows.append(row)

        if generate_fn is not None:
            generations.append(generate_fn(model, strategy, q, results))

    return {
        "model": model, "strategy": strategy, "n_chunks": n_chunks,
        "mean_chunk_chars": _mean(chunk_chars), "index_seconds": index_seconds,
        "rows": rows, "generations": generations, "error": None,
    }


def make_generate_fn(engine):
    """
    Builds the generation step. Replicates the application's generation 
    configuration including the system prompt, retrieved-context 
    formatting, user message, and generation options while using 
    a fixed neutral learner profile.
    
    A fixed neutral learner profile is used to prevent personalisation 
    settings from becoming a confounding variable.
    
    The corresponding test verifies that the harness prompt remains 
    consistent with the application's prompt, preventing unintended 
    experimental divergence.
    """
    system_msg = engine._build_system_prompt("clear and detailed", "intermediate")

    def generate(model, strategy, q, results):
        used = results[:engine.CONTEXT_CHUNKS]
        user_msg = (f"Excerpts from the student's uploaded materials:\n\n"
                    f"{engine._build_context(used)}\n\nQuestion: {q['question']}")
        started = time.time()
        answer, error = "", None
        try:
            response = engine._client.chat(
                model=engine.DEFAULT_MODEL,
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": user_msg}],
                options=engine.GENERATION_OPTIONS,
            )
            answer = (response["message"]["content"] or "").strip()
        except Exception as exc:  
            # Isolate generation failures so one failed configuration does not discard
            # results from the remaining experimental cells.
            error = str(exc)
        norm_answer = normalise(answer)
        norm_phrases = [p for p in (normalise(e) for e in q["evidence"]) if p]
        overlap = (len(phrases_found(norm_answer, norm_phrases)) / len(norm_phrases)
                   if norm_phrases and answer else None)
        return {
            "model": model, "strategy": strategy, "id": q["id"], "type": q["type"],
            "answerable": q["answerable"], "question": q["question"], "answer": answer,
            "refused": (engine._looks_like_non_answer(answer) if answer else None),
            "answer_phrase_overlap_PROXY": overlap,
            "seconds": time.time() - started, "error": error,
            "reference_answer": q["reference_answer"],
        }
    return generate



# aggregation and output --------------------------------------------------------------

def summarise(cells, ks, context_k, seed=0):
    """
    Produces one aggregate row for each embedding-model/chunking-strategy
    configuration and an additional breakdown by question type.
    """
    summary, by_type = [], []
    for cell in cells:
        base = {"model": cell["model"], "strategy": cell["strategy"]}
        if cell.get("error"):
            summary.append({**base, "error": cell["error"]})
            continue
        scored = [r for r in cell["rows"] if r["scored"]]
        answerable_d = [r["top1_distance"] for r in cell["rows"] if r["answerable"] and r["top1_distance"] is not None]
        unanswerable_d = [r["top1_distance"] for r in cell["rows"] if not r["answerable"] and r["top1_distance"] is not None]
        row = {**base, "n_scored": len(scored), "n_chunks": cell["n_chunks"],
               "mean_chunk_chars": cell["mean_chunk_chars"],
               "index_seconds": cell["index_seconds"],
               "mean_query_ms": _mean([r["retrieval_ms"] for r in cell["rows"]]),
               "mrr": _mean([r["reciprocal_rank"] for r in scored])}
        for k in ks:
            for metric in ("any_hit", "full_hit", "coverage", "source_hit"):
                row[f"{metric}@{k}"] = _mean([r[f"{metric}@{k}"] for r in scored])
        lo, hi = bootstrap_ci([r[f"any_hit@{context_k}"] for r in scored], seed=seed)
        row[f"any_hit@{context_k}_ci_lo"], row[f"any_hit@{context_k}_ci_hi"] = lo, hi
        row["mean_top1_dist_answerable"] = _mean(answerable_d)
        row["mean_top1_dist_unanswerable"] = _mean(unanswerable_d)
        row["error"] = ""
        summary.append(row)

        for qtype in sorted({r["type"] for r in scored}):
            subset = [r for r in scored if r["type"] == qtype]
            by_type.append({**base, "type": qtype, "n": len(subset),
                            f"any_hit@{context_k}": _mean([r[f"any_hit@{context_k}"] for r in subset]),
                            f"coverage@{context_k}": _mean([r[f"coverage@{context_k}"] for r in subset])})
    return summary, by_type


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


def _pct(v):
    return "-" if v is None else f"{100 * v:.0f}%"


def render_markdown(summary, context_k, n_scored, unscored_ids, n_unanswerable):
    ok = [r for r in summary if not r.get("error")]
    failed = [r for r in summary if r.get("error")]
    # Sort for presentation only, the ordering is not used for statistical comparison
    ok.sort(key=lambda r: (-(r.get(f"any_hit@{context_k}") or 0), -(r.get("mrr") or 0)))
    lines = [f"# Experiment 1 -- retrieval comparison", "",
             f"Scored questions: **{n_scored}** (answerable, with evidence). "
             f"One question = {100 / n_scored:.1f} percentage points, so gaps of a few points are "
             f"within noise." if n_scored else "No scored questions.", ""]
    if unscored_ids:
        lines += [f"UNSCORED (no evidence supplied): ids {unscored_ids}", ""]
    lines += [f"Unanswerable questions (no retrieval metric): {n_unanswerable}", "",
              f"Sorted by any_hit@{context_k}, then MRR. **Sort order is not a significance test.** "
              f"@{context_k} is the number of chunks the LLM is shown.", "",
              f"| Model | Chunking | any@{context_k} (95% CI) | full@{context_k} | coverage@{context_k} | "
              f"source@{context_k} | MRR | chunks | index s | ms/query |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in ok:
        ci = f"{_pct(r.get(f'any_hit@{context_k}'))} ({_pct(r.get(f'any_hit@{context_k}_ci_lo'))}-{_pct(r.get(f'any_hit@{context_k}_ci_hi'))})"
        lines.append(
            f"| {r['model']} | {r['strategy']} | {ci} | {_pct(r.get(f'full_hit@{context_k}'))} | "
            f"{_pct(r.get(f'coverage@{context_k}'))} | {_pct(r.get(f'source_hit@{context_k}'))} | "
            f"{r['mrr']:.2f} | {r['n_chunks']} | {r['index_seconds']:.1f} | {r['mean_query_ms']:.0f} |")
    if failed:
        lines += ["", "## Cells that failed", ""]
        lines += [f"- {r['model']} / {r['strategy']}: {r['error']}" for r in failed]
    lines += ["", "Distances (in per_question.csv) are model-specific: compare within a cell, never across models.", ""]
    return "\n".join(lines)


def _versions():
    """
    Return the Python and package versions used for reproducibility
    """
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for pkg in ("chromadb", "sentence-transformers", "ollama", "torch"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


# CLI ---------------------------------------------------------------------------

def _cell_complete(path, want_generation):
    """
    Return whether a saved cell contains usable results for the requested run mode
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            cell = json.load(fh)
    except (OSError, ValueError):
        return False
    if cell.get("error"):
        return False
    return (not want_generation) or bool(cell.get("generations"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Experiment 1: embedding model x chunking strategy")
    ap.add_argument("--docs", required=True, help="folder containing the experiment documents")
    ap.add_argument("--questions", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "experiment1_questions.csv"))
    ap.add_argument("--out", default="results/exp1")
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    ap.add_argument("--generate", action="store_true",
                    help="also have the LLM answer every question in every cell (SLOW: hours on CPU)")
    ap.add_argument("--resume", action="store_true", help="reuse cells already completed in --out")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the questions, documents and evidence, then stop before indexing")
    ap.add_argument("--allow-missing-evidence", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    for s in args.strategies:
        if s not in DEFAULT_STRATEGIES:
            print(f"ERROR: unknown chunking strategy '{s}' (choose from {', '.join(DEFAULT_STRATEGIES)})")
            return 2

    try:
        questions, warnings = load_questions(args.questions)
    except (QuestionFileError, OSError) as exc:
        print(f"ERROR reading {args.questions}:\n{exc}")
        return 2
    for w in warnings:
        print(f"WARNING: {w}")

    if not os.path.isdir(args.docs):
        print(f"ERROR: --docs '{args.docs}' is not a folder")
        return 2
    paths, problems = discover_docs(args.docs)
    if problems:
        print("ERROR: these files failed validation and would not be accepted by the app either:")
        for p in problems:
            print(f"  - {p}")
        return 2
    if not paths:
        print("ERROR: no supported documents found in --docs")
        return 2
    uncovered = check_doc_coverage(questions, [os.path.basename(p) for p in paths])
    if uncovered:
        print(f"ERROR: no document filename contains the course code(s): {', '.join(uncovered)}")
        return 2
    print("Documents (every one is indexed):")
    for p in paths:
        print(f"  - {os.path.basename(p)}")

    os.makedirs(args.out, exist_ok=True)
    print("Extracting text (cached after the first run)...")
    doc_texts = extract_all(paths, os.path.join(args.out, "extracted"))
    for name, text in doc_texts.items():
        print(f"  {name}: {len(text):,} characters")
        if not text.strip():
            print(f"WARNING: no text extracted from {name}")

    missing = validate_evidence(questions, doc_texts)
    if missing:
        print("\nEVIDENCE NOT FOUND in the extracted document text (fix the phrase or the document):")
        for qid, doc, phrase in missing:
            print(f"  Q{qid} [{doc}]: '{phrase}'")
        if not args.allow_missing_evidence:
            print("Aborting: such a phrase can never be found by any configuration, so it would "
                  "silently score as a miss in all cells. Use --allow-missing-evidence to override.")
            return 2

    if args.dry_run:
        n_cells = len(args.models) * len(args.strategies)
        print(f"\nDry run OK: {len(questions)} questions, {n_cells} cells. Nothing was indexed.")
        if args.generate:
            print(f"--generate would make {n_cells * len(questions)} LLM calls "
                  f"(at ~45-50 s each on CPU that is roughly {n_cells * len(questions) * 47 / 3600:.1f} hours).")
        return 0

    # Configure an isolated ChromaDB path before importing rag because rag creates
    # its persistent client at import time. Reassign rag.client below as a safeguard
    # when rag has already been imported by the caller.
    chroma_path = os.path.abspath(os.path.join(args.out, "chroma_exp1"))
    os.environ["CHROMA_DB_PATH"] = chroma_path
    import chromadb
    import rag
    import engine
    for m in args.models:
        if m not in rag.EMBEDDING_MODELS:
            print(f"ERROR: unknown embedding model '{m}' (choose from {', '.join(rag.EMBEDDING_MODELS)})")
            return 2
    rag.client = chromadb.PersistentClient(path=chroma_path)
    # Evaluate retrieval at fixed diagnostic depths plus the depths used by the application
    ks = sorted({1, 3, engine.CONTEXT_CHUNKS, engine.RETRIEVAL_TOP_K})
    context_k = engine.CONTEXT_CHUNKS
    generate_fn = make_generate_fn(engine) if args.generate else None
    
    # Record all configuration and environment information required to reproduce the run.
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "started": datetime.now().isoformat(timespec="seconds"),
            "models": args.models, "strategies": args.strategies,
            "chunk_size": args.chunk_size, "overlap": args.overlap, "ks": ks,
            "retrieval_top_k": engine.RETRIEVAL_TOP_K, "context_chunks": engine.CONTEXT_CHUNKS,
            "generation_options": engine.GENERATION_OPTIONS, "llm": engine.DEFAULT_MODEL,
            "generate": args.generate, "seed": args.seed,
            "documents": {os.path.basename(p): validation.sha256_file(p) for p in paths},
            "n_questions": len(questions), "versions": _versions(),
        }, fh, indent=2)

    cells_dir = os.path.join(args.out, "cells")
    os.makedirs(cells_dir, exist_ok=True)
    cells, total = [], len(args.models) * len(args.strategies)
    n = 0
    for model in args.models:
        for strategy in args.strategies:
            n += 1
            path = os.path.join(cells_dir, f"{model}__{strategy}.json")
            label = f"[{n}/{total}] {model} x {strategy}"
            if args.resume and _cell_complete(path, args.generate):
                with open(path, encoding="utf-8") as fh:
                    cells.append(json.load(fh))
                print(f"{label}: reused from a previous run")
                continue
            print(f"{label}: indexing and querying...")
            # Isolate cell-level failures so one configuration does not 
            # prevent the remaining configurations from completing
            try:
                cell = run_cell(rag, model, strategy, doc_texts, questions, ks,
                                args.chunk_size, args.overlap, generate_fn)
            except Exception as exc:  
                hint = " (try: pip install einops)" if "einops" in str(exc).lower() else ""
                cell = {"model": model, "strategy": strategy, "error": f"{type(exc).__name__}: {exc}{hint}"}
                print(f"{label}: FAILED -- {cell['error']}")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(cell, fh, indent=1, default=str)
            cells.append(cell)

    summary, by_type = summarise(cells, ks, context_k, seed=args.seed)
    write_csv(os.path.join(args.out, "summary.csv"), summary, first_columns=("model", "strategy"))
    write_csv(os.path.join(args.out, "by_type.csv"), by_type, first_columns=("model", "strategy", "type"))
    per_question = [{"model": c["model"], "strategy": c["strategy"], **r}
                    for c in cells if not c.get("error") for r in c["rows"]]
    write_csv(os.path.join(args.out, "per_question.csv"), per_question, first_columns=("model", "strategy", "id"))
    if args.generate:
        gens = [g for c in cells if not c.get("error") for g in c["generations"]]
        for g in gens:
            g["manual_score"] = ""
            g["notes"] = ""
        write_csv(os.path.join(args.out, "generations.csv"), gens, first_columns=("model", "strategy", "id"))

    n_scored = sum(1 for q in questions if q["answerable"] and q["evidence"])
    unscored = [q["id"] for q in questions if q["answerable"] and not q["evidence"]]
    n_unanswerable = sum(1 for q in questions if not q["answerable"])
    md = render_markdown(summary, context_k, n_scored, unscored, n_unanswerable)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    print(f"Results written to {os.path.abspath(args.out)}")
    return 1 if any(c.get("error") for c in cells) else 0


if __name__ == "__main__":
    sys.exit(main())