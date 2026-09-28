# CM3070 Multimodal AI Learning Assistant

A local-only RAG (Retrieval-Augmented Generation) study assistant. Upload lecture material
(PDF, Word, PowerPoint, or images), ask questions about uploaded material and receive answers 
intended to be grounded in that material. No cloud AI service is used - retrieval, embeddings, 
and generation all run locally, on your own machine via Ollama.

## How it works

```
Upload -> extract text (PyPDF2 / python-docx / python-pptx / PaddleOCR for images and scans)
       -> chunk -> embed (sentence-transformers) -> store (ChromaDB)
Ask    -> retrieve relevant chunks -> build a prompt (personalised to user profile)
       -> answer (Ollama: Llama 3.2 by default; LLaVA for image questions)
Rate   -> 1-5 understanding rating -> low-rated documents get flagged -> revision suggestions
```

SQLite (`learner.db`) stores the profile and interaction history. ChromaDB (`chroma_db/`)
stores the document embeddings. Both are created automatically on first run.

## Project structure

```
app.py                     Gradio UI and wiring
engine.py                  Retrieval-augmented answer generation, personalisation, clarification
rag.py                     Chunking, embedding, ChromaDB storage/retrieval
material_processor.py      Text/OCR extraction from PDF, docx, pptx, images
db.py                      SQLite: profile, interaction history, weak-topic detection
validation.py              Input validation (uploads, User ID, questions, images)

experiment1_harness.py     Experiment 1: embedding model x chunking strategy comparison
experiment2_harness.py     Experiment 2: LLM generation model comparison (latency/CPU/RAM)
experiment1_questions.csv  The 31-question set both experiments use

feature_validation_checklist.md   Experiment 3: system validation checklist & real OCR results

tests/                     All automated tests (pytest) and the two manual verification scripts
  conftest.py                  shared fixtures (redirects DB/ChromaDB to temp locations)
  test_*.py                    the automated suite - run with `pytest` from the project root
                                (19 of these are marked @pytest.mark.checklist -- see below)
  score_ocr.py                  WER/CER scorer - run manually: `python tests/score_ocr.py samples/`
  e2e_gradio_smoke.py           real-browser-style smoke test - run manually: `python tests/e2e_gradio_smoke.py`

assets/                    the 5 lecture documents (Experiment 1/2 corpus)
samples/                   OCR test samples (Experiment 3): one real document + hand-typed
                           ground truth per category (digital_pdf, screenshot, printed_scan,
                           handwritten)
results/                   Experiment 1/2 output (created when you run the harnesses)
```

`tests/score_ocr.py` and `tests/e2e_gradio_smoke.py` are not `pytest`-collected tests - they're
run directly (`python tests/score_ocr.py ...`) - but they live in `tests/` because both were
written expecting the project root to be *one level above their own location*.

## Setup

```
python -m venv venv
venv\Scripts\activate          (Windows)  or  source venv/bin/activate  (macOS/Linux)
pip install -r requirements.txt
```
`requirements.txt` is in the project root and pins every dependency to the exact version this
project has been built and tested against - Gradio, ChromaDB, sentence-transformers, PyPDF2,
python-docx, python-pptx, PaddleOCR (+ PaddlePaddle/PaddleX), ollama-python, psutil, pytest, and
everything each of those needs underneath. One `pip install` gets a working environment; there's
no separate step for any individual package.

You also need [Ollama](https://ollama.com) running locally with at least `llama3.2` and
`llava:7b` pulled (`ollama pull llama3.2`, `ollama pull llava:7b`), plus whichever models you
want to compare in Experiment 2 (phi3:mini & gemma3:4b chosen for this project).

**PaddleOCR version note:** the current code uses PaddleOCR 3.x's `.predict()` API:
```python
PaddleOCR(lang="en", use_textline_orientation=True, enable_mkldnn=False)
```
confirmed working on Windows with the version pinned in `requirements.txt` (`paddleocr==3.7.0`).
If OCR breaks again after a `pip install --upgrade paddleocr`, check `pip show paddleocr`
against PaddleOCR's own upgrade notes before changing `material_processor.py` - don't guess
at the fix blind.

## Running the app

```
python app.py
```
Serves at `127.0.0.1:7860`.

## Running the tests

```
pytest tests/
```
517 tests as of this update - functional correctness, multi-user isolation, personalisation,
the clarification flow, image questions, weak-topic detection, input validation, both
experiment harnesses, and OCR extraction. All pass in a stubbed sandbox environment (no real
Ollama/PaddleOCR needed to run them) and have also been confirmed against the real app.

For just the rows `feature_validation_checklist.md` tracks:
```
pytest -m checklist -v
```
These are the real tests from `test_functional.py`, `test_multiuser.py`,
`test_personalization.py`, and `test_features.py`, marked
`@pytest.mark.checklist` - running the marker selects them in place, so there
is no separate file and no test is ever collected (or run) twice.

## Running the experiments

**Experiment 1** (embedding model x chunking strategy):
```
python experiment1_harness.py --docs assets --questions experiment1_questions.csv --out results/exp1 --dry-run
python experiment1_harness.py --docs assets --questions experiment1_questions.csv --out results/exp1
```

**Experiment 2** (LLM generation model comparison - retrieval held fixed):
```
python experiment2_harness.py --docs assets --models llama3.2 phi3:mini gemma3:latest --process-name llama-server.exe --out results/exp2
```
Use the exact tags `ollama list` shows on your machine, not the names above.

**Experiment 3** (system validation): `pytest -m checklist -v` for the automated rows, plus
`python tests/score_ocr.py samples/` for the 4 OCR rows.
Results already recorded in `feature_validation_checklist.md`.

## Current status

- **Experiment 1:** run for real, all 9 cells. Retrieval accuracy doesn't statistically separate
  the 9 configurations at this sample size; sliding_window chunking shows a consistent edge
  across all three embedding models; Nomic costs far more to run with no accuracy benefit.
- **Experiment 2:** ran with zero errors on the real machine, compares the models on speed 
  and reliability only, skipping human judgement due to time limitation. 
- **Experiment 3:** all 18 checklist rows pass for real; all four OCR categories have a real
  result (handwritten scores worst, as expected; a counterintuitive finding is written up in
  `feature_validation_checklist.md` about why the non-OCR PDF path scores worse than two of the
  three OCR paths).

## Known limitations

Understanding ratings conflate answer quality with topic difficulty; the rating gate is
per-browser-tab state, not a server-side lock; document-name matching in questions is lexical,
not semantic; OCR results above are each a single sample (n=1) and illustrate pipeline behaviour
rather than a statistically powered claim.