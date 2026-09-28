# Feature Validation Checklist (Experiment 3)

This is filled in as I am running the automated tests and any manual checks. 
This maps directly to the Feature Validation Checklist in my FPR section 5.5.1.

**Rows 1-18 (all automated):** run `pytest -m checklist -v` -- one command, one
line of output per row below, nothing to hunt for across the rest of the suite.
These 19 tests (18 rows + row 11's more literal match, see its own note below)
are marked `@pytest.mark.checklist` in their real home files -- no separate
file, no duplication; a fix to one of them is never duplicated anywhere else.

**Rows 19-22 (OCR):** `samples/` has real matched pairs for all four
categories (within samples). Row 19 (digital_pdf) has already been scored
for real, below -- see the finding. Rows 20-22 need PaddleOCR, which this
run `python score_ocr.py samples/` to test those

| # | Feature | Test | Automated? | Result | Notes |
|---|---------|------|:---:|:---:|-------|
| 1 | Learner profile creation | `test_profile_create_and_retrieve` | Yes | | |
| 2 | Learner profile update | `test_profile_overwrite_on_resave` | Yes | | |
| 3 | Missing profile handled gracefully | `test_profile_missing_user_returns_none` | Yes | | |
| 4 | Interaction logging | `test_generate_answer_logs_the_interaction` | Yes | | |
| 5 | Weak topic detection (low feedback) | `test_log_interaction_and_weak_topic_detected` | Yes | | |
| 6 | Weak topic exclusion (high feedback) | `test_weak_topics_excludes_high_feedback_topics` | Yes | | |
| 7 | Weak topic threshold respected | `test_weak_topics_respects_threshold` | Yes | | |
| 8 | Document upload + indexing (happy path) | `test_upload_and_index_success` | Yes | | |
| 9 | Upload with no file selected | `test_upload_and_index_no_file_selected` | Yes | | |
| 10 | Upload with no extractable text | `test_upload_and_index_no_extractable_text` | Yes | | |
| 11 | Upload error handling (bad file type) | `test_upload_and_index_handles_unsupported_file` | Yes | | |
| 12 | Chat handler delegates correctly | `test_respond_delegates_to_pipeline` | Yes | | |
| 13 | Recommendations handler delegates correctly | `test_get_recommendations_btn_delegates` | Yes | | |
| 14 | Multi-user retrieval isolation | `test_documents_are_isolated_per_user` | Yes | | |
| 15 | New user starts with empty collection | `test_new_user_collection_starts_empty` | Yes | | |
| 16 | Same user retrieves across own multiple docs | `test_same_user_can_retrieve_own_multiple_documents` | Yes | | |
| 17 | Personalisation: system prompt varies with profile | `test_system_prompt_reflects_learning_style_and_difficulty` | Yes | | |
| 18 | Personalisation: sensible defaults with no profile | `test_missing_profile_falls_back_to_defaults` | Yes | | |
| 19 | OCR - digital PDF | `score_ocr.py` (manual dataset) | Partial | **WER 30.95%, CER 5.93%** | CER low (extraction is character-correct); WER high because PyPDF2 drops the space between words whose text-run styling changes ("visual" + "interactive" in different colours -> "visualandinteractive"). Not a project bug -- a PyPDF2 limitation -- but it also degrades retrieval: a fused word can't match a query term via embedding search as well as a correctly spaced one would |
| 20 | OCR - lecture slide screenshot | `score_ocr.py` (manual dataset) | Partial | **WER 12.00%, CER 13.11%** |  PaddleOCR 3.x (`.predict()`, `use_textline_orientation=True`). n=1 -- illustrative, not a statistically powered comparison. |
| 21 | OCR - printed/scanned document | `score_ocr.py` (manual dataset) | Partial | **WER 4.05%, CER 5.86%** | Best WER of all four categories, incl. the non-OCR digital_pdf -- see the note below. NOTE: this sample is a photo of a lit screen, not a printed page on paper. n=1. |
| 22 | OCR - handwritten notes | `score_ocr.py` (manual dataset) | Partial | **WER 37.50%, CER 19.23%** | Real result. Worst of all four categories on BOTH metrics, as anticipated when this checklist was drafted -- to be discussed in the Evaluation chapter as the expected failure mode for handwriting. n=1. |
| 23 | End-to-end manual smoke test (real Ollama + real OCR) | Manual - run the actual Gradio app once end-to-end | No | | Automated tests mock Ollama/embeddings for speed; do this once for real to confirm the wiring holds outside of mocks |

**Result column key:** Pass / Fail / Partial - if fail, will be noted and checked whether it is a test bug or a real defect.

## OCR results summary, and a finding worth its own paragraph in the Evaluation chapter

All four categories now have one real result each (n=1 per category -- illustrative of the
pipeline behaviour, not a statistically powered claim; say so explicitly if you cite these).

| Category | WER | CER |
|---|---|---|
| handwritten | 37.50% | 19.23% |
| digital_pdf | 30.95% | 5.93% |
| screenshot | 12.00% | 13.11% |
| printed_scan | 4.05% | 5.86% |

**Counterintuitive finding: the non-OCR digital_pdf path has a WORSE word error rate than two of
the three real-OCR paths.** This is not because PyPDF2's text extraction is less accurate than
PaddleOCR's recognition, their character error rates are nearly identical (5.93% vs 5.86% for
printed_scan). It is because PyPDF2 drops the space between words whose text-run styling changes
colour (see row 19's note): a handful of missing spaces fuses many words into "wrong" tokens at
the WORD level while barely touching character accuracy. PaddleOCR, working from the rendered
image rather than the PDF's internal text runs, does not have this failure mode at all, it 
segments words by their visual gaps, which are unaffected by colour changes.