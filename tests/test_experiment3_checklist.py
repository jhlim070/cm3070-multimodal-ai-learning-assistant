"""
Experiment 3 - compressed checklist run.

Runs only the 18 automated tests named in feature_validation_checklist.md (rows 1-18),
producing a short, easy-to-transcribe report without scanning the full test suite.

HOW TO RUN:
pytest tests/test_experiment3_checklist.py -v

Tests are imported from their original modules rather than duplicated here. This keeps each
test's implementation in one place while allowing pytest to collect the imported tests as part
of this checklist run. Required module-local fixtures are imported alongside their tests.

NOT COVERED HERE:

Rows 19-22: OCR evaluation, scored by score_ocr.py against real sample documents.
Row 23: manual end-to-end smoke test (e2e_gradio_smoke.py) using the real Gradio app, Ollama,
and OCR.

ROW 11 NOTE:
The checklist's original test name, test_upload_and_index_handles_unsupported_file, was
renamed to test_upload_and_index_handles_parser_failure after extension validation moved
before parsing. The actual bad-file-type case is now covered by
test_unsupported_extension_never_reaches_a_parser; both tests are imported so Row 11 
remains covered while the checklist name is being updated.
"""

# Row 1-3: learner profile
from test_functional import (
    test_profile_create_and_retrieve,
    test_profile_overwrite_on_resave,
    test_profile_missing_user_returns_none,
)

# Row 4: interaction logging
from test_personalization import test_generate_answer_logs_the_interaction

# Rows 5-7: weak-topic detection
# These tests now follow the app's per-document weak-topic rule: a source document,
# rather than a keyword topic, is what gets flagged. The test names remain those
# referenced by the checklist; the fuller rule is covered in test_weak_topics_per_document.py.
from test_functional import (
    test_log_interaction_and_weak_topic_detected,
    test_weak_topics_excludes_high_feedback_topics,
    test_weak_topics_respects_threshold,
)

# Row 8-11: document upload and indexing
from test_functional import (
    test_upload_and_index_success,
    test_upload_and_index_no_file_selected,
    test_upload_and_index_no_extractable_text,
    test_upload_and_index_handles_parser_failure as test_upload_and_index_handles_unsupported_file,
)
# `stub_indexing` is defined as a module-local fixture in test_features.py.
# Re-export it here because the imported checklist test depends on it.
from test_features import stub_indexing, test_unsupported_extension_never_reaches_a_parser

# Row 12-13: chat and recommendation handlers
from test_functional import (
    test_respond_delegates_to_pipeline,
    test_get_recommendations_btn_delegates,
)

# Row 14-16: multi-user retrieval isolation
# `temp_chroma` is likewise a local fixture in test_multiuser.py -- same reasoning as above.
from test_multiuser import (
    temp_chroma,
    test_documents_are_isolated_per_user,
    test_new_user_collection_starts_empty,
    test_same_user_can_retrieve_own_multiple_documents,
)

# Row 17-18: personalisation
from test_personalization import (
    test_system_prompt_reflects_learning_style_and_difficulty,
    test_missing_profile_falls_back_to_defaults,
)

# These imports intentionally remain in the module namespace so pytest 
# collects the imported test functions as part of this checklist run.
__all__ = [
    "test_profile_create_and_retrieve", "test_profile_overwrite_on_resave",
    "test_profile_missing_user_returns_none", "test_generate_answer_logs_the_interaction",
    "test_log_interaction_and_weak_topic_detected", "test_weak_topics_excludes_high_feedback_topics",
    "test_weak_topics_respects_threshold", "test_upload_and_index_success",
    "test_upload_and_index_no_file_selected", "test_upload_and_index_no_extractable_text",
    "test_upload_and_index_handles_unsupported_file", "test_unsupported_extension_never_reaches_a_parser",
    "test_respond_delegates_to_pipeline", "test_get_recommendations_btn_delegates",
    "test_documents_are_isolated_per_user", "test_new_user_collection_starts_empty",
    "test_same_user_can_retrieve_own_multiple_documents",
    "test_system_prompt_reflects_learning_style_and_difficulty", "test_missing_profile_falls_back_to_defaults",
]
