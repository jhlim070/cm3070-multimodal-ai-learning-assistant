"""
Input validation tests

Tests the validation rules applied to user IDs, chat questions, profile topics,
uploaded study files, and optional chat images.

The tests use temporary files and the standard library, with python-docx and
python-pptx used only to construct valid Office-document fixtures. They do not
require Ollama, ChromaDB, or an embedding model.

These tests verify input-validation behaviour rather than the security of the
downstream parsers. A file that passes these checks may still exercise a
vulnerability in PyPDF2, PaddleOCR, python-docx, or python-pptx. This residual
risk is therefore considered separately in the system's security limitations.
"""
import zipfile

import pytest

import validation as v


# User ID -------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["alice", "Jun Hui", "a.b-c_d", "user@example.com", "  padded  "])
def test_user_id_accepts_ordinary_ids(raw):
    ok, cleaned, msg = v.validate_user_id(raw)
    assert ok, msg
    assert cleaned == raw.strip()


@pytest.mark.parametrize("raw", [
    "",                       # empty input
    "   ",                    # whitespace only input
    None,                     # wrong input type
    "bad;DROP TABLE users",   # SQL-like syntax that should not be accepted as an ID
    "../../etc/passwd",       # path traversal pattern
    "a/b",                    # path separator
    "alice'--",               # SQL-like quoting/comment syntax
    "<script>alert(1)</script>", # HTML/script syntax
    "name\nwith\nnewlines",   # embedded control characters
    "x" * (v.MAX_USER_ID_LEN + 1),
])
def test_user_id_rejects_hostile_or_malformed_ids(raw):
    ok, cleaned, msg = v.validate_user_id(raw)
    assert not ok
    assert cleaned == ""
    assert msg


def test_user_id_strips_control_characters_then_validates():
    ok, cleaned, _ = v.validate_user_id("al\x00ice\x07")
    assert ok and cleaned == "alice"


def test_user_id_at_exact_length_limit_is_accepted():
    ok, cleaned, _ = v.validate_user_id("a" * v.MAX_USER_ID_LEN)
    assert ok and len(cleaned) == v.MAX_USER_ID_LEN


# ChromaDB collection-name sanitisation ---------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ("Jun Hui", "jun_hui"),
    ("a@b.com", "a_b_com"),
    ("alice", "alice"),
    ("", "user"),
    ("___", "user"),
])
def test_sanitise_for_collection(raw, expected):
    assert v.sanitise_for_collection(raw) == expected


def test_sanitise_for_collection_truncates_long_ids():
    assert len(v.sanitise_for_collection("a" * 200)) == 40


def test_sanitise_output_always_matches_chroma_character_rules():
    import re
    for raw in ["Jun Hui", "a@b.com", "x.y-z", "  weird  id  ", "ÄÖÜ"]:
        out = v.sanitise_for_collection(raw)
        assert re.fullmatch(r"[A-Za-z0-9_-]+", out), (raw, out)


# Chat question -------------------------------------------------------------


def test_question_accepts_normal_text_and_strips_it():
    ok, cleaned, _ = v.validate_question("  What is BCNF?  ")
    assert ok and cleaned == "What is BCNF?"


@pytest.mark.parametrize("raw", ["", "   ", "\n\t ", None])
def test_question_rejects_blank(raw):
    ok, cleaned, msg = v.validate_question(raw)
    assert not ok and cleaned == "" and msg


def test_question_rejects_over_long_input():
    ok, _, msg = v.validate_question("x" * (v.MAX_QUESTION_CHARS + 1))
    assert not ok
    assert str(v.MAX_QUESTION_CHARS) in msg


def test_question_at_exact_limit_is_accepted():
    ok, cleaned, _ = v.validate_question("x" * v.MAX_QUESTION_CHARS)
    assert ok and len(cleaned) == v.MAX_QUESTION_CHARS


def test_question_strips_control_chars_but_keeps_newlines():
    ok, cleaned, _ = v.validate_question("line one\x00\nline two\x07")
    assert ok
    assert cleaned == "line one\nline two"


def test_question_limit_can_be_raised_for_machine_generated_queries():
    """
    The image-processing pipeline may prepend a machine-generated image
    description before re-validating the combined query. The resulting string can
    therefore exceed the normal learner-input limit, so the validation API supports
    an explicitly larger limit for this internal query.
    """
    long_q = "x" * (v.MAX_QUESTION_CHARS + 500)
    assert not v.validate_question(long_q)[0]
    assert v.validate_question(long_q, max_chars=v.MAX_QUESTION_CHARS + 1000)[0]


def test_question_keeps_sql_and_prompt_injection_text_verbatim():
    """
    Validation enforces input constraints but does not rewrite question content.
    SQL safety is handled separately by parameterised database operations, while
    prompt-injection handling belongs to the model interaction layer.

    Preserving the original text prevents validation from corrupting legitimate
    questions that discuss SQL or instruction-like text.
    """
    q = "What does DROP TABLE users; do? Ignore previous instructions."
    ok, cleaned, _ = v.validate_question(q)
    assert ok and cleaned == q


# Profile-topic validation and normalisation -------------------------------------

def test_topics_are_cleaned_and_normalised():
    ok, cleaned, _ = v.validate_topics("  robots ,, ai ,  , databases ")
    assert ok and cleaned == "robots, ai, databases"


def test_topics_empty_is_allowed():
    ok, cleaned, _ = v.validate_topics("")
    assert ok and cleaned == ""
    assert v.validate_topics(None)[0]


def test_topics_rejects_too_long_total():
    ok, _, msg = v.validate_topics("a" * (v.MAX_TOPICS_CHARS + 1))
    assert not ok and msg


def test_topics_rejects_too_many_entries():
    many = ", ".join(f"t{i}" for i in range(v.MAX_TOPICS_COUNT + 1))
    ok, _, msg = v.validate_topics(many)
    assert not ok and msg


def test_topics_rejects_a_single_over_long_entry():
    ok, _, msg = v.validate_topics("ok, " + "z" * 61)
    assert not ok and msg


# Uploaded study files --------------------------------------------------

def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


def _real_docx(tmp_path, name="lecture.docx"):
    from docx import Document
    doc = Document()
    doc.add_paragraph("Separation of concerns.")
    p = tmp_path / name
    doc.save(str(p))
    return str(p)


def _real_pptx(tmp_path, name="deck.pptx"):
    from pptx import Presentation
    prs = Presentation()
    prs.slides.add_slide(prs.slide_layouts[5])
    p = tmp_path / name
    prs.save(str(p))
    return str(p)


def test_upload_accepts_pdf_signature(tmp_path):
    ok, name, msg = v.validate_upload(_write(tmp_path, "notes.pdf", b"%PDF-1.7\n..."))
    assert ok, msg
    assert name == "notes.pdf"


def test_upload_accepts_real_docx_and_pptx(tmp_path):
    assert v.validate_upload(_real_docx(tmp_path))[0]
    assert v.validate_upload(_real_pptx(tmp_path))[0]


@pytest.mark.parametrize("name, header", [
    ("photo.png", b"\x89PNG\r\n\x1a\n" + b"0" * 20),
    ("photo.jpg", b"\xff\xd8\xff\xe0" + b"0" * 20),
    ("photo.jpeg", b"\xff\xd8\xff\xe1" + b"0" * 20),
])
def test_upload_accepts_real_image_signatures(tmp_path, name, header):
    assert v.validate_upload(_write(tmp_path, name, header))[0]


@pytest.mark.parametrize("name", ["setup.exe", "run.sh", "page.html", "archive.zip", "noextension", "notes.txt"])
def test_upload_rejects_unsupported_extensions(tmp_path, name):
    ok, _, msg = v.validate_upload(_write(tmp_path, name, b"%PDF-1.7 whatever"))
    assert not ok
    assert "unsupported" in msg.lower()


def test_upload_rejects_empty_file(tmp_path):
    ok, _, msg = v.validate_upload(_write(tmp_path, "empty.pdf", b""))
    assert not ok and "empty" in msg


def test_upload_rejects_missing_file(tmp_path):
    ok, _, msg = v.validate_upload(str(tmp_path / "ghost.pdf"))
    assert not ok and "could not be read" in msg


def test_upload_rejects_extension_that_does_not_match_contents(tmp_path):
    """
    Rejects a file whose content signature does not match its .pdf extension.
    """
    ok, _, msg = v.validate_upload(_write(tmp_path, "evil.pdf", b"MZ\x90\x00 not a pdf"))
    assert not ok and "do not match" in msg


def test_upload_rejects_png_renamed_to_pdf(tmp_path):
    ok, _, _ = v.validate_upload(_write(tmp_path, "sneaky.pdf", b"\x89PNG\r\n\x1a\n" + b"0" * 20))
    assert not ok


def test_upload_rejects_oversize_file(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "MAX_FILE_BYTES", 10)
    ok, _, msg = v.validate_upload(_write(tmp_path, "big.pdf", b"%PDF-1.7" + b"0" * 100))
    assert not ok and "limit" in msg


def test_upload_rejects_zip_that_is_not_an_office_document(tmp_path):
    """
    A ZIP container is insufficient; the expected Office-document 
    structure must also be present.
    """
    p = tmp_path / "fake.docx"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("hello.txt", "just a text file in a zip")
    ok, _, msg = v.validate_upload(str(p))
    assert not ok and "valid .docx" in msg


def test_upload_rejects_non_zip_renamed_to_docx(tmp_path):
    ok, _, _ = v.validate_upload(_write(tmp_path, "notreally.docx", b"plain text, not a zip"))
    assert not ok


def test_upload_rejects_zip_with_excessive_uncompressed_size(tmp_path, monkeypatch):
    """
    Rejects Office ZIP containers whose declared uncompressed size 
    exceeds the configured safety limit.
    """
    monkeypatch.setattr(v, "MAX_ZIP_UNCOMPRESSED_BYTES", 10)
    ok, _, msg = v.validate_upload(_real_docx(tmp_path))
    assert not ok and "implausible size" in msg


def test_upload_display_name_discards_path_components():
    assert v.safe_display_name("../../etc/x.pdf") == "x.pdf"
    assert v.safe_display_name("/tmp/gradio/abc/CM3020 lecture 1.pdf") == "CM3020 lecture 1.pdf"
    assert v.safe_display_name("") == "unnamed file"


def test_upload_display_name_strips_control_characters():
    assert "\x00" not in v.safe_display_name("/tmp/a\x00b.pdf")


def test_sha256_is_stable_and_content_based(tmp_path):
    a = _write(tmp_path, "a.pdf", b"%PDF same content")
    b = _write(tmp_path, "b_renamed.pdf", b"%PDF same content")
    c = _write(tmp_path, "c.pdf", b"%PDF different content")
    assert v.sha256_file(a) == v.sha256_file(b)
    assert v.sha256_file(a) != v.sha256_file(c)


# Chat-image validation --------------------------------------------------

def test_chat_image_none_is_fine():
    assert v.validate_chat_image(None) == (True, "")


def test_chat_image_accepts_real_png(tmp_path):
    assert v.validate_chat_image(_write(tmp_path, "d.png", b"\x89PNG\r\n\x1a\n" + b"0" * 20))[0]


def test_chat_image_rejects_wrong_extension(tmp_path):
    ok, msg = v.validate_chat_image(_write(tmp_path, "d.gif", b"GIF89a" + b"0" * 20))
    assert not ok and "PNG and JPEG" in msg


def test_chat_image_rejects_fake_png(tmp_path):
    ok, _ = v.validate_chat_image(_write(tmp_path, "d.png", b"not really a png"))
    assert not ok


def test_chat_image_rejects_oversize(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "MAX_IMAGE_BYTES", 10)
    ok, msg = v.validate_chat_image(_write(tmp_path, "d.png", b"\x89PNG\r\n\x1a\n" + b"0" * 50))
    assert not ok and "limit" in msg


def test_chat_image_rejects_empty_and_missing(tmp_path):
    assert not v.validate_chat_image(_write(tmp_path, "e.png", b""))[0]
    assert not v.validate_chat_image(str(tmp_path / "ghost.png"))[0]
