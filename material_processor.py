"""
Extract text from supported study-material formats for the personalised 
multimodal AI learning assistant

Supports PDF, image, DOCX, and PPTX files. PDF pages with a native text
layer are extracted directly, pages without extractable text are rendered
to images and processed with PaddleOCR. DOCX files include paragraphs and
tables, while PPTX files include shape text, tables, and speaker notes.
"""

import os
import tempfile
from PyPDF2 import PdfReader
from paddleocr import PaddleOCR
import fitz
from docx import Document as DocxDocument
from pptx import Presentation

# Configure PaddleOCR once and reuse the pipeline for all image-based
# extraction. Text-line orientation detection is enabled for rotated text.
ocr = PaddleOCR(lang="en", use_textline_orientation=True, enable_mkldnn=False)  


def extract_text_from_pdf(pdf_path):
    """ 
    Extract text directly from pages with a native text layer. Pages without
    extractable text are rendered to images and processed with OCR.
    """
    reader = PdfReader(pdf_path)
    text = ""
    doc = None  # only opened via fitz if a page actually needs OCR
    try:
        for i, page in enumerate(reader.pages):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                # Use a double newline so the downstream RAG chunker 
                # can recognise page-level or paragraph boundaries
                text += page_text + "\n\n"
                continue
            # Open with PyMuPDF lazily only if a page requires OCR
            if doc is None:
                doc = fitz.open(pdf_path)
            # Render the page at sufficient resolution for OCR while avoiding
            # unnecessarily large intermediate images
            pixmap = doc[i].get_pixmap(dpi=200)
            # PaddleOCR receives a file path, so save the rendered page temporarily
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                pixmap.save(tmp_path)
                text += extract_text_from_image(tmp_path) + "\n\n"
            finally:
                # Always remove the temporary file after OCR completes
                os.remove(tmp_path)
    finally:
        if doc is not None:
            doc.close()
    return text

# PaddleOCR 3.x returns result objects containing recognised text in `rec_texts`
# This helper isolates result-format handling from the main OCR extraction logic.
def _extract_rec_texts(res):
    """
    Return the recognised text lines from a PaddleOCR result item.
    
    Raise an error for an unknown representation rather than silently
    discarding recognised text.
    """
    try:
        texts = res["rec_texts"]
        if texts is not None:
            return list(texts)
    except (TypeError, KeyError, IndexError):
        pass
    texts = getattr(res, "rec_texts", None)          # attribute-style
    if texts is not None:
        return list(texts)
    payload = getattr(res, "json", None)              # nested under a `.json` property
    if isinstance(payload, dict):
        inner = payload.get("res", payload)
        if isinstance(inner, dict) and inner.get("rec_texts") is not None:
            return list(inner["rec_texts"])
    raise ValueError(
        f"Could not find recognised text in a PaddleOCR result of type {type(res)!r} "
        f"(tried res['rec_texts'], res.rec_texts, res.json['res']['rec_texts']): {res!r:.500}. "
        "This usually means the installed PaddleOCR version's result shape has changed again; "
        "check `pip show paddleocr` and PaddleOCR's own upgrade notes.")


def extract_text_from_image(img_path):
    result = ocr.predict(img_path)
    # No result items were returned, so there is no OCR text to extract.
    if not result:
        return ""
    lines = []
    for item in result:
        if item is None:
            continue
        else:
# --------------------------------------------------------------------------------------------
            # NEW (3.x) shape: item is a result object: pull every line it found.
            lines.extend(_extract_rec_texts(item))
    return " ".join(t.strip() for t in lines if t and t.strip())


""" 
DOCX files store text structurally, so OCR is unnecessary. Paragraphs
and tables are processed separately because table content is not included
in `Document.paragraphs`
"""
def extract_text_from_docx(docx_path):
    doc = DocxDocument(docx_path)
    text = ""
    for para in doc.paragraphs:
        if para.text.strip():
            # Preserve paragraph boundaries expected by downstream RAG chunker
            text += para.text.strip() + "\n\n"
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                text += row_text + "\n\n"
    return text


# PPTX text can occur in shape text frames, tables, and speaker notes.
# These sources are processed separately to avoid omitting slide content.
def extract_text_from_pptx(pptx_path):
    prs = Presentation(pptx_path)
    text = ""
    for slide in prs.slides:
        # Extract text in the order provided by python-pptx. This may not 
        # exactly match the visual reading order of complex slide layouts.
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                text += shape.text_frame.text.strip() + "\n\n"
            if shape.has_table:
                for row in shape.table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        text += row_text + "\n\n"
        # Include speaker notes because they may contain explanatory 
        # content that is not visible in the slide's main text
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                text += notes + "\n\n"
    return text

# Dispatch the file to the extractor corresponding to its extension
def process_file(file_path):
    """
    Extract text from a supported study-material file
    """
    lower_path = file_path.lower()
    if lower_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif lower_path.endswith(('.png', '.jpg', '.jpeg')):
        return extract_text_from_image(file_path)
    elif lower_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif lower_path.endswith('.pptx'):
        return extract_text_from_pptx(file_path)
    else:
        raise ValueError("Unsupported file type")