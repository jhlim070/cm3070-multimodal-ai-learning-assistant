import os
from PyPDF2 import PdfReader
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')   # download models first

# Reads each page then joins extracted text
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Get a list of text boxes and joins the detected strings using OCR
def extract_text_from_image(img_path):
    result = ocr.ocr(img_path, cls=True)
    text = ""
    for line in result[0]:
        text += line[1][0] + " "
    return text.strip()

# checks the file extension and calls the appropriate function
def process_file(file_path):
    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return extract_text_from_image(file_path)
    else:
        raise ValueError("Unsupported file type")