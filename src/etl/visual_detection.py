from io import BytesIO

from pypdf import PdfReader

import fitz
import pytesseract
from PIL import Image



def detect_pdf_visual_status(pdf_bytes: bytes) -> str:
    """Detect whether a PDF likely needs OCR or visual extraction."""
    reader = PdfReader(BytesIO(pdf_bytes))

    page_count = len(reader.pages)
    text_character_count = 0
    image_count = 0

    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_character_count += len(page_text.strip())

        resources = page.get("/Resources") or {}
        xobjects = resources.get("/XObject")

        if not xobjects:
            continue

        for xobject in xobjects.get_object().values():
            resolved_object = xobject.get_object()

            if resolved_object.get("/Subtype") == "/Image":
                image_count += 1

    if page_count == 0:
        return "PDF unreadable"

    if text_character_count == 0 and image_count > 0:
        return "OCR needed - scanned/image PDF"

    if image_count > 0:
        return "PDF text extracted + visual content detected"

    if text_character_count < page_count * 80:
        return "OCR review recommended - low text PDF"

    return "PDF text extracted"


def extract_pdf_ocr_text(pdf_bytes: bytes, max_pages: int = 5) -> str:
    """Extract OCR text from rendered PDF pages using local Tesseract."""
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    ocr_text_parts = []

    pages_to_process = min(len(pdf_document), max_pages)

    for page_index in range(pages_to_process):
        page = pdf_document[page_index]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)

        image = Image.frombytes(
            "RGB",
            [pixmap.width, pixmap.height],
            pixmap.samples,
        )

        page_text = pytesseract.image_to_string(image).strip()

        if page_text:
            ocr_text_parts.append(f"[OCR page {page_index + 1}]\n{page_text}")

    return "\n\n".join(ocr_text_parts)


def try_extract_pdf_ocr_text(pdf_bytes: bytes) -> tuple[str, str]:
    """Try OCR extraction and return extracted text plus status label."""
    try:
        ocr_text = extract_pdf_ocr_text(pdf_bytes)

        if ocr_text.strip():
            return ocr_text, "OCR text extracted from scanned/image PDF"

        return "", "OCR attempted - no text detected"
    except Exception as error:
        return "", f"OCR needed - extraction unavailable ({error})"
    