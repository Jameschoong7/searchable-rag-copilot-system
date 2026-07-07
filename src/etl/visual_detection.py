from io import BytesIO

from pypdf import PdfReader


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