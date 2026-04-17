"""Preprocessing: raw document file -> clean text string.

Accepts a PDF or image file, extracts text (OCR fallback for scanned pages),
and returns a cleaned plain string.
"""

import re
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MIN_CHARS_PER_PAGE = 50  # below this, treat page as scanned and OCR it


def preprocess(path: str | Path) -> str:
    """Extract and clean text from a PDF or image file."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        text = _extract_pdf(path)
    elif ext in IMAGE_EXTS:
        text = pytesseract.image_to_string(Image.open(path))
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return _clean(text)


def _extract_pdf(path: Path) -> str:
    """Extract text from each page; OCR pages that look scanned."""
    parts = []
    with fitz.open(path) as doc:
        for page in doc:
            text = page.get_text()
            if len(text.strip()) < MIN_CHARS_PER_PAGE:
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                text = pytesseract.image_to_string(img)
            parts.append(text)
    return "\n".join(parts)


def _clean(text: str) -> str:
    """Normalize unicode, collapse whitespace, drop empty-line runs."""
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]

    cleaned, blank = [], False
    for line in lines:
        if line:
            cleaned.append(line)
            blank = False
        elif not blank:
            cleaned.append("")
            blank = True
    return "\n".join(cleaned).strip()


if __name__ == "__main__":
    import sys
    print(preprocess(sys.argv[1]))
