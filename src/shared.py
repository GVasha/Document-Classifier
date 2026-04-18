import pickle
import re
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Paths — anchored to repo root regardless of working directory
_REPO_ROOT    = Path(__file__).resolve().parent.parent
DATASET_DIR   = _REPO_ROOT / "data"
ARTIFACTS_DIR = _REPO_ROOT / "artifacts"
OCR_CACHE     = ARTIFACTS_DIR / "ocr_cache.csv"
MODELS_DIR    = ARTIFACTS_DIR / "models"

CLASSES = ["email", "invoice", "questionnaire", "scientific_pub"]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MIN_CHARS_PER_PAGE = 50  # below this, treat PDF page as scanned and OCR it


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Normalize unicode, collapse inline whitespace, drop consecutive blank lines."""
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


def _extract_pdf(path: Path) -> str:
    """Extract text page by page; OCR any page whose text is too sparse."""
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


def preprocess(path: str | Path) -> str:
    """Extract and clean text from a PDF or image, preserving line structure."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        text = _extract_pdf(path)
    elif ext in IMAGE_EXTS:
        text = pytesseract.image_to_string(Image.open(path))
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return _clean(text)


def extract_text(image_path) -> str:
    """OCR a single image into a flat string (newlines collapsed).

    Used for building the OCR cache used in notebook training.
    """
    try:
        img  = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang="eng")
        text = text.replace("\n", " ")
        return re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        print(f"Error reading {Path(image_path).name}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def extract_features(text: str) -> dict:
    t     = text.lower()
    words = text.split()
    n     = max(len(words), 1)
    nums  = re.findall(r'\b\d+(?:[.,]\d+)?\b', text)

    return {
        # structural
        "word_count":           len(words),
        "char_count":           len(text),
        "avg_word_len":         float(np.mean([len(w) for w in words])) if words else 0.0,
        "number_count":         len(nums),
        "number_to_word_ratio": len(nums) / n,
        "unique_word_ratio":    len(set(words)) / n,
        # email
        "has_from":         int(bool(re.search(r'\bfrom\s*:', t))),
        "has_to":           int(bool(re.search(r'\bto\s*:', t))),
        "has_subject":      int(bool(re.search(r'\bsubject\s*:', t))),
        "email_addr_count": len(re.findall(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', t)),
        # invoice
        "has_invoice":    int(bool(re.search(r'\binvoice\b', t))),
        "has_total":      int(bool(re.search(r'\btotal\b', t))),
        "has_vat":        int(bool(re.search(r'\bvat\b|\btax\b', t))),
        "currency_count": len(re.findall(r'[$\u20ac\xa3]\s*\d|\b(?:usd|eur|gbp)\b', t)),
        "date_count":     len(re.findall(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', text)),
        "has_iban":       int(bool(re.search(r'\biban\b', t))),
        "has_due":        int(bool(re.search(r'\bdue\b', t))),
        # questionnaire
        "question_mark_count": text.count('?'),
        "question_count":      len(re.findall(r'\bquestion\b', t)),
        "response_count":      len(re.findall(r'\bresponse\b', t)),
        # scientific
        "has_abstract":   int(bool(re.search(r'\babstract\b', t))),
        "has_references": int(bool(re.search(r'\breferences\b', t))),
        "has_et_al":      int(bool(re.search(r'\bet al\b', t))),
        "figure_count":   len(re.findall(r'\bfig(?:ure)?\b', t)),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ocr_cache(max_per_class: int = 300) -> pd.DataFrame:
    if OCR_CACHE.exists():
        df = pd.read_csv(OCR_CACHE)
        df["text"] = df["text"].fillna("")
        print("Loaded from cache:", OCR_CACHE)
    else:
        rows = []
        for label in CLASSES:
            paths = sorted((DATASET_DIR / label).glob("*.png"))[:max_per_class]
            print(f"Processing {label} ({len(paths)} files)...")
            for p in paths:
                rows.append({"path": str(p), "label": label, "text": extract_text(p)})
        df = pd.DataFrame(rows)
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        df.to_csv(OCR_CACHE, index=False)
        print("OCR complete. Saved to:", OCR_CACHE)
    print(f"Dataset: {df.shape}")
    return df
