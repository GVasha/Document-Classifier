import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Paths
DATASET_DIR  = Path("data")
ARTIFACTS_DIR = Path("artifacts")
OCR_CACHE    = ARTIFACTS_DIR / "ocr_cache.csv"
MODELS_DIR   = ARTIFACTS_DIR / "models"

CLASSES = ["email", "invoice", "questionnaire", "scientific_pub"]


def extract_text(image_path) -> str:
    try:
        img  = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang="eng")
        text = text.replace("\n", " ")
        return re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        print(f"Error reading {Path(image_path).name}: {e}")
        return ""


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
