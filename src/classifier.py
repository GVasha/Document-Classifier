import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "artifacts" / "models"
sys.path.insert(0, str(BASE_DIR))

from shared import preprocess, extract_features
from invoice_extractor import extract_fields_from_file

BASE_WEIGHT = 0.7
LGBM_WEIGHT   = 0.3

with open(MODELS_DIR / "best_classifier.pkl", "rb") as f:
    _logreg = pickle.load(f)

with open(MODELS_DIR / "lgbm_classifier.pkl", "rb") as f:
    _lgbm = pickle.load(f)

# Canonical class order from LogReg; used to align LightGBM probability columns.
_classes = list(_logreg.classes_)
_lgbm_col_order = [list(_lgbm.classes_).index(c) for c in _classes]


def _get_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[0]
    scores = model.decision_function(X)[0]
    scores = scores - scores.max()
    exp = np.exp(scores)
    return exp / exp.sum()


def predict_category(doc_path: str) -> str:
    text = preprocess(doc_path)

    p_base = _get_proba(_logreg, [text])

    feats  = pd.DataFrame([extract_features(text)])
    p_lgbm = _lgbm.predict_proba(feats)[0][_lgbm_col_order]

    p_combined = BASE_WEIGHT * p_base + LGBM_WEIGHT * p_lgbm
    return _classes[int(np.argmax(p_combined))]


def classify(doc_path: str) -> dict:
    label = predict_category(doc_path)
    result = {"label": label}
    if label == "invoice":
        result["invoice_fields"] = extract_fields_from_file(doc_path)
    return result


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("Usage: python classifier.py <document_path>")
    else:
        print(json.dumps(classify(sys.argv[1]), indent=2))
