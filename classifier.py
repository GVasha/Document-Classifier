from pathlib import Path
import pickle

from scan import preprocess

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_classifier.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def predict_category(doc_path: str) -> str:
    """
    Predict the class of a document by:
    1. extracting text with scan.preprocess()
    2. applying the trained classifier
    """
    text = preprocess(doc_path)
    prediction = model.predict([text])[0]
    return prediction


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python classifier.py <document_path>")
    else:
        doc_path = sys.argv[1]
        print(predict_category(doc_path))