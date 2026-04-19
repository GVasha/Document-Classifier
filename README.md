# Document Classifier

Classifies scanned document images or PDFs into one of four categories: **email**, **invoice**, **questionnaire**, **scientific_pub**. If the document is classified as an invoice, it also extracts structured fields from it.

## How it works

**Classification** — each document goes through two parallel pipelines whose probability outputs are combined in a weighted ensemble:

1. **TF-IDF + Logistic Regression (weight 0.7)** — OCR text is vectorized using word and character n-grams, then classified by a tuned Logistic Regression. Achieves ~96.25% accuracy on the test set.
2. **LightGBM on hand-crafted features (weight 0.3)** — ~24 regex-based features capture structural signals (word count, number density), class-specific keywords (invoice totals, email headers, question marks, academic markers), and pattern counts (dates, currency symbols, email addresses). Achieves ~90% accuracy standalone.

The combined prediction is the argmax of the weighted probability sum.

**Invoice extraction** — if the document is classified as an invoice, a rule-based extractor runs on the OCR text and returns six structured fields: invoice number, invoice date, due date, issuer name, recipient name, and total amount.

## Repository layout

```text
src/
    classifier.py        # main entry point: classify() and predict_category()
    invoice_extractor.py # rule-based invoice field extraction
    shared.py            # preprocess(), extract_features(), load_ocr_cache()
notebooks/
    classification_tests.ipynb        # TF-IDF + SVM training, evaluation, model selection
    classification_lgbm.ipynb         # LightGBM training on hand-crafted features
    invoice_information_extraction.ipynb  # invoice extractor development and evaluation
requirements.txt
artifacts/                 # generated at training time (gitignored)
    models/
        best_classifier.pkl    # TF-IDF + Tuned Linear SVM
        lgbm_classifier.pkl    # LightGBM
    ocr_cache.csv              # cached OCR output (~30 min to generate)
data/                      # training images (gitignored)
    email/
    invoice/
    questionnaire/
    scientific_pub/
```

## Data and pre-built artifacts

The dataset contains **500 scanned document images per class** (2,000 total), sourced from RVL-CDIP.
Training uses the first 300 per class; the remaining 200 are available for unseen-document validation.

Pre-built artifacts (OCR cache + trained models) are available for download so you can run inference or retrain without re-running OCR from scratch:

[Download data and artifacts from Google Drive](https://drive.google.com/drive/folders/1c3HbPby2vNByxhA5BlHwk-5DU1mTXjD4?usp=sharing)

Place the downloaded folders at the repo root so the layout matches the tree above.

## Setup

### 1. Install Tesseract

Download and run the installer from <https://github.com/UB-Mannheim/tesseract/wiki>

After installing, add it to your system PATH (PowerShell as admin):

```powershell
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\Tesseract-OCR", "Machine")
```

Restart your terminal and verify: `tesseract --version`

### 2. Create and activate a virtual environment

```powershell
python -m venv statsvenv
.\statsvenv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```powershell
pip install -r requirements.txt
```

## Training

Skip this section if you downloaded the pre-built artifacts above.

Run each notebook **top to bottom** in this order:

1. `notebooks/classification_tests.ipynb` — trains TF-IDF + SVM models, saves `artifacts/models/best_classifier.pkl`
2. `notebooks/classification_lgbm.ipynb` — trains LightGBM, saves `artifacts/models/lgbm_classifier.pkl`

The first run OCRs the full dataset and caches results to `artifacts/ocr_cache.csv` (~30 min). All subsequent runs load from cache instantly.

## Inference

```python
import sys
sys.path.insert(0, "src")
from classifier import classify

result = classify("path/to/document.pdf")
print(result["label"])  # e.g. "invoice"

# for invoices, extracted fields are included:
# result["invoice_fields"] -> {"invoice_number": ..., "invoice_date": ..., ...}
```

Or from the command line (output is JSON):

```powershell
python src/classifier.py path/to/document.pdf
```

Supported input formats: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`
