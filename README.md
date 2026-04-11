# Document Classifier

Extracts and cleans text from PDF and image files, with OCR fallback for scanned pages.

## Requirements

- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (system binary, separate from the Python package)

## Setup

### 1. Install Tesseract

Download and run the installer from:  
https://github.com/UB-Mannheim/tesseract/wiki

Default install path: `C:\Program Files\Tesseract-OCR`

After installing, add Tesseract to your system PATH (run PowerShell as admin):

```powershell
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\Tesseract-OCR", "Machine")
```

Then restart your terminal (or fully reopen VS Code) and verify:

```powershell
tesseract --version
```

### 2. Create and activate a virtual environment

```powershell
python -m venv statsvenv
.\statsvenv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```powershell
pip install -r requirements.txt
```

## Usage

```powershell
python scan.py <path-to-file>
```

Supported input formats: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`
