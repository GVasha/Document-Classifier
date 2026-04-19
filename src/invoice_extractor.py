"""Unified invoice field extraction for pipeline usage.

This module centralizes the extraction logic previously kept in the notebook.
It exposes two main functions:

- extract_fields(text, layout="auto")
- extract_fields_from_file(path, layout="auto")
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pytesseract
from PIL import Image, ImageOps

from shared import preprocess

SUPPORTED_EXTS = {".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
DATE_YEAR_MIN = 1960
DATE_YEAR_MAX = 2030

DATE_REGEX = re.compile(
    r"\b(?:\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}|\d{4}[\-/]\d{1,2}[\-/]\d{1,2}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+\d{1,2},?\s+\d{2,4})\b",
    flags=re.IGNORECASE,
)
_AMOUNT_BODY = r"(?:\d{1,3}(?:[.,\s]\d{3})+|\d+)(?:[.,]\d{2})?"
AMOUNT_REGEX = re.compile(
    rf"(?<!\d)(?<!\d )(?:(?:\$|EUR|GBP|USD|AED|INR)\s*)?(?:{_AMOUNT_BODY})(?:\s*(?:USD|EUR|GBP|AED|INR))?(?![\d.,])"
)
TOTAL_AMOUNT_CAPTURE = rf"(?:(?:\$|EUR|GBP|USD|AED|INR)\s*)?(?:{_AMOUNT_BODY})(?:\s*(?:USD|EUR|GBP|AED|INR))?"

_INLINE_DATE_ANY = re.compile(
    r"\b\d{1,2}[./-]\s*\d{1,2}[./-]\s*\d{2,4}\b"
    r"|\b\d{1,2}\s*[-/]\s*[A-Za-z]{3,9}\s*[-/]\s*\d{2,4}\b"
    r"|\b[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4}\b"
    r"|\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b"
    r"|\b\d{4}[./-]\s*\d{1,2}[./-]\s*\d{1,2}\b",
    flags=re.IGNORECASE,
)

_DUE_LINE_PATTERNS = [
    re.compile(r"(?i)(?:due|[o0]ue|dne|cue)\s*(?:date|dae|data|dste)?\s*[:.\-\s]+\s*(.+)$"),
    re.compile(r"(?i)\bdue\s*date\s*[:.\-\s]+\s*(.+)$"),
    re.compile(r"(?i)\bdate\s*due\s*[:.\-\s]+\s*(.+)$"),
    re.compile(r"(?i)\bpayment\s*due\s*[:.\-\s]+\s*(.+)$"),
    re.compile(r"(?i)\b(?:latest\s*)?payment\s*(?:by|before)\s*[:.\-\s]+\s*(.+)$"),
    re.compile(r"(?i)\bpay(?:ment)?\s*by\s*[:.\-\s]+\s*(.+)$"),
    re.compile(r"(?i)\bremit(?:tance)?\s*(?:by|before)?\s*[:.\-\s]+\s*(.+)$"),
    re.compile(r"(?i)\bbalance\s*due\s*[:.\-\s]+\s*(.+)$"),
    re.compile(r"(?i)\b(?:deadline|good\s*until|must\s*be\s*paid)\s*[:.\-\s]+\s*(.+)$"),
    re.compile(r"(?i)\b(?:total\s+)?payment\s+due\s+(?:in|within)\s+(\d{1,3})\s*days?\b"),
    re.compile(r"(?i)\bdue\s+(?:in|within)\s+(\d{1,3})\s*days?\b"),
    re.compile(r"(?i)\bpay(?:ment)?\s+(?:in|within)\s+(\d{1,3})\s*days?\b"),
    re.compile(r"(?i)\bterms?\s*[:.\-\s]*(?:pay(?:ment)?\s*)?(?:in|within)\s+(\d{1,3})\s*days?\b"),
    re.compile(r"(?i)\bdue\s*[:.\-\s]+\s*(.+)$"),
]
_DUEISH_LINE_HINT = re.compile(
    r"(?i)\b(due|payment\s*due|pay\s*by|remit(?:tance)?|balance\s*due|"
    r"latest\s*payment|deadline|good\s*until)\b",
)


def _clean_ocr_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _ocr_image_multi_pass(path: Path) -> str:
    from PIL import ImageFilter, ImageEnhance

    img = Image.open(path)
    gray = ImageOps.grayscale(img)

    w, h = gray.size
    if max(w, h) < 1800:
        scale = 2
        gray = gray.resize((w * scale, h * scale), Image.BICUBIC)

    fast_cfg = "--oem 3 --psm 6"
    alt_cfg = "--oem 3 --psm 11"
    auto_cfg = "--oem 3 --psm 3"
    outputs: list[str] = []

    try:
        txt = _clean_ocr_text(pytesseract.image_to_string(gray, config=fast_cfg))
        if txt:
            outputs.append(txt)
    except Exception:
        pass

    if not outputs or len(outputs[0]) < 120:
        try:
            enhanced = ImageOps.autocontrast(gray)
            sharpened = ImageEnhance.Sharpness(enhanced).enhance(2.0)
            txt = _clean_ocr_text(pytesseract.image_to_string(sharpened, config=alt_cfg))
            if txt:
                outputs.append(txt)
        except Exception:
            pass

    if not outputs or max(len(x) for x in outputs) < 140:
        try:
            denoised = gray.filter(ImageFilter.MedianFilter(size=3))
            txt = _clean_ocr_text(pytesseract.image_to_string(denoised, config=auto_cfg))
            if txt:
                outputs.append(txt)
        except Exception:
            pass

    if not outputs or max(len(x) for x in outputs) < 140:
        for thr in (120, 145, 175):
            try:
                bw = gray.point(lambda x, t=thr: 255 if x > t else 0)
                txt = _clean_ocr_text(pytesseract.image_to_string(bw, config=alt_cfg))
                if txt:
                    outputs.append(txt)
            except Exception:
                pass

    if not outputs or max(len(x) for x in outputs) < 140:
        try:
            high_contrast = ImageEnhance.Contrast(gray).enhance(2.5)
            bw_hc = high_contrast.point(lambda x: 255 if x > 140 else 0)
            txt = _clean_ocr_text(pytesseract.image_to_string(bw_hc, config=fast_cfg))
            if txt:
                outputs.append(txt)
        except Exception:
            pass

    try:
        base_txt = _clean_ocr_text(preprocess(path))
        if base_txt:
            outputs.append(base_txt)
    except Exception:
        pass

    if not outputs:
        raise RuntimeError(f"OCR failed for file: {path}")

    def _score(t: str) -> tuple[int, int]:
        alnum = sum(ch.isalnum() for ch in t)
        return (alnum, len(t))

    merged_lines: list[str] = []
    seen = set()
    for txt in sorted(outputs, key=_score, reverse=True):
        for ln in txt.splitlines():
            key = ln.lower().strip()
            if key and key not in seen:
                seen.add(key)
                merged_lines.append(ln)
    return "\n".join(merged_lines) if merged_lines else max(outputs, key=_score)


def extract_text_robust(path: str | Path) -> str:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type: {ext}")
    if ext in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}:
        return _ocr_image_multi_pass(path)
    return preprocess(path)


def _first_match(patterns: Iterable[str], text: str) -> Optional[str]:
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            value = m.group(1).strip(" :-\t\n")
            if value:
                return value
    return None


def _normalize_amount(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    raw = value.strip()
    raw = re.sub(r"[^0-9,.-]", "", raw)
    if not raw:
        return None

    if raw.count(",") > 0 and raw.count(".") > 0:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "")
            raw = raw.replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif raw.count(",") > 0 and raw.count(".") == 0:
        parts = raw.split(",")
        if len(parts[-1]) == 2:
            raw = "".join(parts[:-1]).replace("-", "") + "." + parts[-1]
        else:
            raw = "".join(parts)

    raw = raw.strip("-")
    try:
        return f"{float(raw):.2f}"
    except ValueError:
        return None


def _parse_date_candidate(value: str | None) -> str | None:
    if not value:
        return None

    val = re.sub(r"\s+", " ", value.strip()).replace(",", "")
    val = val.replace(".", "/")

    fmts = [
        "%d/%m/%Y",
        "%d/%m/%y",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d-%m-%y",
        "%m-%d-%Y",
        "%m-%d-%y",
        "%Y-%m-%d",
        "%d-%b-%Y",
        "%d-%b-%y",
        "%d/%b/%Y",
        "%d/%b/%y",
        "%b %d %Y",
        "%B %d %Y",
        "%b %d %y",
        "%B %d %y",
        "%b. %d %Y",
        "%B. %d %Y",
        "%b. %d %y",
        "%B. %d %y",
        "%d %b %Y",
        "%d %B %Y",
        "%d %b %y",
        "%d %B %y",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(val, fmt).date()
            if DATE_YEAR_MIN <= dt.year <= DATE_YEAR_MAX:
                return dt.isoformat()
        except ValueError:
            continue

    m = DATE_REGEX.search(val)
    if m:
        found = m.group(0).strip()
        if found.lower() == val.strip().lower():
            return None
        return _parse_date_candidate(found)
    return None


def _parse_any_date_in_chunk(chunk: str | None, *, prefer_last: bool = False) -> str | None:
    if not chunk:
        return None
    chunk = chunk.strip()
    if (d := _parse_date_candidate(chunk)):
        return d

    found: list[str] = []
    for m in _INLINE_DATE_ANY.finditer(chunk):
        if (d := _parse_date_candidate(m.group(0))):
            found.append(d)
    if not found:
        return None
    if prefer_last or re.search(r"(?i)\bdue\b|pay(?:ment)?\s*by|remit|balance\s*due", chunk):
        return found[-1]
    return found[0]


def _extract_days_term(value: str | None) -> int | None:
    if not value:
        return None
    v = value.strip()
    for pat in (
        r"(?i)\b(?:net|in|within|before)\s*(\d{1,3})\s*days?\b",
        r"(?i)\b(\d{1,3})\s*days?\b",
        r"^\s*(\d{1,3})\s*$",
    ):
        m = re.search(pat, v)
        if m:
            try:
                n = int(m.group(1))
                if 1 <= n <= 365:
                    return n
            except ValueError:
                continue
    return None


def _looks_like_contact_or_address(line: str) -> bool:
    low = line.lower().strip()
    if not low:
        return True
    low_stripped = re.sub(r"^(?:mr|mrs|ms|miss|dr|prof|sir|dame|rev)\.?\s+", "", low).strip()
    if "@" in low or re.search(r"\b(?:www\.|http|https|\.com\b|\.org\b|\.net\b)\b", low):
        return True
    if re.search(r"\b(?:email|e-mail|phone|tel|fax|mobile|contact|site|website)\b", low):
        return True
    if re.search(
        r"\b(?:street|avenue|ave\.?|road|rd\.?|boulevard|blvd\.?|lane|ln\.?|"
        r"drive|court|circle|cir\.?|highway|hwy\.?|terrace|ter\.?|"
        r"suite|ste\.?|apt\.?|apartment|unit|floor|fl\.?|po\s*box|zip|postal|city|state|country)\b",
        low_stripped,
    ):
        return True
    if re.search(r"\d\s*(?:st|dr|ct)\.?\b", low_stripped):
        return True
    if re.match(r"^\s*\d+[A-Za-z-]*\b", low):
        return True
    if re.search(r"\b\d{4,6}\b", low):
        return True
    if re.search(r"\b[A-Z]{2}\s+\d{4,6}\b", line):
        return True
    if re.search(r"\b\d{3,}\b", low) and re.search(r"\b(?:us|usa|uk|eu|ca|de|fr)\b", low):
        return True
    return False


def _looks_like_name_line(line: str) -> bool:
    low = line.lower().strip()
    if not low or _looks_like_contact_or_address(line):
        return False
    label_norm = re.sub(r"[^a-z]+", " ", low).strip()
    if label_norm in {
        "bill to",
        "billed to",
        "ship to",
        "invoice to",
        "from",
        "seller",
        "buyer",
        "customer",
        "client",
        "recipient",
        "issuer",
        "vendor",
        "supplier",
        "remit to",
    }:
        return False
    if re.search(r"\b(?:gst|vat|tax|invoice|total|subtotal|amount|due|date)\b", low):
        return False
    table_kw = re.search(r"\b(?:item|items|quantity|qty|price|description|unit\s*price|rate|sku)\b", low)
    if table_kw:
        word_count = len(re.findall(r"[A-Za-z&'.-]+", line))
        if re.search(r"\d", line) or ":" in line or word_count >= 3:
            return False
    if re.match(r"^\s*\d", line):
        return False
    if line.isupper() and len(line.split()) >= 2:
        return False
    words = re.findall(r"[A-Za-z&'.-]+", line)
    if not (1 <= len(words) <= 8):
        return False
    if sum(ch.isdigit() for ch in line) > 2:
        return False
    alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]
    return len(alpha_words) >= 2 or any(
        w.lower().strip(".,") in {"inc", "ltd", "llc", "corp", "company", "group", "gmbh"}
        for w in alpha_words
    )


def _likely_company_line(line: str) -> bool:
    low = line.lower().strip()
    if len(low) < 4:
        return False
    if _looks_like_contact_or_address(low):
        return False
    if re.search(
        r"invoice|date|total|amount|page|phone|fax|www|http|remittance|estimate|"
        r"duplicate|triplicate|duns|received|routing|protective order|gst|vat|tax",
        low,
    ):
        return False

    digits = sum(c.isdigit() for c in line)
    letters = sum(c.isalpha() for c in line)
    if digits > 6:
        return False
    if letters < 5:
        return False

    words = re.findall(r"[A-Za-z&'.-]+", line)
    if len(words) < 2:
        return False

    corp = re.search(
        r"\b(inc\.?|corp\.?|co\.?|llc|ltd|company|corporation|associates|group|industries|"
        r"services|solutions|international|holdings|trading|enterprises|s\.a\.?|gmbh)\b",
        low,
    )
    if corp:
        return True

    alpha_ratio = letters / max(len(line), 1)
    return alpha_ratio >= 0.5 and len(words) <= 8


def _is_valid_invoice_number(value: str | None) -> bool:
    if not value:
        return False

    token = value.strip().upper().replace(" ", "")
    if len(token) < 4 or len(token) > 28:
        return False

    banned_sub = (
        "DAYSNET",
        "30DAY",
        "SHEET",
        "ADDITION",
        "CHARGES",
        "MISCELLANEOUS",
        "PHOTOGRAPHER",
        "COMMISSION",
        "ESTIMATE",
        "INSERTION",
        "NSERTION",
        "BILLING",
        "PLACEMENT",
        "PRODUCTION",
        "RECEIVED",
        "ROUTING",
    )
    if any(s in token for s in banned_sub):
        return False

    banned_exact = {
        "INVOICE",
        "VOICE",
        "OICE",
        "REPORT",
        "REPGRT",
        "ORIGINAL",
        "DUPLICATE",
        "TRIPLICATE",
        "TRFLICATE",
        "BALANCE",
        "SALE",
        "EXPORT",
        "DATE",
        "NUMBER",
        "NUMBERS",
        "NONE",
        "NANO",
        "NANOO",
        "OATE",
        "MUNBER",
        "PHOTOGRAPHERARTIST",
        "NET",
        "TOTAL",
        "GROSS",
        "PINRT",
    }
    if token in banned_exact:
        return False

    if re.search(r"[A-Z]{10,}", token) and sum(ch.isdigit() for ch in token) < 2:
        return False

    dnorm = re.sub(r"[^0-9A-Z]", "", token).replace("O", "0").replace("I", "1").replace("L", "1")
    if re.fullmatch(r"\d{10}", dnorm):
        return False
    if re.fullmatch(r"\d{3}[-.]?\d{3}[-.]?\d{4}", dnorm):
        return False

    if re.fullmatch(r"\d+[\-/]\d+[\-/]\d+", token):
        return False
    if re.fullmatch(r"\d{5}(?:-\d{4})?", token):
        return False

    digit_count = sum(ch.isdigit() for ch in token)
    alpha_count = sum(ch.isalpha() for ch in token)
    if len(token) > 12 and alpha_count > 8 and digit_count < 4:
        return False
    if len(token) <= 5 and alpha_count >= 2 and digit_count <= 2:
        return False
    if alpha_count == 0 and digit_count < 5:
        return False
    if alpha_count > 0 and digit_count < 2:
        return False
    return True


def _extract_invoice_number(text: str, lines: list[str]) -> str | None:
    label_patterns = [
        r"(?:invoice|inv|factura)\s*(?:number|no\.?|#|num(?:ber)?|id)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{2,28})",
        r"(?:our|your|customer)\s+ref(?:erence)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{2,28})",
        r"(?:job|order|document)\s*(?:number|no\.?|#)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{2,28})",
        r"(?:stmt|statement)\s*(?:number|no\.?|#)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{2,28})",
        r"(?:no\.?|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{3,28})",
    ]
    for p in label_patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            cand = m.group(1).strip(" -:")
            if _is_valid_invoice_number(cand):
                return cand

    skip_ln = re.compile(r"duns|phone|tel\.|fax|zip|postal", re.I)
    for ln in lines[:45]:
        if skip_ln.search(ln):
            continue
        for tok in re.findall(r"\b[A-Z0-9][A-Z0-9\-/]{3,27}\b", ln, flags=re.I):
            if _is_valid_invoice_number(tok):
                return tok
    return None


_CORP_SUFFIXES = {"INC", "LLC", "LTD", "CORP", "CO", "PLC", "SA", "GMBH", "GROUP", "USA", "UK"}


def _clean_party_name(name: str | None) -> str | None:
    """Remove trailing OCR noise from a party name (parenthesized junk, stray uppercase fragments)."""
    if not name:
        return name
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    name = re.sub(r"\s*\([^)]*$", "", name).strip()
    m = re.search(r"\s+([A-Z]{1,4})$", name)
    if m and m.group(1) not in _CORP_SUFFIXES:
        name = name[:m.start()].strip()
    name = re.sub(r"[^A-Za-z0-9.]+$", "", name).strip()
    return name if len(name) >= 3 else None


def _extract_party(text: str, lines: list[str], role: str) -> str | None:
    _RECIPIENT_LABEL_KWS = (
        r"buyer|bill[\s_-]*to|invoice[\s_-]*to|customer|client|sold[\s_-]*to|ship[\s_-]*to|recipient"
    )
    _ISSUER_LABEL_KWS = (
        r"from|bill[\s_-]*from|vendor|supplier|seller|issuer|remit[\s_-]*to"
    )

    if role == "recipient":
        label_patterns = [
            rf"^\s*(?:{_RECIPIENT_LABEL_KWS})\b",
            rf"\b(?:{_RECIPIENT_LABEL_KWS})\b\s*:",
        ]
        inline_patterns = [
            rf"(?im)^\s*(?:{_RECIPIENT_LABEL_KWS})\s*[:\-]\s*([^\n]{{2,80}})$",
            rf"(?im)\b(?:{_RECIPIENT_LABEL_KWS})\b\s*[:\-]\s*([^\n]{{2,80}})",
        ]
    else:
        label_patterns = [
            rf"^\s*(?:{_ISSUER_LABEL_KWS})\b",
            rf"\b(?:{_ISSUER_LABEL_KWS})\b\s*:",
        ]
        inline_patterns = [
            rf"(?im)^\s*(?:{_ISSUER_LABEL_KWS})\s*[:\-]\s*([^\n]{{2,80}})$",
            rf"(?im)\b(?:{_ISSUER_LABEL_KWS})\b\s*[:\-]\s*([^\n]{{2,80}})",
        ]

    def _clean_inline_party_value(value: str) -> str:
        value = value.strip(" :-\t")
        value = re.sub(r"^[^A-Za-z]+", "", value)
        value = re.sub(r"^[a-z](?=[A-Z])", "", value)
        value = re.split(r"\s{2,}|\t+", value, maxsplit=1)[0].strip()
        value = re.split(
            r"\b(?:invoi\w*\s*#?|invoi\w*\s*no\.?|invoi\w*\s*id|invoi\w*\s*number|"
            r"date|due\s*date|tel|phone|email|site|www|http)\b",
            value,
            maxsplit=1,
            flags=re.I,
        )[0].strip(" ,;:-")
        return value

    def _base_party_checks(cand: str) -> bool:
        cand = cand.strip(" :-\t")
        if not cand or _looks_like_contact_or_address(cand):
            return False
        if re.search(r"\b(?:gst|vat|tax|subtotal|total|amount\s*due)\b", cand, re.I):
            return False
        if re.search(r"\b(?:item|items|quantity|qty|price|description|unit\s*price|rate|sku)\b", cand, re.I):
            word_count = len(re.findall(r"[A-Za-z&'.-]+", cand))
            if re.search(r"\d", cand) or ":" in cand or word_count >= 3:
                return False
        return True

    def _issuer_blocklist(cand: str) -> bool:
        low = cand.lower()
        if re.search(rf"\b(?:{_RECIPIENT_LABEL_KWS})\b", low):
            return True
        if re.search(r"\b(?:po\s*number|purchase\s*order)\b", low):
            return True
        if re.search(r"\b(?:invoice\s*(?:number|no\.?|#)|inv\s*no)\b", low):
            return True
        if re.search(r"\b(?:date|total|amount|gst|vat|tax)\b", low):
            return True
        return False

    def _labeled_value_ok(cand: str) -> bool:
        cand = cand.strip(" :-\t")
        if not cand:
            return False
        if _looks_like_contact_or_address(cand):
            return False
        if re.search(r"\b(?:gst|vat|tax|subtotal|total|amount\s*due)\b", cand, re.I):
            return False
        if re.search(r"\b(?:item|items|quantity|qty|price|description|unit\s*price|rate|sku)\b", cand, re.I):
            word_count = len(re.findall(r"[A-Za-z&'.-]+", cand))
            if re.search(r"\d", cand) or ":" in cand or word_count >= 3:
                return False
        if sum(ch.isalpha() for ch in cand) < 3:
            return False
        return True

    def _is_party_candidate(cand: str) -> bool:
        cand = cand.strip(" :-\t")
        if not _base_party_checks(cand):
            return False
        if role == "issuer" and _issuer_blocklist(cand):
            return False
        if role == "recipient":
            return _looks_like_name_line(cand)
        return _likely_company_line(cand) or _looks_like_name_line(cand)

    def _is_preferred_name_candidate(cand: str) -> bool:
        cand = cand.strip(" :-\t")
        return _base_party_checks(cand) and _looks_like_name_line(cand)

    if role == "recipient":
        _FUZZY_LABELS = (
            r"bill\s*.{0,2}o|buyer|invoice\s*.{0,2}o|sold\s*.{0,2}o|"
            r"ship\s*.{0,2}o|customer|client|recipient"
        )
    else:
        _FUZZY_LABELS = (
            r"bill\s*from|from|vendor|supplier|seller|issuer|remit\s*.{0,2}o"
        )
    fuzzy_inline_patterns = [
        rf"(?im)^\s*(?:{_FUZZY_LABELS})\s*\W{{0,3}}\s*([^\n]{{2,80}})$",
        rf"(?im)\b(?:{_FUZZY_LABELS})\s*\W{{0,3}}\s*([^\n]{{2,80}})",
    ]

    label_seen = False

    for pat in inline_patterns:
        for m in re.finditer(pat, text, flags=re.I):
            label_seen = True
            inline = _clean_inline_party_value(m.group(1))
            if _is_preferred_name_candidate(inline):
                return inline
            if _labeled_value_ok(inline):
                return inline

    if not label_seen:
        for pat in fuzzy_inline_patterns:
            for m in re.finditer(pat, text, flags=re.I):
                label_seen = True
                inline = _clean_inline_party_value(m.group(1))
                if _is_preferred_name_candidate(inline):
                    return inline
                if _labeled_value_ok(inline):
                    return inline

    anchored_hits: list[tuple[int, str]] = []
    for i, ln in enumerate(lines):
        low = ln.lower().strip()
        if not any(re.search(p, low) for p in label_patterns):
            continue
        label_seen = True

        if ":" in ln:
            inline = _clean_inline_party_value(ln.split(":", 1)[1])
            if _is_preferred_name_candidate(inline):
                return inline
            if _labeled_value_ok(inline):
                anchored_hits.append((10, inline))

        for j in range(i + 1, min(i + 4, len(lines))):
            cand = lines[j].strip(" :-\t")
            if _is_preferred_name_candidate(cand):
                return cand
            if _labeled_value_ok(cand):
                score = max(1, 8 - (j - i))
                anchored_hits.append((score, cand))
                break

    if anchored_hits:
        anchored_hits.sort(key=lambda x: x[0], reverse=True)
        return anchored_hits[0][1]

    if role == "recipient" and label_seen:
        return None

    def _party_score(ln: str) -> int:
        low = ln.lower()
        score = 0
        for kw in (
            "inc", "corp", "llc", "ltd", "group", "international", "company", "services", "solutions"
        ):
            if kw in low:
                score += 4
        if "&" in ln:
            score += 2
        if _looks_like_name_line(ln):
            score += 2
        if role == "recipient" and re.search(r"\b(?:buyer|customer|client|recipient)\b", low):
            score += 2
        if role == "issuer" and re.search(r"\b(?:vendor|supplier|seller|issuer)\b", low):
            score += 2
        return score

    top_candidates = sorted(
        [ln for ln in lines[:35] if _is_party_candidate(ln)],
        key=_party_score,
        reverse=True,
    )
    if role == "issuer" and top_candidates:
        return top_candidates[0]
    if role == "recipient" and len(top_candidates) >= 2:
        first = top_candidates[0]
        for cand in top_candidates[1:]:
            if cand[:32].lower() != first[:32].lower():
                return cand
        return top_candidates[1]
    if role == "recipient" and top_candidates:
        return top_candidates[0]
    return None


def _pick_total_from_amounts(amounts: list[str]) -> str | None:
    vals = sorted({float(a) for a in amounts if a}, reverse=True)
    if not vals:
        return None
    while len(vals) >= 2 and vals[0] > 25 * vals[1] and vals[1] >= 1:
        vals.pop(0)
    best = vals[0]
    if best >= 500_000 and len(vals) >= 2 and vals[1] < best / 10:
        best = vals[1]
    return f"{best:.2f}"


def _is_tax_or_subtotal_line(line: str) -> bool:
    low = line.lower()
    return bool(
        re.search(
            r"\b(?:gst|vat|tax\b|withholding|cgst|sgst|igst|hst|pst|qst|"
            r"subtotal|sub\s*total|item\s*total|line\s*total|before\s*tax|"
            r"taxable|tax\s*amount|net\s*amount|net\s*total)\b",
            low,
        )
    )


def _extract_total_amount(text: str, lines: list[str]) -> str | None:
    def _collect_amounts_from_lines(candidates_lines: list[str]) -> list[str]:
        vals: list[str] = []
        for ln in candidates_lines:
            for amt in AMOUNT_REGEX.findall(ln):
                norm = _normalize_amount(amt)
                if norm:
                    vals.append(norm)
        return vals

    final_total_label = (
        r"(?:grand\s*total|final\s*total|total\s*due|amount\s*due|"
        r"amount\s*payable|balance\s*due|invoice\s*total|amount\s*to\s*pay|total)"
    )

    direct_candidates: list[str] = []
    for ln in lines:
        if _is_tax_or_subtotal_line(ln):
            continue
        if not re.search(final_total_label, ln, re.I):
            continue
        m = re.search(
            rf"(?i)^\s*{final_total_label}(?!\s*[_-]?\s*(?:gst|vat|tax|sub\s*total|subtotal|net))\s*[:\-]?\s*({TOTAL_AMOUNT_CAPTURE})\s*$",
            ln,
        )
        if not m:
            m = re.search(
                rf"(?i)\b{final_total_label}(?!\s*[_-]?\s*(?:gst|vat|tax|sub\s*total|subtotal|net))\b\s*[:\-]?\s*({TOTAL_AMOUNT_CAPTURE})(?=\s*$)",
                ln,
            )
        if m:
            norm = _normalize_amount(m.group(1))
            if norm:
                direct_candidates.append(norm)

    if direct_candidates:
        picked = _pick_total_from_amounts(direct_candidates)
        if picked:
            return picked

    strongest_total_lines = [
        ln for ln in lines
        if re.search(final_total_label, ln, re.IGNORECASE)
        and not _is_tax_or_subtotal_line(ln)
    ]
    candidates = _collect_amounts_from_lines(strongest_total_lines)
    if candidates:
        picked = _pick_total_from_amounts(candidates)
        if picked:
            return picked

    secondary_total_lines = [
        ln for ln in lines
        if re.search(r"^\s*(?:total|grand\s*total|final\s*total)\b|\b(?:total\s*payable|total\s*amount|invoice\s*total)\b", ln, re.IGNORECASE)
        and not _is_tax_or_subtotal_line(ln)
    ]
    candidates = _collect_amounts_from_lines(secondary_total_lines)
    if candidates:
        picked = _pick_total_from_amounts(candidates)
        if picked:
            return picked

    return None


def _extract_parties_from_combined_labels(lines: list[str]) -> tuple[str | None, str | None]:
    """Handle OCR where 'Billed to' and 'From' appear on the same line."""
    combined_label_re = re.compile(
        r"(?i)\b(?:bill(?:ed)?\s*to|invoice\s*to|customer|client|recipient)\b.*\bfrom\b"
    )
    stop_re = re.compile(
        r"(?i)\b(?:item|items|quantity|qty|price|amount|subtotal|total|payment|note|invoice\s*#|invoice\s*number)\b"
    )
    for i, ln in enumerate(lines[:60]):
        if not combined_label_re.search(ln):
            continue
        for j in range(i + 1, min(i + 5, len(lines))):
            cand = lines[j].strip(" :-\t")
            if not cand or stop_re.search(cand):
                break
            left, right = _split_possible_dual_names(cand)
            if left and right:
                return left, right
    return None, None


def _labeled_due_date_from_lines(lines: list[str], invoice_date: str | None) -> str | None:
    skip_start = re.compile(
        r"(?i)^(the|upon|receipt|immediately|cash|wire|ach|transfer|see\s+reverse|t\.?\s*b\.?\s*d\.?|asap)\b",
    )
    for ln in lines[:120]:
        s = ln.strip()
        if not s:
            continue
        for pat in _DUE_LINE_PATTERNS:
            m = pat.search(s)
            if not m:
                continue
            frag = m.group(1).strip()
            if not frag or skip_start.match(frag):
                continue
            pd = _parse_any_date_in_chunk(frag, prefer_last=True)
            if not pd:
                continue
            try:
                y = datetime.fromisoformat(pd).year
                if DATE_YEAR_MIN > y or y > DATE_YEAR_MAX:
                    continue
            except Exception:
                continue
            if invoice_date and pd == invoice_date:
                continue
            return pd
    return None


def _due_dates_from_dueish_lines(lines: list[str], invoice_date: str | None) -> str | None:
    for ln in lines[:150]:
        s = ln.strip()
        if not s or not _DUEISH_LINE_HINT.search(s):
            continue
        if re.search(r"(?i)\b(invoice|issue|issued)\s*date\b", s) and not re.search(r"(?i)\bdue\b", s):
            continue
        dates: list[str] = []
        for m in _INLINE_DATE_ANY.finditer(s):
            if (d := _parse_date_candidate(m.group(0))):
                dates.append(d)
        if not dates and (d := _parse_date_candidate(s)):
            dates.append(d)
        if not dates:
            continue
        if invoice_date:
            for pd in dates:
                if pd != invoice_date:
                    return pd
            return dates[-1]
        return dates[-1]
    return None


def _sanitize_date_pair(invoice_date: str | None, due_date: str | None) -> tuple[str | None, str | None]:
    def _valid_year(d: str | None) -> bool:
        if not d:
            return False
        try:
            y = datetime.fromisoformat(d).year
            return DATE_YEAR_MIN <= y <= DATE_YEAR_MAX
        except Exception:
            return False

    if not _valid_year(invoice_date):
        invoice_date = None
    if not _valid_year(due_date):
        due_date = None
    return invoice_date, due_date


def _extract_fields_generic(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    invoice_number = _extract_invoice_number(text, lines)

    invoice_date_raw = _first_match(
        [
            r"invoice\s*date\s*[:\-]?\s*([^\n]{3,45})",
            r"date\s*of\s*issue\s*[:\-]?\s*([^\n]{3,45})",
            r"(?:issue|issued)\s*date\s*[:\-]?\s*([^\n]{3,45})",
            r"(?im)^\s*Date\s*:\s*([^\n]{3,45})",
            r"(?m)^\s*Date\s*[:\-]\s*([0-9]{1,2}[-/][A-Za-z]{3,9}[-/][0-9]{2,4})",
            r"dated\s*[:\-]?\s*([^\n]{3,45})",
        ],
        text,
    )
    due_date_raw = _first_match(
        [
            r"(?im)^\s*Due\s+Date\s*:\s*([^\n;]{3,55})",
            r"(?i)\bdue\s+date\s*:\s*([^\n;]{3,55})",
            r"(?is)due\s*date\s*[:.\-]?\s*\n\s*([^\n;]{3,55})",
            r"due\s*date\s*[:.\-]?\s*([^\n;]{3,55})",
            r"date\s*due\s*[:.\-]?\s*([^\n;]{3,55})",
            r"payment\s*due\s*[:.\-]?\s*([^\n;]{3,55})",
            r"(?:latest\s*)?payment\s*(?:by|before)\s*[:.\-]?\s*([^\n;]{3,55})",
            r"pay(?:ment)?\s*by\s*[:.\-]?\s*([^\n;]{3,55})",
            r"remit(?:tance)?\s*(?:by|before)?\s*[:.\-]?\s*([^\n;]{3,55})",
            r"balance\s*due\s*[:.\-]?\s*([^\n;]{3,55})",
            r"(?:deadline|good\s*until)\s*[:.\-]?\s*([^\n;]{3,55})",
            r"(?m)^\s*due\s*[:.\-]\s*([^\n;]{3,55})",
            r"(?i)\b(?:total\s+)?payment\s+due\s+(?:in|within)\s*(\d{1,3})\s*days?\b",
            r"(?i)\bdue\s+(?:in|within)\s*(\d{1,3})\s*days?\b",
            r"(?i)\bpay(?:ment)?\s+(?:in|within)\s*(\d{1,3})\s*days?\b",
            r"(?i)\bwithin\s*(\d{1,3})\s*days?\s*(?:of\s*(?:invoice|issue|receipt))?\b",
            r"(?i)\bterms?\s*[:.\-]?\s*(?:pay(?:ment)?\s*)?(?:in|within)\s*(\d{1,3})\s*days?\b",
            r"(?:payment\s*)?terms\s*[:.\-]?\s*net\s*(\d{1,3})\s*days?",
            r"\bnet\s*(\d{1,3})\s*days?\b",
        ],
        text,
    )

    date_candidates = [d for d in DATE_REGEX.findall(text)]
    loose_date_candidates = re.findall(r"\b\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{2,4}\b", text)
    alpha_month_dates = re.findall(r"\b\d{1,2}\s*[-/]\s*[A-Za-z]{3,9}\s*[-/]\s*\d{2,4}\b", text)

    invoice_date = _parse_any_date_in_chunk(invoice_date_raw)
    due_date = _parse_any_date_in_chunk(due_date_raw, prefer_last=True)

    if not invoice_date and date_candidates:
        invoice_date = _parse_date_candidate(date_candidates[0])
    if not invoice_date and loose_date_candidates:
        invoice_date = _parse_date_candidate(loose_date_candidates[0])
    if not invoice_date and alpha_month_dates:
        invoice_date = _parse_date_candidate(alpha_month_dates[0])

    if not due_date and len(date_candidates) > 1:
        due_date = _parse_date_candidate(date_candidates[1])
    if not due_date and len(loose_date_candidates) > 1:
        due_date = _parse_date_candidate(loose_date_candidates[1])
    if not due_date and len(alpha_month_dates) > 1:
        due_date = _parse_date_candidate(alpha_month_dates[1])

    if not due_date:
        due_date = _labeled_due_date_from_lines(lines, invoice_date)
    if not due_date:
        due_date = _due_dates_from_dueish_lines(lines, invoice_date)
    if not due_date and invoice_date:
        due_date = _labeled_due_date_from_lines(lines, None) or _due_dates_from_dueish_lines(lines, None)

    due_days = _extract_days_term(due_date_raw)
    if not due_date and due_days and invoice_date:
        try:
            inv_dt = datetime.fromisoformat(invoice_date)
            due_date = (inv_dt + timedelta(days=due_days)).date().isoformat()
        except Exception:
            pass

    invoice_date, due_date = _sanitize_date_pair(invoice_date, due_date)

    recipient_name, issuer_name = _extract_parties_from_combined_labels(lines)
    if not issuer_name:
        issuer_name = _extract_party(text, lines, "issuer")
    if not recipient_name:
        recipient_name = _extract_party(text, lines, "recipient")
    issuer_name = _clean_party_name(issuer_name)
    recipient_name = _clean_party_name(recipient_name)
    total_amount = _extract_total_amount(text, lines)

    if recipient_name and issuer_name and recipient_name.strip().lower() == issuer_name.strip().lower():
        recipient_name = None

    return {
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "due_date": due_date,
        "issuer_name": issuer_name,
        "recipient_name": recipient_name,
        "total_amount": total_amount,
    }


def _extract_block_between(text: str, start_label: str, stop_label: str) -> str | None:
    pattern = rf"{start_label}\s*:\s*(.*?){stop_label}\s*:"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    block = m.group(1).strip()
    return block if block else None


def _looks_like_name(line: str) -> bool:
    low = line.lower().strip()
    if not low:
        return False
    if re.search(r"invoice|date|issue|tax\s*id|iban|items|summary|total|qty|gross|net|vat", low):
        return False
    if sum(ch.isdigit() for ch in line) > 2:
        return False
    words = re.findall(r"[A-Za-z&'.,-]+", line)
    return len(words) >= 1 and sum(ch.isalpha() for ch in line) >= 4


def _first_name_in_block(block: str | None) -> str | None:
    if not block:
        return None
    lines = [ln.strip(" :-\t") for ln in block.splitlines() if ln.strip()]
    for ln in lines:
        cleaned = re.sub(r"\b(?:seller|client)\s*:\s*", "", ln, flags=re.IGNORECASE).strip()
        if cleaned.lower() in {"seller", "client"}:
            continue
        if _looks_like_name(cleaned):
            return cleaned
    return None


def _extract_clear_money_values(text: str) -> list[str]:
    strict_money = re.findall(r"\b\d{1,3}(?:\s\d{3})*(?:[.,]\d{2})\b", text)
    vals = [_normalize_amount(v) for v in strict_money]
    return [v for v in vals if v is not None and float(v) < 1_000_000]


def _split_possible_dual_names(line: str) -> tuple[str | None, str | None]:
    tokens = [t for t in line.split() if t]
    if len(tokens) < 4:
        return None, None
    corp_words = ("inc", "inc.", "ltd", "ltd.", "llc", "plc", "corp", "corporation")

    if tokens[-1].lower().strip(".,") in corp_words and len(tokens) >= 4:
        left = " ".join(tokens[:-2]).strip(" ,")
        right = " ".join(tokens[-2:]).strip(" ,")
        if _looks_like_name(left) and _looks_like_name(right):
            return left, right

    for i in range(2, len(tokens) - 1):
        left = " ".join(tokens[:i]).strip(" ,")
        right = " ".join(tokens[i:]).strip(" ,")
        if not (_looks_like_name(left) and _looks_like_name(right)):
            continue
        if any(w in right.lower() for w in corp_words) and not any(w in left.lower() for w in corp_words):
            return left, right

    for i in range(2, len(tokens) - 1):
        left = " ".join(tokens[:i]).strip(" ,")
        right = " ".join(tokens[i:]).strip(" ,")
        if _looks_like_name(left) and _looks_like_name(right):
            lw, rw = len(left.split()), len(right.split())
            if 1 <= lw <= 5 and 1 <= rw <= 5:
                return left, right

    return None, None


def _extract_fields_clear_layout(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    invoice_no_raw = _first_match(
        [
            r"invoice\s*no\.?\s*:\s*([A-Z0-9\-/]{4,})",
            r"invoice\s*(?:number|#)\s*:\s*([A-Z0-9\-/]{4,})",
        ],
        text,
    )
    invoice_number = invoice_no_raw if _is_valid_invoice_number(invoice_no_raw) else None

    invoice_date_raw = _first_match(
        [
            r"date\s*of\s*issue\s*:\s*([0-9]{1,2}[\-/][0-9]{1,2}[\-/][0-9]{2,4})",
            r"date\s*of\s*issue\s*:\s*\n\s*([0-9]{1,2}[\-/][0-9]{1,2}[\-/][0-9]{2,4})",
        ],
        text,
    )
    invoice_date = _parse_date_candidate(invoice_date_raw)
    if not invoice_date:
        loose_dates = re.findall(r"\b\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}\b", text)
        parsed_dates = [d for d in (_parse_date_candidate(x) for x in loose_dates) if d]
        if parsed_dates:
            invoice_date = parsed_dates[0]

    due_date_raw = _first_match(
        [
            r"due\s*date\s*:\s*([0-9]{1,2}[\-/][0-9]{1,2}[\-/][0-9]{2,4})",
            r"payment\s*due\s*:\s*([0-9]{1,2}[\-/][0-9]{1,2}[\-/][0-9]{2,4})",
            r"(?i)\b(?:total\s+)?payment\s+due\s+(?:in|within)\s*(\d{1,3})\s*days?\b",
            r"(?i)\bdue\s+(?:in|within)\s*(\d{1,3})\s*days?\b",
            r"(?i)\bpay(?:ment)?\s+(?:in|within)\s*(\d{1,3})\s*days?\b",
            r"(?i)\bwithin\s*(\d{1,3})\s*days?\s*(?:of\s*(?:invoice|issue|receipt))?\b",
            r"terms\s*:\s*net\s*(\d{1,3})\s*days",
        ],
        text,
    )
    due_date = _parse_date_candidate(due_date_raw)
    due_days = _extract_days_term(due_date_raw)
    if not due_date and due_days and invoice_date:
        try:
            inv_dt = datetime.fromisoformat(invoice_date)
            due_date = (inv_dt + timedelta(days=due_days)).date().isoformat()
        except Exception:
            pass
    if due_date and invoice_date and due_date == invoice_date:
        due_date = None

    seller_block = _extract_block_between(text, "Seller", "Tax\\s*Id")
    client_block = _extract_block_between(text, "Client", "Tax\\s*Id")
    issuer_name = _first_name_in_block(seller_block)
    recipient_name = _first_name_in_block(client_block)

    if not recipient_name or (issuer_name and recipient_name and issuer_name == recipient_name):
        dual = re.search(r"seller\s*:\s*client\s*:?\s*\n\s*([^\n]{6,120})", text, flags=re.IGNORECASE)
        if dual:
            left, right = _split_possible_dual_names(dual.group(1))
            if left and right:
                issuer_name, recipient_name = left, right

    if not issuer_name:
        issuer_name = _first_match([r"seller\s*:\s*([^\n:]{3,80})"], text)
    if not recipient_name:
        recipient_name = _first_match([r"client\s*:\s*([^\n:]{3,80})"], text)

    if issuer_name and recipient_name and issuer_name == recipient_name:
        left, right = _split_possible_dual_names(issuer_name)
        if left and right:
            issuer_name, recipient_name = left, right

    total_amount = None
    total_line = next((ln for ln in lines if re.search(r"^total\b", ln, re.IGNORECASE)), None)
    if total_line:
        vals = _extract_clear_money_values(total_line)
        if vals:
            total_amount = vals[-1]

    if not total_amount:
        summary_block = text.split("SUMMARY", 1)[1] if "SUMMARY" in text else text
        summary_vals = _extract_clear_money_values(summary_block)
        if summary_vals:
            total_amount = summary_vals[-1]

    return {
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "due_date": due_date,
        "issuer_name": issuer_name,
        "recipient_name": recipient_name,
        "total_amount": total_amount,
    }


def _looks_like_invoice_clear_template(text: str) -> bool:
    if "SUMMARY" not in text.upper():
        return False
    if not re.search(r"(?i)\binvoice\s*no\.?\s*:", text):
        return False
    seller = _extract_block_between(text, "Seller", "Tax\\s*Id")
    client = _extract_block_between(text, "Client", "Tax\\s*Id")
    return bool(seller and client)


def extract_fields(text: str, layout: str = "auto") -> dict:
    """Extract invoice fields from text.

    Args:
        text: OCR/extracted invoice text.
        layout: "auto", "clear", or "generic".
    """
    layout = (layout or "auto").strip().lower()
    if layout not in {"auto", "clear", "generic"}:
        layout = "auto"
    if layout == "clear" or (layout == "auto" and _looks_like_invoice_clear_template(text)):
        return _extract_fields_clear_layout(text)
    return _extract_fields_generic(text)


def extract_fields_from_file(path: str | Path, layout: str = "auto") -> dict:
    """Extract invoice fields directly from a PDF/image path."""
    text = extract_text_robust(path)
    return extract_fields(text, layout=layout)


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python invoice_extractor.py <path-to-file> [layout]")
    src_path = Path(sys.argv[1])
    src_layout = sys.argv[2] if len(sys.argv) > 2 else "auto"
    result = extract_fields_from_file(src_path, layout=src_layout)
    print(json.dumps(result, indent=2, ensure_ascii=True))
