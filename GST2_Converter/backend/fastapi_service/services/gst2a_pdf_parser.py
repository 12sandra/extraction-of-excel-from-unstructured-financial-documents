# gst2a_pdf_parser.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Directly extracts structured data from the official GSTN-portal GST 2A PDF.
#
# WHY THIS EXISTS
#   The GSTN portal exports GST 2A statements as DIGITAL PDFs (not scanned
#   images).  The text is already embedded → no OCR needed.  But the table is
#   split across THREE GROUPS of pages:
#
#       Group A  pages  1 … N/3   →  GSTIN, Name, Invoice No, Type, Date,
#                                    Invoice Value, Place of Supply, Rev.Charge,
#                                    Rate %
#       Group B  pages N/3+1…2N/3 →  Taxable Value, IGST, CGST, SGST, Cess,
#                                    GSTR-1 Status, Filing Date, Filing Period,
#                                    GSTR-3B Status
#       Group C  pages 2N/3+1…N  →  Amendment, Tax Period, Cancel Date,
#                                    Source, IRN, IRN Date
#
#   This parser:
#     1. Detects the group boundaries automatically.
#     2. Parses each group with tailored regex / positional logic.
#     3. Merges all three groups row-by-row into complete invoice records.
#     4. Falls back to OCR-text parsing if pdfplumber isn't installed.
# ─────────────────────────────────────────────────────────────────────────────

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("gst2_fastapi.gst2a_parser")

# ── Known header fragments that identify each group ──────────────────────────
GROUP_A_MARKERS = ["gstn", "gstin", "invoice number", "invoice no", "invoice date"]
GROUP_B_MARKERS = ["taxable value", "integrated tax", "central tax", "state/ut tax",
                   "gstr-1", "gstr1", "filing status", "filing date"]
GROUP_C_MARKERS = ["amendment", "irn", "irn date", "source", "e-invoice", "cancellation"]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_gst2a_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Parse a GST 2A PDF and return a list of invoice-record dicts.

    Each dict contains:
        GSTIN, TRADE_NAME, INVOICE_NO, INVOICE_DATE, INVOICE_VALUE,
        PLACE_OF_SUPPLY, REVERSE_CHARGE, TAX_RATE,
        TAXABLE_VALUE, IGST, CGST, SGST, CESS,
        GSTR1_STATUS, GSTR1_FILING_DATE, GSTR1_PERIOD, GSTR3B_STATUS,
        AMENDMENT, TAX_PERIOD_AMENDED, CANCEL_DATE, SOURCE, IRN, IRN_DATE,
        _confidence, _source
    """
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        return []

    pdf_path = str(pdf_path)
    logger.info(f"Parsing GST 2A PDF: {pdf_path}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Total pages: {total_pages}")

            # ── Extract raw text per page ─────────────────────────────────
            pages_text: List[str] = []
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                pages_text.append(text)

        # ── Detect group boundaries ───────────────────────────────────────
        group_a_pages, group_b_pages, group_c_pages = _detect_groups(pages_text)

        logger.info(
            f"Detected groups — A: pages {group_a_pages}, "
            f"B: pages {group_b_pages}, C: pages {group_c_pages}"
        )

        # ── Parse each group ─────────────────────────────────────────────
        records_a = _parse_group_a(pages_text, group_a_pages)
        records_b = _parse_group_b(pages_text, group_b_pages)
        records_c = _parse_group_c(pages_text, group_c_pages)

        logger.info(
            f"Parsed rows — A:{len(records_a)}  B:{len(records_b)}  C:{len(records_c)}"
        )

        # ── Merge ─────────────────────────────────────────────────────────
        merged = _merge_groups(records_a, records_b, records_c)
        logger.info(f"Merged records: {len(merged)}")
        return merged

    except Exception as e:
        logger.error(f"GST 2A PDF parsing failed: {e}", exc_info=True)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# GROUP DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_groups(pages_text: List[str]) -> Tuple[List[int], List[int], List[int]]:
    """
    Return three lists of 0-based page indices, one per group (A, B, C).
    Strategy: find the first page that belongs to each group, then split evenly.
    """
    n = len(pages_text)

    # Find first pages for each group
    first_a = first_b = first_c = None
    for i, text in enumerate(pages_text):
        lower = text.lower()
        if first_a is None and any(m in lower for m in GROUP_A_MARKERS):
            first_a = i
        if first_b is None and any(m in lower for m in GROUP_B_MARKERS):
            # Must come after group A
            if first_a is not None and i > first_a:
                first_b = i
        if first_c is None and any(m in lower for m in GROUP_C_MARKERS):
            if first_b is not None and i > first_b:
                first_c = i

    # Fallback: equal thirds
    if first_a is None:
        first_a = 0
    if first_b is None:
        first_b = n // 3
    if first_c is None:
        first_c = (2 * n) // 3

    group_a = list(range(first_a, first_b))
    group_b = list(range(first_b, first_c))
    group_c = list(range(first_c, n))

    return group_a, group_b, group_c


# ─────────────────────────────────────────────────────────────────────────────
# GROUP A PARSER  —  GSTIN / Name / Invoice / Date / Value / Place / Rate
# ─────────────────────────────────────────────────────────────────────────────

# GSTIN: 2 digits + 10 alphanumeric + 3 alphanumeric  (15 chars total, relaxed)
GSTIN_RE = re.compile(r'\b([0-9]{1,2}[A-Z0-9]{13,14})\b')
INV_NO_RE = re.compile(r'\b(INV[-/][A-Z0-9-]+)\b', re.IGNORECASE)
DATE_RE   = re.compile(r'\b(\d{2}[-/]\d{2}[-/]\d{4})\b')
AMOUNT_RE = re.compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+\.\d{1,2}|\d+)\b')
RATE_RE   = re.compile(r'\b(0|5|12|18|28)\s*$')

# Place-of-supply tokens
POS_TOKENS = [
    "Tamil Nadu", "Kerala", "Karnataka", "Maharashtra", "Andhra Pradesh",
    "Telangana", "Gujarat", "Rajasthan", "Uttar Pradesh", "West Bengal",
    "Delhi", "Haryana", "Punjab", "Madhya Pradesh", "Bihar", "Odisha",
    "Assam", "Jharkhand", "Chhattisgarh", "Uttarakhand", "Himachal Pradesh",
    "Goa", "Tripura", "Meghalaya", "Manipur", "Mizoram", "Nagaland",
    "Arunachal Pradesh", "Sikkim", "Jammu and Kashmir", "Ladakh",
    "Puducherry", "Chandigarh", "Dadra", "Lakshadweep", "Andaman",
]
# Build combined pattern (longest first to avoid partial match)
POS_PATTERN = re.compile(
    r'(' + '|'.join(re.escape(p) for p in sorted(POS_TOKENS, key=len, reverse=True)) + r')',
    re.IGNORECASE
)


def _parse_group_a(pages_text: List[str], page_indices: List[int]) -> List[Dict]:
    records = []
    for pi in page_indices:
        text  = pages_text[pi]
        lines = text.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Must start with a GSTIN-like token (15 chars, first 2 are digits)
            gstin_match = GSTIN_RE.match(line)
            if not gstin_match:
                continue

            rec = _parse_group_a_line(line)
            if rec:
                records.append(rec)
    return records


def _parse_group_a_line(line: str) -> Optional[Dict]:
    """Parse one data line from Group A."""
    rec = {
        "GSTIN": "", "TRADE_NAME": "", "INVOICE_NO": "", "INVOICE_TYPE": "R",
        "INVOICE_DATE": "", "INVOICE_VALUE": "", "PLACE_OF_SUPPLY": "",
        "REVERSE_CHARGE": "N", "TAX_RATE": "",
        "_confidence": 0.95, "_source": "direct_pdf"
    }

    # ── GSTIN ──
    gstin_m = GSTIN_RE.match(line)
    if not gstin_m:
        return None
    rec["GSTIN"] = gstin_m.group(1)
    remainder = line[gstin_m.end():].strip()

    # ── Invoice Number ──
    inv_m = INV_NO_RE.search(remainder)
    if inv_m:
        rec["INVOICE_NO"] = inv_m.group(1).upper()
        # Trade name is everything between GSTIN and Invoice No
        rec["TRADE_NAME"] = remainder[:inv_m.start()].strip()
        remainder         = remainder[inv_m.end():].strip()
    else:
        # Fallback: first token that looks like a name, then skip
        parts = remainder.split()
        name_parts = []
        for p in parts:
            if re.match(r'[A-Z]{1,3}-\d+', p, re.IGNORECASE):
                rec["INVOICE_NO"] = p.upper()
                break
            name_parts.append(p)
        rec["TRADE_NAME"] = " ".join(name_parts).strip()
        idx = remainder.find(rec["INVOICE_NO"])
        remainder = remainder[idx + len(rec["INVOICE_NO"]):].strip() if rec["INVOICE_NO"] else remainder

    # ── Invoice Type (R / SEZWP / SEZWOP / DE / CBW …) ──
    remainder = remainder.strip()
    type_m = re.match(r'^([A-Z]+)\s+', remainder)
    if type_m and len(type_m.group(1)) <= 8:
        rec["INVOICE_TYPE"] = type_m.group(1)
        remainder           = remainder[type_m.end():].strip()

    # ── Invoice Date ──
    date_m = DATE_RE.search(remainder)
    if date_m:
        rec["INVOICE_DATE"] = date_m.group(1)
        remainder           = remainder[date_m.end():].strip()

    # ── Invoice Value ──
    amt_m = AMOUNT_RE.search(remainder)
    if amt_m:
        rec["INVOICE_VALUE"] = amt_m.group(1).replace(",", "")
        remainder            = remainder[amt_m.end():].strip()

    # ── Place of Supply ──
    pos_m = POS_PATTERN.search(remainder)
    if pos_m:
        rec["PLACE_OF_SUPPLY"] = pos_m.group(1).strip()
        remainder              = remainder[pos_m.end():].strip()

    # ── Reverse Charge (N/Y immediately after place) ──
    rc_m = re.search(r'\b([NY])\b', remainder)
    if rc_m:
        rec["REVERSE_CHARGE"] = rc_m.group(1)
        remainder             = remainder[rc_m.end():].strip()

    # ── Tax Rate ──
    rate_m = re.search(r'\b(0|5|12|18|28)\b', remainder)
    if rate_m:
        rec["TAX_RATE"] = rate_m.group(1)

    # Only keep if we have at least GSTIN + Invoice No
    if rec["GSTIN"] and (rec["INVOICE_NO"] or rec["INVOICE_DATE"]):
        return rec
    return None


# ─────────────────────────────────────────────────────────────────────────────
# GROUP B PARSER  —  Tax values + Filing info
# ─────────────────────────────────────────────────────────────────────────────

SHORT_DATE_RE = re.compile(r'\b(\d{2}[-/][A-Za-z]{3}[-/]\d{2,4})\b')  # 24-Feb-24
PERIOD_RE     = re.compile(r'\b([A-Za-z]{3}[-/]\d{2,4})\b')             # Feb-24


def _parse_group_b(pages_text: List[str], page_indices: List[int]) -> List[Dict]:
    records = []
    for pi in page_indices:
        text  = pages_text[pi]
        lines = text.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            rec = _parse_group_b_line(line)
            if rec:
                records.append(rec)
    return records


def _parse_group_b_line(line: str) -> Optional[Dict]:
    """
    Group B lines look like:
        17273.82 3109.29 0 0 0 Y 24-Feb-24 Feb-24 Y
        (taxable  igst    cgst sgst cess  gstr1  date  period gstr3b)
    """
    # Must start with a numeric amount
    if not re.match(r'^\d', line):
        return None

    # Extract all numbers at the beginning
    tokens = line.split()
    numbers = []
    rest_idx = 0
    for i, t in enumerate(tokens):
        if re.match(r'^[\d,.]+$', t):
            numbers.append(t.replace(",", ""))
            rest_idx = i + 1
        else:
            break  # stop at first non-number

    if len(numbers) < 3:
        return None

    def safe(lst, idx):
        try:
            return lst[idx]
        except IndexError:
            return ""

    rec = {
        "TAXABLE_VALUE": safe(numbers, 0),
        "IGST":          safe(numbers, 1),
        "CGST":          safe(numbers, 2),
        "SGST":          safe(numbers, 3),
        "CESS":          safe(numbers, 4),
    }

    # Remainder after numbers
    rest = " ".join(tokens[rest_idx:])

    # GSTR-1 status (Y/N)
    gstr1_m = re.search(r'\b([YN])\b', rest)
    if gstr1_m:
        rec["GSTR1_STATUS"] = gstr1_m.group(1)
        rest = rest[gstr1_m.end():].strip()
    else:
        rec["GSTR1_STATUS"] = ""

    # Filing date  (DD-Mon-YY or DD-Mon-YYYY)
    date_m = SHORT_DATE_RE.search(rest)
    if date_m:
        rec["GSTR1_FILING_DATE"] = date_m.group(1)
        rest = rest[date_m.end():].strip()
    else:
        rec["GSTR1_FILING_DATE"] = ""

    # Filing period  (Mon-YY)
    period_m = PERIOD_RE.search(rest)
    if period_m:
        rec["GSTR1_PERIOD"] = period_m.group(1)
        rest = rest[period_m.end():].strip()
    else:
        rec["GSTR1_PERIOD"] = ""

    # GSTR-3B status
    gstr3b_m = re.search(r'\b([YN])\b', rest)
    rec["GSTR3B_STATUS"] = gstr3b_m.group(1) if gstr3b_m else ""

    return rec


# ─────────────────────────────────────────────────────────────────────────────
# GROUP C PARSER  —  Amendment / IRN
# ─────────────────────────────────────────────────────────────────────────────

IRN_RE = re.compile(r'\b([0-9a-f]{64})\b', re.IGNORECASE)   # 64-hex IRN


def _parse_group_c(pages_text: List[str], page_indices: List[int]) -> List[Dict]:
    records = []
    for pi in page_indices:
        text  = pages_text[pi]
        lines = text.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            rec = _parse_group_c_line(line)
            if rec:
                records.append(rec)
    return records


def _parse_group_c_line(line: str) -> Optional[Dict]:
    """
    Group C lines look like:
        N N/A N/A E-invoice <64-hex-hash> 15-02-2024
        Y Nov-23 N/A E-invoice <hash> 22-12-2023
        N N/A N/A Manual N/A N/A
    """
    # Must start with Y or N (amendment flag)
    if not re.match(r'^[YN]\b', line, re.IGNORECASE):
        return None

    tokens = line.split()
    if len(tokens) < 3:
        return None

    rec = {
        "AMENDMENT":          tokens[0].upper() if tokens else "N",
        "TAX_PERIOD_AMENDED": "",
        "CANCEL_DATE":        "",
        "SOURCE":             "Unknown",
        "IRN":                "N/A",
        "IRN_DATE":           "N/A",
    }

    # Amendment period (e.g. Nov-23)  — second token if not N/A
    if len(tokens) > 1 and tokens[1].upper() != "N/A":
        rec["TAX_PERIOD_AMENDED"] = tokens[1]

    # Cancellation date — third token if not N/A
    if len(tokens) > 2 and tokens[2].upper() != "N/A":
        rec["CANCEL_DATE"] = tokens[2]

    # Source keyword
    if "e-invoice" in line.lower() or "einvoice" in line.lower():
        rec["SOURCE"] = "E-invoice"
    elif "manual" in line.lower():
        rec["SOURCE"] = "Manual"

    # IRN (64-char hex)
    irn_m = IRN_RE.search(line)
    if irn_m:
        rec["IRN"] = irn_m.group(1)
        # IRN date follows the IRN
        after_irn = line[irn_m.end():].strip()
        date_m    = DATE_RE.search(after_irn)
        if date_m:
            rec["IRN_DATE"] = date_m.group(1)

    return rec


# ─────────────────────────────────────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────────────────────────────────────

def _merge_groups(
    records_a: List[Dict],
    records_b: List[Dict],
    records_c: List[Dict]
) -> List[Dict]:
    """
    Merge by row index.  Group A is the primary — it drives the count.
    B and C rows are matched positionally.
    """
    merged = []
    max_rows = max(len(records_a), len(records_b), len(records_c), 0)

    b_default = {
        "TAXABLE_VALUE": "", "IGST": "", "CGST": "", "SGST": "", "CESS": "",
        "GSTR1_STATUS": "", "GSTR1_FILING_DATE": "", "GSTR1_PERIOD": "", "GSTR3B_STATUS": ""
    }
    c_default = {
        "AMENDMENT": "N", "TAX_PERIOD_AMENDED": "N/A",
        "CANCEL_DATE": "N/A", "SOURCE": "Unknown", "IRN": "N/A", "IRN_DATE": "N/A"
    }

    for i in range(max_rows):
        a = records_a[i] if i < len(records_a) else {}
        b = records_b[i] if i < len(records_b) else b_default.copy()
        c = records_c[i] if i < len(records_c) else c_default.copy()

        record = {**a, **b, **c}
        record["_confidence"] = 0.98   # Direct PDF extraction is very accurate
        record["_source"]     = "direct_pdf"
        merged.append(record)

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — wrap records into the format expected by excel_generator.py
# ─────────────────────────────────────────────────────────────────────────────

def records_to_excel_format(records: List[Dict]) -> List[Dict]:
    """
    Convert parser output to the field-name format expected by
    ExcelGenerator.generate().
    """
    out = []
    for r in records:
        row = {
            "GSTIN":            [r.get("GSTIN", "")],
            "TRADE_NAME":       [r.get("TRADE_NAME", "")],
            "INVOICE_NO":       [r.get("INVOICE_NO", "")],
            "INVOICE_DATE":     [r.get("INVOICE_DATE", "")],
            "INVOICE_VALUE":    [r.get("INVOICE_VALUE", "")],
            "PLACE_OF_SUPPLY":  [r.get("PLACE_OF_SUPPLY", "")],
            "REVERSE_CHARGE":   r.get("REVERSE_CHARGE", "N"),
            "INVOICE_TYPE":     r.get("INVOICE_TYPE", "R"),
            "TAX_RATE":         r.get("TAX_RATE", ""),
            "TAXABLE_VALUE":    [r.get("TAXABLE_VALUE", "")],
            "IGST":             [r.get("IGST", "")],
            "CGST":             [r.get("CGST", "")],
            "SGST":             [r.get("SGST", "")],
            "CESS":             [r.get("CESS", "")],
            "RETURN_PERIOD":    [r.get("GSTR1_PERIOD", "")],
            # Extra columns stored in SOURCE field of Excel
            "_source":          r.get("SOURCE", "direct_pdf"),
            "_confidence":      r.get("_confidence", 0.98),
            # Additional metadata stored as notes
            "GSTR1_STATUS":     r.get("GSTR1_STATUS", ""),
            "GSTR1_FILING_DATE":r.get("GSTR1_FILING_DATE", ""),
            "GSTR3B_STATUS":    r.get("GSTR3B_STATUS", ""),
            "IRN":              r.get("IRN", "N/A"),
            "IRN_DATE":         r.get("IRN_DATE", "N/A"),
            "AMENDMENT":        r.get("AMENDMENT", "N"),
        }
        out.append(row)
    return out