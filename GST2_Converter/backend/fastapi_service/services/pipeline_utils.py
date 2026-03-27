# pipeline_utils.py — Shared helper functions for the OCR pipeline path

import logging
from typing import List, Dict, Any

logger = logging.getLogger("gst2_fastapi.pipeline_utils")


def merge_table_and_fields(
    tables: List[Dict[str, Any]],
    classified: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Combine Paddle table results with LayoutLMv3 field classifications
    into a unified list of invoice records.
    """
    records = []

    # ── Try table-based extraction first ────────────────────────────────────
    for table in tables:
        rows = table.get("rows", [])
        if len(rows) < 2:
            continue

        headers = [cell.lower().strip() for cell in rows[0].get("cells", [])]

        for row in rows[1:]:
            cells = row.get("cells", [])
            if not any(cells):
                continue

            record: Dict[str, Any] = {}

            for header, cell in zip(headers, cells):
                h = header.lower()
                if "gstin" in h:
                    record["GSTIN"] = [cell]
                elif "trade" in h or ("name" in h and "trade" in h):
                    record["TRADE_NAME"] = [cell]
                elif "invoice" in h and "no" in h:
                    record["INVOICE_NO"] = [cell]
                elif "invoice" in h and "date" in h:
                    record["INVOICE_DATE"] = [cell]
                elif "invoice" in h and "value" in h:
                    record["INVOICE_VALUE"] = [cell]
                elif "taxable" in h:
                    record["TAXABLE_VALUE"] = [cell]
                elif "igst" in h or "integrated" in h:
                    record["IGST"] = [cell]
                elif "cgst" in h or "central" in h:
                    record["CGST"] = [cell]
                elif "sgst" in h or "state" in h or "utgst" in h:
                    record["SGST"] = [cell]
                elif "cess" in h:
                    record["CESS"] = [cell]
                elif "return" in h and "period" in h:
                    record["RETURN_PERIOD"] = [cell]
                elif "place" in h and "supply" in h:
                    record["PLACE_OF_SUPPLY"] = [cell]
                elif "rate" in h:
                    record["TAX_RATE"] = cell
                elif "reverse" in h:
                    record["REVERSE_CHARGE"] = cell

            if record:
                record["_confidence"] = 0.90
                record["_source"]     = "table_extraction"
                records.append(record)

    # ── Fall back to LayoutLMv3 classified output if no table records ────────
    if not records:
        for page_data in classified:
            record = {k: v for k, v in page_data.items() if not k.startswith("_")}
            record["_confidence"] = page_data.get("_confidence", 0.0)
            record["_source"]     = page_data.get("_source", "layoutlm")
            records.append(record)

    return records