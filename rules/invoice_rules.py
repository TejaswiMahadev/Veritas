import re
from typing import Dict, List, Any

ALL_FLAGS = [
    {"id": "INV_AMOUNT_MISMATCH", "severity": "critical", "description": "Invoice total does not match line items or taxable+tax."},
    {"id": "INV_ZERO_NEGATIVE_PRICING", "severity": "high", "description": "Zero or negative price found in line items."},
    {"id": "INV_GSTIN_FORMAT", "severity": "critical", "description": "Vendor or buyer GSTIN format invalid."},
    {"id": "INV_PAN_MISMATCH", "severity": "high", "description": "PAN extracted from GSTIN does not match provided PAN."},
    {"id": "INV_MISSING_SIGNATURE", "severity": "critical", "description": "Missing digital signature or seal."},
    {"id": "INV_ROUNDING_ANOMALY", "severity": "medium", "description": "Rounding inconsistencies detected."},
]

GSTIN_REGEX = re.compile(r'^\d{2}[A-Z0-9]{10}\d[A-Z][A-Z0-9]$')


def _sum_line_items(line_items: List[Dict[str, Any]]) -> float:
    s = 0.0
    for li in line_items:
        val = li.get('total_amount') or li.get('total') or li.get('amount') or 0
        try:
            s += float(val)
        except Exception:
            continue
    return s


def detect_flags(document: Dict[str, Any]) -> List[Dict[str, Any]]:

    flags = []
    doc = document.get('document', {})
    amounts = doc.get('amounts', {})
    line_items = doc.get('line_items', []) or []

    total = amounts.get('total_amount') or 0.0
    taxable = amounts.get('taxable_amount') or 0.0
    tax = amounts.get('tax_amount') or 0.0
    line_sum = _sum_line_items(line_items)
    if abs(line_sum - float(total)) > 1.0:
        flags.append({
            "id": "INV_AMOUNT_MISMATCH",
            "description": f"Line items sum {line_sum} != total {total}",
            "severity": "critical",
            "evidence": {"line_sum": line_sum, "total": total}
        })
    try:
        if abs((float(taxable) + float(tax)) - float(total)) > 1.0:
            flags.append({
                "id": "INV_AMOUNT_MISMATCH",
                "description": f"Taxable {taxable} + tax {tax} != total {total}",
                "severity": "critical",
                "evidence": {"taxable": taxable, "tax": tax, "total": total}
            })
    except Exception:
        pass
    for idx, li in enumerate(line_items):
        val = li.get('unit_price') or li.get('total_amount') or 0
        try:
            if float(val) <= 0:
                flags.append({
                    "id": "INV_ZERO_NEGATIVE_PRICING",
                    "description": f"Line item {idx} has zero/negative price: {val}",
                    "severity": "high",
                    "evidence": {"line_item": li}
                })
        except Exception:
            continue

    vendor_gstin = doc.get('vendor', {}).get('gstin')
    buyer_gstin = doc.get('buyer', {}).get('gstin')
    for role, gst in (('vendor', vendor_gstin), ('buyer', buyer_gstin)):
        if gst:
            if not GSTIN_REGEX.match(str(gst)):
                flags.append({
                    "id": "INV_GSTIN_FORMAT",
                    "description": f"{role} GSTIN invalid: {gst}",
                    "severity": "critical",
                    "evidence": {f"{role}_gstin": gst}
                })

    vendor_pan = doc.get('vendor', {}).get('pan')
    if vendor_gstin and vendor_pan:
        extracted_pan = None
        try:
            extracted_pan = str(vendor_gstin)[2:12]
        except Exception:
            extracted_pan = None
        if extracted_pan and vendor_pan and extracted_pan != vendor_pan:
            flags.append({
                "id": "INV_PAN_MISMATCH",
                "description": f"Vendor PAN {vendor_pan} != extracted from GSTIN {extracted_pan}",
                "severity": "high",
                "evidence": {"vendor_pan": vendor_pan, "extracted_pan": extracted_pan}
            })

    auth = doc.get('authentication', {})
    has_sig = auth.get('has_digital_signature') or auth.get('has_authentication')
    has_seal = auth.get('has_seal')
    if not has_sig and not has_seal:
        flags.append({
            "id": "INV_MISSING_SIGNATURE",
            "description": "No digital signature or physical seal present.",
            "severity": "critical",
            "evidence": {"authentication": auth}
        })

    frac = abs(float(total) - round(float(total)))
    if frac > 0 and frac < 0.5:
        flags.append({
            "id": "INV_ROUNDING_ANOMALY",
            "description": f"Unusual fractional part in total: {frac}",
            "severity": "medium",
            "evidence": {"total": total}
        })

    return flags
