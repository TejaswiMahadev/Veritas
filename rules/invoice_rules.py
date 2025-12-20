import re
from typing import Dict, List, Any
from datetime import datetime

GSTIN_REGEX = re.compile(r'^\d{2}[A-Z0-9]{10}\d[A-Z][A-Z0-9]$')
DATE_FMT = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"]

# Define possible flags for INVOICE ruleset
ALL_FLAGS = [
    {"id": "INV_AMOUNT_MISMATCH", "severity": "critical", "description": "Invoice total does not match line items or taxable+tax."},
    {"id": "INV_ZERO_NEGATIVE_PRICING", "severity": "high", "description": "Zero or negative price found in line items."},
    {"id": "INV_GSTIN_FORMAT", "severity": "critical", "description": "Vendor or buyer GSTIN format invalid."},
    {"id": "INV_PAN_MISMATCH", "severity": "high", "description": "PAN extracted from GSTIN does not match provided PAN."},
    {"id": "INV_MISSING_SIGNATURE", "severity": "critical", "description": "Missing digital signature or seal."},
    {"id": "INV_ROUNDING_ANOMALY", "severity": "medium", "description": "Rounding inconsistencies detected."},
    {"id": "INV_MISSING_FIELD", "severity": "critical", "description": "Critical field missing (Invoice Number, Date, or Total)."},
    {"id": "INV_FUTURE_DATE", "severity": "high", "description": "Invoice date is in the future."},
    {"id": "INV_DUE_DATE_MISMATCH", "severity": "medium", "description": "Due date is earlier than invoice date."},
    {"id": "INV_TAX_RATE_ANOMALY", "severity": "medium", "description": " implied tax rate does not match standard slabs (5, 12, 18, 28%)."},
    {"id": "INV_IRN_MISSING", "severity": "critical", "description": "IRN missing for vendor > 5 Cr turnover."},
    {"id": "INV_IRP_UPLOAD_DELAY", "severity": "critical", "description": "Invoice uploaded to IRP > 30 days after generation."},
    {"id": "INV_GOV_SUPPLY_NONCOMPLIANCE", "severity": "critical", "description": "Government supply > 5 Cr missing IRN."},
    {"id": "INV_EXEMPT_MISMATCH", "severity": "medium", "description": "Exempt vendor issuing IRN or non-exempt missing IRN."},
]


def _sum_line_items(line_items: List[Dict[str, Any]]) -> float:
    s = 0.0
    for li in line_items:
        # Accept different field names for total
        val = li.get('total_amount') or li.get('total') or li.get('amount') or 0
        try:
            s += float(val)
        except Exception:
            continue
    return s


def detect_flags(document: Dict[str, Any], vendor_profile: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Return list of triggered flags with evidence."""
    flags = []
    doc = document.get('document', {})
    amounts = doc.get('amounts', {})
    line_items = doc.get('line_items', []) or []

    total = amounts.get('total_amount') or 0.0
    taxable = amounts.get('taxable_amount') or 0.0
    tax = amounts.get('tax_amount') or 0.0

    # 1. Line item sum vs invoice total
    line_sum = _sum_line_items(line_items)
    if abs(line_sum - float(total)) > 1.0:
        flags.append({
            "id": "INV_AMOUNT_MISMATCH",
            "description": f"Line items sum {line_sum} != total {total}",
            "severity": "critical",
            "evidence": {"line_sum": line_sum, "total": total}
        })

    # 2. taxable + tax == total
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

    # 3. Zero or negative pricing
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

    # 4. GSTIN format validation
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

    # 5. PAN vs GSTIN extraction (if PAN present in document)
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

    # 6. Missing signature/seal
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

    # 7. Rounding anomalies (tiny heuristic)
    frac = abs(float(total) - round(float(total)))
    if frac > 0 and frac < 0.5:
        flags.append({
            "id": "INV_ROUNDING_ANOMALY",
            "description": f"Unusual fractional part in total: {frac}",
            "severity": "medium",
            "evidence": {"total": total}
        })

    # 8. Critical field check
    crit_fields = []
    if not doc.get('invoice_number'): crit_fields.append('invoice_number')
    if not doc.get('invoice_date'): crit_fields.append('invoice_date')
    if not amounts.get('total_amount'): crit_fields.append('total_amount')
    
    if crit_fields:
        flags.append({
            "id": "INV_MISSING_FIELD",
            "description": f"Missing critical fields: {', '.join(crit_fields)}",
            "severity": "critical",
            "evidence": {"missing": crit_fields}
        })

    # 9. Date validation
    def _parse_date(ds):
        for fmt in DATE_FMT:
            try:
                return datetime.strptime(ds, fmt)
            except:
                pass
        return None

    inv_date_str = doc.get('invoice_date')
    due_date_str = doc.get('due_date')
    inv_date = _parse_date(inv_date_str) if inv_date_str else None
    due_date = _parse_date(due_date_str) if due_date_str else None

    if inv_date:
        if inv_date > datetime.now():
             flags.append({
                "id": "INV_FUTURE_DATE",
                "description": f"Invoice date {inv_date_str} is in the future.",
                "severity": "high",
                "evidence": {"invoice_date": inv_date_str}
            })

    if inv_date and due_date:
        if due_date < inv_date:
             flags.append({
                "id": "INV_DUE_DATE_MISMATCH",
                "description": f"Due date {due_date_str} is before invoice date {inv_date_str}",
                "severity": "medium",
                "evidence": {"invoice_date": inv_date_str, "due_date": due_date_str}
            })

    # 10. Tax Rate Anomaly
    # Standard India GST rates: 5, 12, 18, 28
    if taxable > 0 and tax > 0:
        implied_rate = (float(tax) / float(taxable)) * 100
        # Check closest standard slab
        slabs = [5, 12, 18, 28]
        is_standard = False
        for slab in slabs:
            if abs(implied_rate - slab) < 1.0: # 1% tolerance
                is_standard = True
                break
        
        if not is_standard:
             flags.append({
                "id": "INV_TAX_RATE_ANOMALY",
                "description": f"Implied tax rate {implied_rate:.2f}% is not standard (5, 12, 18, 28).",
                "severity": "medium",
                "evidence": {"taxable": taxable, "tax": tax, "implied_rate": implied_rate}
            })

    # 11. E-Invoicing Compliance (Gov Mandates)
    if vendor_profile:
        aato = vendor_profile.get('aato', 0.0)
        category = vendor_profile.get('category', 'Regular')
        is_gov = vendor_profile.get('is_gov_supplier', False)
        
        irn = doc.get('irn') or doc.get('invoice_reference_number')
        
        # Exempt categories (Banking, GTA, Insurance, Passenger Transport, Cinema)
        exempt_cats = ["Banking", "GTA", "Insurance", "Passenger Transport", "Cinema"]
        is_exempt = category in exempt_cats
        
        # Rule: IRN Missing for > 5 Cr (Non-Exempt)
        if aato > 50000000 and not is_exempt and not irn:
            flags.append({
                "id": "INV_IRN_MISSING",
                "description": "Mandatory IRN missing for vendor with AATO > 5 Cr.",
                "severity": "critical",
                "evidence": {"aato": aato, "category": category}
            })

        # Rule: Exempt Mismatch
        if is_exempt and irn:
             flags.append({
                "id": "INV_EXEMPT_MISMATCH",
                "description": f"Exempt category '{category}' vendor issued IRN (Unusual).",
                "severity": "medium",
                "evidence": {"category": category, "irn_present": True}
            })
            
        # Rule: Gov Supply Non-Compliance
        # If vendor is gov supplier > 5 Cr, they MUST issue E-Invoice
        if is_gov and aato > 50000000 and not irn:
             # This is practically covered by INV_IRN_MISSING but good to be specific for gov context
             flags.append({
                "id": "INV_GOV_SUPPLY_NONCOMPLIANCE",
                "description": "Government supplier > 5 Cr missing mandatory IRN.",
                "severity": "critical",
                "evidence": {"is_gov_supplier": True, "aato": aato}
            })

        # Rule: IRP Upload Delay (30 days) - for > 10 Cr AATO
        # We need IRP upload date. In absence, usually embedded in QR or we check current date vs inv date?
        # Requirement: "block or flag invoices uploaded after this window"
        # Since we are auditing, we compare 'current processing time' or 'upload_date' metadata if available.
        # Assuming we check Invoice Date vs Current Date for now if not provided.
        # Ideally we extract 'AckDate' from IRN details, but let's use current time as proxy for 'Attempted Upload' audit
        if aato >= 100000000 and irn and inv_date: 
             # Check if invoice is already > 30 days old from NOW (assuming we are validating upon upload)
             # NOTE: This logic assumes 'verification time' is close to 'upload to IRP'. 
             # If this is historical audit, this logic might flag valid old invoices. 
             # Adjusting to strict interpretation: invoice_date vs Ack Date.
             # Since we don't have Ack Date parsed, let's skip strict check or add placeholder.
             # Placeholder logic: If invoice is > 30 days old and we ARE just seeing it, flag it.
             limit_days = 30
             delta = datetime.now() - inv_date
             if delta.days > limit_days:
                 flags.append({
                    "id": "INV_IRP_UPLOAD_DELAY",
                    "description": f"Invoice date {inv_date_str} is > 30 days old. Ensure IRP upload was within window.",
                    "severity": "critical",
                    "evidence": {"invoice_date": inv_date_str, "age_days": delta.days}
                })

    return flags

