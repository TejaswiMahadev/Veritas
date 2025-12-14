import re
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class NormalizedInvoice:
    document_id: str
    vendor_name: Optional[str] = None
    vendor_gstin: Optional[str] = None
    vendor_pan: Optional[str] = None
    vendor_address: Optional[str] = None
    
    buyer_name: Optional[str] = None
    buyer_gstin: Optional[str] = None
    buyer_department: Optional[str] = None
    
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    po_number: Optional[str] = None
    contract_reference: Optional[str] = None
    
    # Financial data
    total_amount: float = 0.0
    taxable_amount: float = 0.0
    tax_amount: float = 0.0
    currency: str = "INR"
    
    # Line items
    line_items: List[Dict] = None
    
    # Payment terms
    payment_terms: Optional[str] = None
    payment_due_date: Optional[str] = None
    payment_method: Optional[str] = None
    
    # Validation flags
    has_authentication: bool = False
    has_digital_signature: bool = False
    has_seal: bool = False
    
    # Metadata
    extraction_warnings: List[str] = None
    normalization_notes: List[str] = None
    validation_score: float = 0.0
    
    # AI Validation results
    ai_status: Optional[str] = None
    ai_errors: List[str] = None
    consistency_checks: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.line_items is None:
            self.line_items = []
        if self.extraction_warnings is None:
            self.extraction_warnings = []
        if self.normalization_notes is None:
            self.normalization_notes = []
        if self.ai_errors is None:
            self.ai_errors = []
        if self.consistency_checks is None:
            self.consistency_checks = {}


INVOICE_SCHEMA = {
    "required_fields": [
        "vendor_name", "vendor_gstin", "buyer_name", 
        "invoice_number", "invoice_date", "total_amount"
    ],
    "validation_rules": {
        "vendor_gstin": r'^\d{2}[A-Z0-9]{10}\d[A-Z][A-Z0-9]$',
        "buyer_gstin": r'^\d{2}[A-Z0-9]{10}\d[A-Z][A-Z0-9]$',
        "vendor_pan": r'^[A-Z]{5}\d{4}[A-Z]$',
    },
    "consistency_checks": [
        "total_amount = taxable_amount + tax_amount (±1 tolerance)",
        "payment_due_date > invoice_date",
        "sum(line_items) = total_amount"
    ]
}


class DataNormalizer:
    """STEP 2, 3, 4, 5: Normalize text, dates, numbers, booleans"""
    
    @staticmethod
    def normalize_text(text: Any) -> str:
        """STEP 2: Normalize text fields"""
        if not text:
            return ""
        
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'["\']', '', text)
        
        return text
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """STEP 2: Normalize entity names"""
        if not name:
            return ""
        
        name = DataNormalizer.normalize_text(name)
        
        replacements = {
            r'\bPVT\.?\s*LTD\.?\b': 'PVT LTD',
            r'\bLTD\.?\b': 'LTD',
            r'\bCO\.?\b': 'CO',
            r'\bINC\.?\b': 'INC',
        }
        
        for pattern, replacement in replacements.items():
            name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
        
        return name.upper()
    
    @staticmethod
    def normalize_date(date_str: Any) -> Optional[str]:
        """STEP 3: Normalize dates to ISO format"""
        if not date_str:
            return None
        
        if isinstance(date_str, str):
            formats = [
                '%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%d/%m/%y',
                '%d-%b-%Y', '%d %B %Y', '%B %d, %Y', '%d.%m.%Y'
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str.strip(), fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
        
        return str(date_str) if date_str else None
    
    @staticmethod
    def clean_amount(amount_str: Any) -> float:
        """STEP 4: Normalize numbers and currency"""
        if amount_str is None:
            return 0.0
        
        if isinstance(amount_str, (int, float)):
            return float(amount_str)
        
        cleaned = str(amount_str)
        cleaned = re.sub(r'[₹$,\s/-]', '', cleaned)
        
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    
    @staticmethod
    def normalize_bool(value: Any) -> bool:
        """STEP 5: Boolean normalization"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            return v in ["true", "yes", "1", "present"]
        if isinstance(value, int):
            return value == 1
        return False
    
    @staticmethod
    def normalize_gstin(gstin: str) -> Optional[str]:
        """Normalize GSTIN format"""
        if not gstin:
            return None
        
        gstin = re.sub(r'\s+', '', str(gstin)).upper()
        
        if re.match(r'^\d{2}[A-Z0-9]{10}\d[A-Z][A-Z0-9]$', gstin):
            return gstin
        
        return gstin if gstin else None
    
    @staticmethod
    def extract_pan_from_gstin(gstin: str) -> Optional[str]:
        """Extract PAN from GSTIN"""
        if gstin and len(gstin) >= 12:
            normalized = DataNormalizer.normalize_gstin(gstin)
            if normalized:
                return normalized[2:12]
        return None


class GeminiValidator:
    """AI-powered validation and missing field reconstruction"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash"
    
    def detect_date_format(self, date_str: str, context: str) -> Optional[str]:
        """STEP 3: Use Gemini to detect ambiguous date formats"""
        prompt = f"""Analyze this date string and return ONLY the detected format pattern.

Date: "{date_str}"
Context: {context}

Return ONLY one of these formats:
- %d-%m-%Y
- %m-%d-%Y
- %Y-%m-%d
- %d/%m/%Y
- %m/%d/%Y

Return format pattern only, nothing else."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            return response.text.strip()
        except:
            return None
    
    def infer_missing_fields(self, doc_data: Dict, raw_text: str, missing_fields: List[str]) -> Dict[str, Any]:
        """STEP 6: Reconstruct missing fields using Gemini"""
        prompt = f"""You are a document analysis AI. Extract missing invoice fields from the raw text.

MISSING FIELDS: {', '.join(missing_fields)}

CURRENT DATA:
{json.dumps(doc_data, indent=2, default=str)}

RAW TEXT:
{raw_text[:2000]}

Extract the missing fields and return ONLY a JSON object with the inferred values.
Format: {{"field_name": "value", ...}}

Be precise. Return ONLY JSON, no explanation."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            return {}
    
    def validate_consistency(self, doc: NormalizedInvoice) -> Dict[str, Any]:
        """STEP 7: Internal consistency checks"""
        doc_dict = asdict(doc)
        
        prompt = f"""Validate this INVOICE document for consistency issues.

DOCUMENT DATA:
{json.dumps(doc_dict, indent=2, default=str)}

VALIDATION RULES FOR INVOICE:
{json.dumps(INVOICE_SCHEMA, indent=2)}

Check:
1. Financial calculations (total = taxable + tax)
2. Date logic (due date > invoice date)
3. GSTIN format validation
4. Line items sum = total
5. Missing required fields
6. Business rule violations

Return JSON:
{{
    "status": "VALID" | "INVALID" | "WARNING",
    "score": 0.0-1.0,
    "errors": ["list of errors"],
    "warnings": ["list of warnings"],
    "checks": {{
        "financial_consistency": true/false,
        "date_logic": true/false,
        "format_validation": true/false,
        "completeness": true/false
    }}
}}

Return ONLY JSON."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            return {
                "status": "ERROR",
                "score": 0.0,
                "errors": [f"Gemini validation error: {str(e)}"],
                "warnings": [],
                "checks": {}
            }


class InvoiceNormalizer:
    """Main Invoice Normalizer - follows all 11 steps"""
    
    def __init__(self, gemini_api_key: Optional[str] = None, use_gemini: str = "when_needed"):
        """
        Args:
            use_gemini: "always" | "when_needed" | "never"
        """
        self.normalizer = DataNormalizer()
        self.use_gemini = use_gemini
        
        if use_gemini != "never":
            try:
                self.gemini = GeminiValidator(gemini_api_key)
                print(f"✅ Gemini AI enabled (mode: {use_gemini})")
            except Exception as e:
                print(f"⚠️  Gemini disabled: {e}")
                self.use_gemini = "never"
    
    def process_invoice(self, raw_invoice: Dict) -> NormalizedInvoice:
        """Main processing pipeline - executes all 11 steps"""
        
        # STEP 1: Input & Schema Validation
        validation_errors = self._step1_validate_input(raw_invoice)
        
        # STEP 2-5: Normalize all fields
        normalized = self._step2to5_normalize_fields(raw_invoice)
        
        # STEP 6: Missing field reconstruction
        if self._should_use_gemini(normalized):
            self._step6_reconstruct_missing_fields(raw_invoice, normalized)
        
        # STEP 7: Internal consistency checks
        self._step7_consistency_checks(normalized)
        
        # STEP 8: Generate validation score
        normalized.validation_score = self._step8_calculate_score(normalized)
        
        # STEP 9: Create normalization log
        self._step9_create_log(normalized)
        
        # Add initial validation errors
        normalized.extraction_warnings.extend(validation_errors)
        
        return normalized
    
    def _step1_validate_input(self, doc: Dict) -> List[str]:
        """STEP 1: Input & Schema Validation"""
        errors = []
        
        # Check JSON validity
        if not isinstance(doc, dict):
            errors.append("CRITICAL: Invalid JSON structure")
            return errors
        
        # Check required fields based on invoice schema
        required = INVOICE_SCHEMA['required_fields']
        
        for field in required:
            if not self._field_exists(doc, field):
                errors.append(f"Missing required field: {field}")
        
        return errors
    
    def _field_exists(self, doc: Dict, field: str) -> bool:
        """Check if field exists and is not null/empty"""
        field_map = {
            'vendor_name': ['vendor', 'name', 'value'],
            'vendor_gstin': ['vendor', 'gstin', 'value'],
            'buyer_name': ['buyer', 'name', 'value'],
            'invoice_number': ['invoice_number', 'value'],
            'invoice_date': ['invoice_date', 'value'],
            'total_amount': ['total_amount', 'value'],
        }
        
        path = field_map.get(field, [field])
        value = doc
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                return False
        
        return bool(value)
    
    def _step2to5_normalize_fields(self, doc: Dict) -> NormalizedInvoice:
        """STEPS 2-5: Normalize text, dates, numbers, booleans"""
        
        # Extract vendor info - handle None safely
        vendor = doc.get('vendor')
        if vendor and isinstance(vendor, dict):
            vendor_name = vendor.get('name', {}).get('value') if isinstance(vendor.get('name'), dict) else None
            vendor_gstin = vendor.get('gstin', {}).get('value') if isinstance(vendor.get('gstin'), dict) else None
            vendor_pan = vendor.get('pan', {}).get('value') if isinstance(vendor.get('pan'), dict) else None
            vendor_address = vendor.get('address', {}).get('value') if isinstance(vendor.get('address'), dict) else None
        else:
            vendor_name = vendor_gstin = vendor_pan = vendor_address = None
        
        # Extract buyer info - handle None safely
        buyer = doc.get('buyer')
        if buyer and isinstance(buyer, dict):
            buyer_name = buyer.get('name', {}).get('value') if isinstance(buyer.get('name'), dict) else None
            buyer_gstin = buyer.get('gstin', {}).get('value') if isinstance(buyer.get('gstin'), dict) else None
            buyer_dept = buyer.get('department', {}).get('value') if isinstance(buyer.get('department'), dict) else None
        else:
            buyer_name = buyer_gstin = buyer_dept = None
        
        # Helper function to safely get value from ExtractedField dict
        def get_value(field):
            if field and isinstance(field, dict):
                return field.get('value')
            return None
        
        # Financial data (STEP 4)
        total_amount = self.normalizer.clean_amount(get_value(doc.get('total_amount')))
        taxable_amount = self.normalizer.clean_amount(get_value(doc.get('taxable_amount')))
        
        tax_breakdown = doc.get('tax_breakdown')
        tax_amount = 0.0
        if tax_breakdown and isinstance(tax_breakdown, dict):
            tax_amount = (
                self.normalizer.clean_amount(tax_breakdown.get('cgst', 0)) +
                self.normalizer.clean_amount(tax_breakdown.get('sgst', 0)) +
                self.normalizer.clean_amount(tax_breakdown.get('igst', 0))
            )
        
        # Dates (STEP 3)
        invoice_date = self.normalizer.normalize_date(get_value(doc.get('invoice_date')))
        payment_due_date = self.normalizer.normalize_date(get_value(doc.get('payment_due_date')))
        
        # Document references
        invoice_num = get_value(doc.get('invoice_number'))
        po_num = get_value(doc.get('po_number'))
        contract_ref = get_value(doc.get('contract_reference'))
        
        # Authentication (STEP 5: Boolean)
        auth = doc.get('authentication')
        if auth and isinstance(auth, dict):
            has_digital_sig = self.normalizer.normalize_bool(auth.get('has_digital_signature', False))
            has_seal = self.normalizer.normalize_bool(auth.get('has_physical_seal', False))
            has_stamp = self.normalizer.normalize_bool(auth.get('has_stamp', False))
        else:
            has_digital_sig = has_seal = has_stamp = False
        has_auth = any([has_digital_sig, has_seal, has_stamp])
        
        # Payment info
        payment_terms = get_value(doc.get('payment_terms'))
        payment_method = get_value(doc.get('payment_method'))
        
        return NormalizedInvoice(
            document_id=doc.get('document_id', 'UNKNOWN'),
            vendor_name=self.normalizer.normalize_name(vendor_name) if vendor_name else None,
            vendor_gstin=self.normalizer.normalize_gstin(vendor_gstin),
            vendor_pan=vendor_pan,
            vendor_address=self.normalizer.normalize_text(vendor_address),
            buyer_name=self.normalizer.normalize_name(buyer_name) if buyer_name else None,
            buyer_gstin=self.normalizer.normalize_gstin(buyer_gstin),
            buyer_department=buyer_dept,
            invoice_number=invoice_num,
            invoice_date=invoice_date,
            po_number=po_num,
            contract_reference=contract_ref,
            total_amount=total_amount,
            taxable_amount=taxable_amount,
            tax_amount=tax_amount,
            line_items=doc.get('line_items', []),
            payment_terms=payment_terms,
            payment_method=payment_method,
            payment_due_date=payment_due_date,
            has_authentication=has_auth,
            has_digital_signature=has_digital_sig,
            has_seal=has_seal,
            extraction_warnings=doc.get('extraction_warnings', []).copy()
        )
    
    def _should_use_gemini(self, doc: NormalizedInvoice) -> bool:
        """Determine if Gemini should be used"""
        if self.use_gemini == "never":
            return False
        if self.use_gemini == "always":
            return True
        
        # "when_needed" logic
        missing_count = sum([
            not doc.vendor_name,
            not doc.buyer_name,
            not doc.invoice_date,
            doc.total_amount == 0,
        ])
        
        has_inconsistencies = (
            abs((doc.taxable_amount + doc.tax_amount) - doc.total_amount) > 1
            if doc.total_amount > 0 else False
        )
        
        return missing_count > 0 or has_inconsistencies
    
    def _step6_reconstruct_missing_fields(self, raw_doc: Dict, normalized: NormalizedInvoice):
        """STEP 6: Missing field reconstruction using Gemini"""
        missing_fields = []
        
        if not normalized.vendor_name:
            missing_fields.append('vendor_name')
        if not normalized.buyer_department:
            missing_fields.append('buyer_department')
        if not normalized.invoice_date:
            missing_fields.append('invoice_date')
        if not normalized.payment_due_date:
            missing_fields.append('payment_due_date')
        if normalized.total_amount == 0:
            missing_fields.append('total_amount')
        
        if not missing_fields:
            return
        
        raw_text = raw_doc.get('raw_text', '')
        if not raw_text:
            return
        
        try:
            inferred = self.gemini.infer_missing_fields(
                asdict(normalized),
                raw_text,
                missing_fields
            )
            
            for field, value in inferred.items():
                if hasattr(normalized, field) and value:
                    setattr(normalized, field, value)
                    normalized.normalization_notes.append(f"Gemini inferred {field}: {value}")
        except Exception as e:
            normalized.normalization_notes.append(f"Gemini inference failed: {str(e)}")
    
    def _step7_consistency_checks(self, doc: NormalizedInvoice):
        """STEP 7: Internal consistency checks"""
        
        if self.use_gemini != "never":
            try:
                result = self.gemini.validate_consistency(doc)
                doc.ai_status = result.get('status', 'UNKNOWN')
                doc.ai_errors = result.get('errors', [])
                doc.consistency_checks = result.get('checks', {})
                
                if result.get('warnings'):
                    doc.extraction_warnings.extend(result['warnings'])
            except Exception as e:
                doc.ai_status = "ERROR"
                doc.ai_errors = [f"Validation error: {str(e)}"]
        
        # Manual checks
        # Check total = taxable + tax
        calculated_total = doc.taxable_amount + doc.tax_amount
        if abs(calculated_total - doc.total_amount) > 1:
            doc.extraction_warnings.append(
                f"Amount mismatch: Total={doc.total_amount}, "
                f"Calculated={calculated_total}"
            )
        
        # Check dates
        if doc.invoice_date and doc.payment_due_date:
            if doc.payment_due_date < doc.invoice_date:
                doc.extraction_warnings.append(
                    f"Due date ({doc.payment_due_date}) before invoice date ({doc.invoice_date})"
                )
        
        # Check GSTIN
        if doc.vendor_gstin:
            pattern = r'^\d{2}[A-Z0-9]{10}\d[A-Z][A-Z0-9]$'
            if not re.match(pattern, doc.vendor_gstin):
                doc.extraction_warnings.append(f"Invalid GSTIN format: {doc.vendor_gstin}")
        
        # Extract PAN if missing
        if not doc.vendor_pan and doc.vendor_gstin:
            doc.vendor_pan = self.normalizer.extract_pan_from_gstin(doc.vendor_gstin)
            if doc.vendor_pan:
                doc.normalization_notes.append("Extracted PAN from GSTIN")
    
    def _step8_calculate_score(self, doc: NormalizedInvoice) -> float:
        """STEP 8: Generate validation score"""
        scores = []
        
        # Completeness (40%)
        if doc.vendor_name:
            scores.append(0.10)
        if doc.buyer_name:
            scores.append(0.10)
        if doc.invoice_number:
            scores.append(0.10)
        if doc.invoice_date:
            scores.append(0.10)
        
        # Financial (30%)
        if doc.total_amount > 0:
            scores.append(0.15)
        if doc.taxable_amount > 0:
            scores.append(0.15)
        
        # Authentication (15%)
        if doc.has_authentication:
            scores.append(0.15)
        
        # Validation (15%)
        if doc.vendor_gstin:
            pattern = r'^\d{2}[A-Z0-9]{10}\d[A-Z][A-Z0-9]$'
            if re.match(pattern, doc.vendor_gstin):
                scores.append(0.10)
        
        critical_errors = [w for w in doc.extraction_warnings if 'CRITICAL' in w.upper()]
        if not critical_errors:
            scores.append(0.05)
        
        return min(sum(scores), 1.0)
    
    def _step9_create_log(self, doc: NormalizedInvoice):
        """STEP 9: Create normalization log"""
        if not doc.normalization_notes:
            doc.normalization_notes = []
        
        # Add automatic notes
        if doc.vendor_pan and doc.vendor_gstin:
            if doc.vendor_pan == self.normalizer.extract_pan_from_gstin(doc.vendor_gstin):
                doc.normalization_notes.append("PAN validated against GSTIN")
        
        if doc.invoice_date and doc.payment_due_date:
            try:
                inv_dt = datetime.strptime(doc.invoice_date, '%Y-%m-%d')
                due_dt = datetime.strptime(doc.payment_due_date, '%Y-%m-%d')
                days = (due_dt - inv_dt).days
                if not doc.payment_terms:
                    doc.payment_terms = f"Net {days} days"
                    doc.normalization_notes.append(f"Inferred payment terms: Net {days} days")
            except:
                pass
    
    def get_final_output(self, doc: NormalizedInvoice) -> Dict[str, Any]:
        """Generate final structured output"""
        
        structured = {
            "document_id": doc.document_id,
            "vendor": {
                "name": doc.vendor_name,
                "gstin": doc.vendor_gstin,
                "pan": doc.vendor_pan,
                "address": doc.vendor_address
            },
            "buyer": {
                "name": doc.buyer_name,
                "gstin": doc.buyer_gstin,
                "department": doc.buyer_department
            },
            "invoice_details": {
                "invoice_number": doc.invoice_number,
                "invoice_date": doc.invoice_date,
                "po_number": doc.po_number,
                "contract_reference": doc.contract_reference
            },
            "amounts": {
                "total_amount": doc.total_amount,
                "taxable_amount": doc.taxable_amount,
                "tax_amount": doc.tax_amount,
                "currency": doc.currency
            },
            "line_items": doc.line_items,
            "payment": {
                "terms": doc.payment_terms,
                "method": doc.payment_method,
                "due_date": doc.payment_due_date
            },
            "authentication": {
                "has_authentication": doc.has_authentication,
                "has_digital_signature": doc.has_digital_signature,
                "has_seal": doc.has_seal
            }
        }
        
        validation_info = {
            "score": doc.validation_score,
            "warnings": doc.extraction_warnings,
            "normalization_notes": doc.normalization_notes,
            "overall_status": self._get_overall_status(doc)
        }

        ai_validation = {
            "status": doc.ai_status,
            "errors": doc.ai_errors,
            "checks": doc.consistency_checks
        }

        return {
            "document_type": "INVOICE",
            "invoice": structured,
            "validation": validation_info,
            "ai_validation": ai_validation
        }

    def _get_overall_status(self, doc: NormalizedInvoice) -> str:
        """Determine overall document status"""
        if doc.ai_status == "INVALID" or doc.validation_score < 0.5:
            return "REJECTED"
        elif doc.ai_status == "WARNING" or doc.validation_score < 0.8:
            return "NEEDS_REVIEW"
        else:
            return "APPROVED"


# Demo
if __name__ == "__main__":
    print("=" * 80)
    print("INVOICE NORMALIZATION & VALIDATION PIPELINE")
    print("=" * 80)
    
    # Choose Gemini mode
    api_key = os.getenv('GOOGLE_API_KEY')
    gemini_mode = "when_needed"  # or "always" or "never"
    
    if api_key:
        print(f"✅ Gemini mode: {gemini_mode}")
        normalizer = InvoiceNormalizer(use_gemini=gemini_mode)
    else:
        print("⚠️  Gemini disabled (no API key)")
        normalizer = InvoiceNormalizer(use_gemini="never")
    
    # Sample invoice
    sample_invoice = {
        "document_id": "INV-2025-001",
        "vendor": {
            "name": {"value": "Techware Solutions Pvt Ltd", "confidence": 0.95},
            "gstin": {"value": "29AABCT2345F1Z3", "confidence": 0.9},
            "address": {"value": "14 MG Road, Bengaluru", "confidence": 0.8}
        },
        "buyer": {
            "name": {"value": "Department of Health – Karnataka", "confidence": 0.9},
            "gstin": {"value": "29GOVT5678A1Z7", "confidence": 0.9}
        },
        "invoice_number": {"value": "INV-7782"},
        "invoice_date": {"value": "28/11/2025"},
        "total_amount": {"value": "₹11,800/-"},
        "taxable_amount": {"value": "10000"},
        "tax_breakdown": {"cgst": 900.0, "sgst": 900.0, "igst": 0.0},
        "line_items": [
            {
                "description": "Printer Ink Cartridge",
                "quantity": 5,
                "unit_price": 2000,
                "total_amount": 10000,
                "cgst_amount": 900,
                "sgst_amount": 900
            }
        ],
        "payment_due_date": {"value": "15-12-2025"},
        "payment_method": {"value": "NEFT"},
        "authentication": {
            "has_digital_signature": "true",
            "has_physical_seal": "false",
            "has_stamp": "yes"
        },
        "raw_text": "INVOICE\nTechware Solutions Pvt Ltd\n29AABCT2345F1Z3\nInvoice No: INV-7782\nDate: 28/11/2025\nTo: Department of Health Karnataka\nAmount: Rs 11,800/-",
        "extraction_warnings": []
    }
    
    print("\n" + "=" * 80)
    print(f"PROCESSING INVOICE - {sample_invoice['document_id']}")
    print("=" * 80)
    
    # Process through pipeline
    normalized = normalizer.process_invoice(sample_invoice)
    
    # Display results
    print(f"\n📄 Invoice: {normalized.document_id}")
    print(f"🏢 Vendor: {normalized.vendor_name or 'N/A'}")
    print(f"🏛️  Buyer: {normalized.buyer_name or 'N/A'}")
    print(f"💰 Total Amount: {normalized.total_amount or 'N/A'}")
    print(f"📅 Invoice Date: {normalized.invoice_date or 'N/A'}")
    print(f"📅 Payment Due Date: {normalized.payment_due_date or 'N/A'}")
    print(f"⚖️  Validation Status: {normalized.validation_score}")
    print(f"⚖️  AI Validation Status: {normalized.ai_status}")
    print(f"⚖️  Overall Status: {normalizer._get_overall_status(normalized)}")
    print("=" * 80)
    
    # Print structured output
    print("\nSTRUCTURED OUTPUT")
    print("=" * 80)
    final_result = normalizer.get_final_output(normalized)

    print(json.dumps(final_result["invoice"], indent=4))
    
    # Print validation info
    print("\nVALIDATION INFO")
    print("=" * 80)
    print(json.dumps(final_result["validation"], indent=4))
    
    # Print AI validation info
    print("\nAI VALIDATION INFO")
    print("=" * 80)
    print(json.dumps(final_result["ai_validation"], indent=4))