import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from google import genai
from google.genai import types

try:
    from cache_manager import get_cache_manager
    CACHE_ENABLED = True
except ImportError:
    CACHE_ENABLED = False
    print("Cache manager not available - all requests will hit API")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("pdfplumber not installed - fallback text extraction unavailable")

try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("pytesseract/PIL/pdf2image not installed - fallback OCR unavailable")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBCXzmSRgvYJENKIQPORG1ib_At5qzSncs")
USE_VERTEX = os.getenv('USE_VERTEX_AI', 'false').lower() == 'true'

try:
    if USE_VERTEX:
        print("Initializing Vertex AI Client...")
        client = genai.Client(
            vertexai=True,
            project=os.getenv('GCP_PROJECT_ID'),
            location=os.getenv('GCP_LOCATION', 'us-central1')
        )
    else:
        client = genai.Client(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except Exception as e:
    GEMINI_AVAILABLE = False
    client = None
    print(f"Gemini API unavailable: {e}")

@dataclass
class ExtractedField:
    value: Any
    confidence: float
    source: str

@dataclass
class LineItem:
    description: str
    hsn_sac_code: Optional[str]
    quantity: Optional[float]
    unit: Optional[str]
    unit_price: Optional[float]
    discount: Optional[float]
    taxable_amount: float
    cgst_rate: Optional[float]
    cgst_amount: Optional[float]
    sgst_rate: Optional[float]
    sgst_amount: Optional[float]
    igst_rate: Optional[float]
    igst_amount: Optional[float]
    total_amount: float


@dataclass
class TaxBreakdown:
    cgst: Optional[float]
    cgst_rate: Optional[float]
    sgst: Optional[float]
    sgst_rate: Optional[float]
    igst: Optional[float]
    igst_rate: Optional[float]
    cess: Optional[float]
    tcs: Optional[float]
    total_tax: float
    tax_amounts_match: bool

@dataclass
class VendorInfo:
    name: ExtractedField
    gstin: Optional[ExtractedField]
    pan: Optional[ExtractedField]
    address: Optional[ExtractedField]
    contact: Optional[ExtractedField]
    email: Optional[ExtractedField]
    bank_account: Optional[ExtractedField]
    ifsc_code: Optional[ExtractedField]

@dataclass
class BuyerInfo:
    name: ExtractedField
    gstin: Optional[ExtractedField]
    address: Optional[ExtractedField]
    department: Optional[ExtractedField]
    cost_center: Optional[ExtractedField]

@dataclass
class DocumentAuthentication:
    has_digital_signature: bool
    has_physical_seal: bool
    has_stamp: bool
    signature_names: List[str]
    authorization_level: Optional[str]
    qr_code_present: bool
    irn_number: Optional[str]

@dataclass
class InvoiceData:
    document_id: str
    raw_text: str
    vendor: Optional[VendorInfo]
    buyer: Optional[BuyerInfo]
    invoice_number: Optional[ExtractedField]
    invoice_date: Optional[ExtractedField]
    invoice_type: Optional[ExtractedField]
    place_of_supply: Optional[ExtractedField]
    reverse_charge: Optional[bool]
    po_number: Optional[ExtractedField]
    po_date: Optional[ExtractedField]
    challan_number: Optional[ExtractedField]
    contract_reference: Optional[ExtractedField]
    total_amount: Optional[ExtractedField]
    currency: str
    taxable_amount: Optional[ExtractedField]
    discount: Optional[ExtractedField]
    line_items: List[LineItem]
    tax_breakdown: Optional[TaxBreakdown]
    payment_terms: Optional[ExtractedField]
    payment_method: Optional[ExtractedField]
    payment_due_date: Optional[ExtractedField]
    advance_paid: Optional[ExtractedField]
    balance_due: Optional[ExtractedField]
    authentication: DocumentAuthentication
    tds_applicable: Optional[bool]
    tds_rate: Optional[float]
    msme_registered: Optional[bool]
    notes: Optional[str]
    confidence_score: float
    extraction_timestamp: str
    extraction_warnings: List[str]

class InvoiceParsingAgent:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.client = client
        self.model_name = model_name
        self.supported_formats = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff']
        
    def validate_document(self, file_path: str) -> bool:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        return True
    
    def _read_file_bytes(self, file_path: str) -> tuple[bytes, str]:
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        mime_types = {
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.tiff': 'image/tiff'
        }
        
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        
        mime_type = mime_types.get(suffix, 'application/pdf')
        return file_bytes, mime_type
    
    def extract_text_with_ocr(self, file_path: str) -> str:
        if GEMINI_AVAILABLE and client:
            try:
                file_bytes, mime_type = self._read_file_bytes(file_path)
                
                prompt = """
Extract ALL text from this document with maximum accuracy.

Instructions:
1. Preserve the original structure and layout
2. Extract all text including headers, body, tables, and footers
3. Maintain table structures using clear formatting
4. Include all numbers, dates, and amounts exactly as shown
5. Extract text from stamps, seals, and signatures if visible
6. Return only the raw extracted text, no commentary

Extract the complete text now:
"""
                
                pdf_part = types.Part.from_bytes(
                    mime_type=mime_type,
                    data=file_bytes
                )
                
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, pdf_part]
                )
                
                print("Text extracted using Gemini API")
                return response.text
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                    print(f"Gemini API quota exhausted, falling back to local extraction...")
                else:
                    print(f"Gemini API error: {e}, falling back to local extraction...")
        
        if PDFPLUMBER_AVAILABLE:
            try:
                text = self._extract_with_pdfplumber(file_path)
                if text and len(text.strip()) > 100:
                    print("Text extracted using pdfplumber (fallback)")
                    return text
            except Exception as e:
                print(f"pdfplumber extraction failed: {e}")
        
        if TESSERACT_AVAILABLE:
            try:
                text = self._extract_with_tesseract(file_path)
                if text and len(text.strip()) > 50:
                    print("Text extracted using Tesseract OCR (fallback)")
                    return text
            except Exception as e:
                print(f"Tesseract OCR failed: {e}")

        raise RuntimeError(
            "All extraction methods failed. "
            "Please ensure: (1) Gemini API key is valid, OR "
            "(2) pdfplumber is installed (pip install pdfplumber), OR "
            "(3) Tesseract OCR is installed (pip install pytesseract pdf2image)"
        )
    
    def _extract_with_pdfplumber(self, file_path: str) -> str:
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    
    def _extract_with_tesseract(self, file_path: str) -> str:
        images = convert_from_path(file_path)
        
        text_parts = []
        for i, image in enumerate(images):
            page_text = pytesseract.image_to_string(image)
            if page_text:
                text_parts.append(f"--- Page {i+1} ---\n{page_text}")
        
        return "\n\n".join(text_parts)
    
    def extract_invoice_fields(self, file_path: str) -> Dict[str, Any]:
        """Extract invoice-specific fields"""
        try:
            file_bytes, mime_type = self._read_file_bytes(file_path)
            
            extraction_prompt = """
You are an expert financial document parser for government procurement fraud detection.
Extract ALL information from this invoice/bill with extreme precision.

Return a JSON object with these fields (null if not found):
{
    "vendor": {
        "name": "Full legal name",
        "gstin": "Vendor GST number (format: 29ABCDE1234F1Z5)",
        "pan": "Vendor PAN (format: ABCDE1234F)",
        "address": "Complete address",
        "contact": "Phone number",
        "email": "Email address",
        "bank_account": "Bank account number",
        "ifsc_code": "IFSC code"
    },
    "buyer": {
        "name": "Department/Organization name",
        "gstin": "BUYER GST number (CRITICAL - often missed)",
        "address": "Buyer address",
        "department": "Specific department",
        "cost_center": "Cost center or budget code"
    },
    "invoice_number": "Invoice/Bill number",
    "invoice_date": "Date in YYYY-MM-DD format",
    "invoice_type": "Tax Invoice/Bill of Supply/Export Invoice",
    "place_of_supply": "State name or code",
    "reverse_charge": true/false,
    
    "po_number": "Purchase Order number (CRITICAL)",
    "po_date": "PO date in YYYY-MM-DD",
    "challan_number": "Delivery challan number",
    "contract_reference": "Contract/Agreement reference",
    
    "total_amount": numeric without currency,
    "currency": "INR/USD/etc",
    "taxable_amount": numeric,
    "discount": numeric,
    
    "line_items": [
        {
            "description": "Item/Service description",
            "hsn_sac_code": "HSN/SAC code (CRITICAL - 4-8 digits)",
            "quantity": numeric,
            "unit": "pieces/kg/hours/etc",
            "unit_price": numeric,
            "discount": numeric,
            "taxable_amount": numeric,
            "cgst_rate": percentage,
            "cgst_amount": numeric,
            "sgst_rate": percentage,
            "sgst_amount": numeric,
            "igst_rate": percentage,
            "igst_amount": numeric,
            "total_amount": numeric
        }
    ],
    
    "tax_breakdown": {
        "cgst": total CGST amount,
        "cgst_rate": percentage,
        "sgst": total SGST amount,
        "sgst_rate": percentage,
        "igst": total IGST amount,
        "igst_rate": percentage,
        "cess": cess amount if any,
        "tcs": TCS amount if any,
        "total_tax": sum of all taxes
    },
    
    "payment_terms": "Net 30/Advance/etc",
    "payment_method": "NEFT/RTGS/Cheque/Cash",
    "payment_due_date": "YYYY-MM-DD",
    "advance_paid": numeric,
    "balance_due": numeric,
    
    "authentication": {
        "has_digital_signature": true/false,
        "has_physical_seal": true/false,
        "has_stamp": true/false,
        "signature_names": ["Name1", "Name2"],
        "authorization_level": "Level 1/Level 2/etc",
        "qr_code_present": true/false,
        "irn_number": "IRN number if e-invoice"
    },
    
    "tds_applicable": true/false,
    "tds_rate": percentage,
    "msme_registered": true/false,
    
    "notes": "Any special terms, conditions, or remarks"
}

Return ONLY the JSON object, no other text.
"""
            
            pdf_part = types.Part.from_bytes(
                mime_type=mime_type,
                data=file_bytes
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[extraction_prompt, pdf_part]
            )
            
            extracted_data = self._parse_json_response(response.text)
            extracted_data = self._normalize_extracted_data(extracted_data)
            extracted_data['warnings'] = self._validate_invoice(extracted_data)
            
            return extracted_data
            
        except Exception as e:
            raise RuntimeError(f"Field extraction failed: {str(e)}")
    
    def _validate_invoice(self, data: Dict[str, Any]) -> List[str]:
        """Validate invoice data and return warnings"""
        warnings = []
        
        buyer_gstin = data.get('buyer', {}).get('gstin')
        if not buyer_gstin:
            warnings.append("CRITICAL: Buyer GSTIN missing - cannot verify buyer identity")
        
        if not data.get('po_number'):
            warnings.append("WARNING: PO number missing - cannot link to approval")
        
        vendor = data.get('vendor', {})
        if not vendor.get('pan'):
            gstin = vendor.get('gstin')
            if gstin and len(gstin) >= 10:
                derived_pan = gstin[2:12]
                warnings.append(f"INFO: Vendor PAN derived from GSTIN: {derived_pan}")
            else:
                warnings.append("WARNING: Vendor PAN missing - identity verification limited")
        
        line_items = data.get('line_items', [])
        missing_hsn_count = sum(1 for item in line_items if not item.get('hsn_sac_code'))
        if missing_hsn_count > 0:
            warnings.append(f"WARNING: {missing_hsn_count} line items missing HSN/SAC codes")
        
        tax_breakdown = data.get('tax_breakdown', {})
        total_tax = tax_breakdown.get('total_tax', 0)
        calculated_tax = sum([
            tax_breakdown.get('cgst', 0) or 0,
            tax_breakdown.get('sgst', 0) or 0,
            tax_breakdown.get('igst', 0) or 0,
            tax_breakdown.get('cess', 0) or 0,
            tax_breakdown.get('tcs', 0) or 0
        ])
        
        if abs(total_tax - calculated_tax) > 1:
            warnings.append(f"ERROR: Tax mismatch - Stated: {total_tax}, Calculated: {calculated_tax}")
        
        auth = data.get('authentication', {})
        if not any([
            auth.get('has_digital_signature'),
            auth.get('has_physical_seal'),
            auth.get('has_stamp')
        ]):
            warnings.append("CRITICAL: No authentication markers found (signature/seal/stamp)")
        
        payment_method = data.get('payment_method')
        if payment_method:
            payment_method_upper = str(payment_method).upper()
            if payment_method_upper in ['CASH', 'BEARER CHEQUE']:
                warnings.append(f"HIGH RISK: Payment method is {payment_method_upper}")
        
        return warnings
    
    def _normalize_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and clean extracted data"""
        for field in ['total_amount', 'taxable_amount', 'discount']:
            if field in data and data[field]:
                data[field] = self._parse_amount(str(data[field]))
        
        for field in ['invoice_date', 'po_date', 'payment_due_date']:
            if field in data and data[field]:
                data[field] = self._normalize_date(data[field])
        
        return data
    
    def _parse_amount(self, amount_str: str) -> float:
        """Parse amount string to float"""
        cleaned = re.sub(r'[₹$€,\s]', '', str(amount_str))
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to YYYY-MM-DD format"""
        if not date_str:
            return None
        
        formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', 
            '%m/%d/%Y', '%Y/%m/%d', '%d-%b-%Y',
            '%d %B %Y', '%d %b %Y'
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(str(date_str).strip(), fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return date_str
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from Gemini response"""
        text = response_text.strip()
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {str(e)}")
    
    def calculate_confidence_score(self, extracted_data: Dict[str, Any]) -> float:
        """Calculate overall confidence score for invoice"""
        score = 0.0
        
        critical_fields = {
            'vendor.gstin': 0.10,
            'buyer.gstin': 0.10,
            'invoice_number': 0.08,
            'invoice_date': 0.08,
            'total_amount': 0.08,
            'po_number': 0.06,
        }
        
        for field, weight in critical_fields.items():
            if '.' in field:
                parent, child = field.split('.')
                if extracted_data.get(parent) and extracted_data[parent].get(child):
                    score += weight
            else:
                if extracted_data.get(field):
                    score += weight
        
        return round(min(score + 0.3, 1.0), 2)
    
    def process_invoice(self, file_path: str) -> InvoiceData:
        """Process invoice and extract all relevant data"""
        print(f"Processing Invoice: {Path(file_path).name}")
        
        if CACHE_ENABLED:
            cache = get_cache_manager()
            cached_result = cache.get_cached_result(file_path)
            if cached_result:
                print("Using cached extraction result (0 API calls)")
                result_data = cached_result.get('result', {})
                return self._reconstruct_invoice_data(result_data)
        
        self.validate_document(file_path)
        print("Document validated")
        
        print("Extracting text...")
        raw_text = self.extract_text_with_ocr(file_path)
        print(f"Text extracted ({len(raw_text)} characters)")
        
        print("Extracting invoice structured fields...")
        extracted = self.extract_invoice_fields(file_path)
        print("Fields extracted")
        
        vendor_data = extracted.get('vendor', {})
        buyer_data = extracted.get('buyer', {})
        vendor = self._build_vendor_info(vendor_data)
        buyer = self._build_buyer_info(buyer_data)
        
        confidence = self.calculate_confidence_score(extracted)
        print(f"Overall confidence: {confidence}")
        
        if extracted.get('warnings'):
            print(f"\nValidation Warnings ({len(extracted['warnings'])}):")
            for warning in extracted['warnings'][:5]:
                print(f"   • {warning}")

        auth_data = extracted.get('authentication', {})
        authentication = DocumentAuthentication(
            has_digital_signature=auth_data.get('has_digital_signature', False),
            has_physical_seal=auth_data.get('has_physical_seal', False),
            has_stamp=auth_data.get('has_stamp', False),
            signature_names=auth_data.get('signature_names', []),
            authorization_level=auth_data.get('authorization_level'),
            qr_code_present=auth_data.get('qr_code_present', False),
            irn_number=auth_data.get('irn_number')
        )

        invoice_data = InvoiceData(
            document_id=self._generate_document_id(),
            raw_text=raw_text,
            vendor=vendor,
            buyer=buyer,
            invoice_number=ExtractedField(extracted.get('invoice_number'), 0.95, 'gemini') if extracted.get('invoice_number') else None,
            invoice_date=ExtractedField(extracted.get('invoice_date'), 0.9, 'gemini') if extracted.get('invoice_date') else None,
            invoice_type=ExtractedField(extracted.get('invoice_type'), 0.8, 'gemini') if extracted.get('invoice_type') else None,
            place_of_supply=ExtractedField(extracted.get('place_of_supply'), 0.85, 'gemini') if extracted.get('place_of_supply') else None,
            reverse_charge=extracted.get('reverse_charge'),
            po_number=ExtractedField(extracted.get('po_number'), 0.95, 'gemini') if extracted.get('po_number') else None,
            po_date=ExtractedField(extracted.get('po_date'), 0.85, 'gemini') if extracted.get('po_date') else None,
            challan_number=ExtractedField(extracted.get('challan_number'), 0.8, 'gemini') if extracted.get('challan_number') else None,
            contract_reference=ExtractedField(extracted.get('contract_reference'), 0.8, 'gemini') if extracted.get('contract_reference') else None,
            total_amount=ExtractedField(extracted.get('total_amount'), 0.95, 'gemini') if extracted.get('total_amount') else None,
            currency=extracted.get('currency', 'INR'),
            taxable_amount=ExtractedField(extracted.get('taxable_amount'), 0.9, 'gemini') if extracted.get('taxable_amount') else None,
            discount=ExtractedField(extracted.get('discount'), 0.85, 'gemini') if extracted.get('discount') else None,
            line_items=extracted.get('line_items', []),
            tax_breakdown=extracted.get('tax_breakdown'),
            payment_terms=ExtractedField(extracted.get('payment_terms'), 0.8, 'gemini') if extracted.get('payment_terms') else None,
            payment_method=ExtractedField(extracted.get('payment_method'), 0.85, 'gemini') if extracted.get('payment_method') else None,
            payment_due_date=ExtractedField(extracted.get('payment_due_date'), 0.85, 'gemini') if extracted.get('payment_due_date') else None,
            advance_paid=ExtractedField(extracted.get('advance_paid'), 0.8, 'gemini') if extracted.get('advance_paid') else None,
            balance_due=ExtractedField(extracted.get('balance_due'), 0.8, 'gemini') if extracted.get('balance_due') else None,
            authentication=authentication,
            tds_applicable=extracted.get('tds_applicable'),
            tds_rate=extracted.get('tds_rate'),
            msme_registered=extracted.get('msme_registered'),
            notes=extracted.get('notes'),
            confidence_score=confidence,
            extraction_timestamp=datetime.now().isoformat(),
            extraction_warnings=extracted.get('warnings', [])
        )
        
        print("Invoice processing complete!")

        if CACHE_ENABLED:
            cache = get_cache_manager()
            cache.save_to_cache(file_path, self.export_to_json(invoice_data))
        
        return invoice_data
    
    def _reconstruct_invoice_data(self, cached_json: str) -> InvoiceData:
        """Reconstruct InvoiceData from cached JSON string"""
        try:
            data = json.loads(cached_json) if isinstance(cached_json, str) else cached_json
            return data
        except Exception as e:
            print(f"Cache reconstruction failed: {e}")
            raise RuntimeError("Failed to reconstruct from cache")
    
    def _build_vendor_info(self, vendor_data: Dict) -> VendorInfo:
        return VendorInfo(
            name=ExtractedField(vendor_data.get('name'), 0.9, 'gemini'),
            gstin=ExtractedField(vendor_data.get('gstin'), 0.95, 'gemini') if vendor_data.get('gstin') else None,
            pan=ExtractedField(vendor_data.get('pan'), 0.9, 'gemini') if vendor_data.get('pan') else None,
            address=ExtractedField(vendor_data.get('address'), 0.8, 'gemini') if vendor_data.get('address') else None,
            contact=ExtractedField(vendor_data.get('contact'), 0.85, 'gemini') if vendor_data.get('contact') else None,
            email=ExtractedField(vendor_data.get('email'), 0.85, 'gemini') if vendor_data.get('email') else None,
            bank_account=ExtractedField(vendor_data.get('bank_account'), 0.9, 'gemini') if vendor_data.get('bank_account') else None,
            ifsc_code=ExtractedField(vendor_data.get('ifsc_code'), 0.9, 'gemini') if vendor_data.get('ifsc_code') else None
        )
    
    def _build_buyer_info(self, buyer_data: Dict) -> BuyerInfo:
        return BuyerInfo(
            name=ExtractedField(buyer_data.get('name'), 0.9, 'gemini'),
            gstin=ExtractedField(buyer_data.get('gstin'), 0.95, 'gemini') if buyer_data.get('gstin') else None,
            address=ExtractedField(buyer_data.get('address'), 0.8, 'gemini') if buyer_data.get('address') else None,
            department=ExtractedField(buyer_data.get('department'), 0.75, 'gemini') if buyer_data.get('department') else None,
            cost_center=ExtractedField(buyer_data.get('cost_center'), 0.7, 'gemini') if buyer_data.get('cost_center') else None
        )

    def _generate_document_id(self) -> str:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"INV-{timestamp}"
    
    def export_to_json(self, invoice_data: Union[InvoiceData, Dict[str, Any]]) -> str:
        if isinstance(invoice_data, dict):
            return json.dumps(invoice_data, indent=2)

        def default_serializer(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        return json.dumps(asdict(invoice_data), indent=2, default=default_serializer)

if __name__ == "__main__":
    agent = InvoiceParsingAgent()

    folder = Path("sample_invoices")
 
    if not folder.exists():
        print(f"Folder '{folder}' does not exist.")
    else:
        pdf_files = list(folder.glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {folder}.")
        else:
            print(f"Found {len(pdf_files)} PDF files.")

            # Process first file
            first_pdf = pdf_files[0]
            print(f"\nProcessing: {first_pdf.name}\n")

            try:
                result = agent.process_invoice(str(first_pdf))

                print(f"\n{'='*60}")
                print(f"INVOICE EXTRACTION SUMMARY")
                print(f"{'='*60}")

                print(f"\nInvoice ID: {result.document_id}")
                print(f"Confidence: {result.confidence_score}")

                if result.vendor:
                    print(f"\n👤 Vendor: {result.vendor.name.value}")
                    if result.vendor.gstin:
                        print(f"   GSTIN: {result.vendor.gstin.value}")

                if result.buyer:
                    print(f"\n🏢 Buyer: {result.buyer.name.value}")
                    if result.buyer.gstin:
                        print(f"   GSTIN: {result.buyer.gstin.value}")

                if result.invoice_number:
                    print(f"\n💰 Invoice No: {result.invoice_number.value}")
                if result.total_amount:
                    print(f"   Amount: {result.currency} {result.total_amount.value:,.2f}")

                # Save to JSON
                json_output = agent.export_to_json(result)
                output_file = f"extracted_invoice_{result.document_id}.json"

                with open(output_file, 'w') as f:
                    f.write(json_output)

                print(f"\n💾 Saved to: {output_file}")

            except Exception as e:
                print(f"❌ Error while processing invoice: {str(e)}")
                import traceback
                traceback.print_exc()
