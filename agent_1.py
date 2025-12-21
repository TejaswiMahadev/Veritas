import os
import json
import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict

# OCR Libraries
import pdfplumber
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract or Pillow not installed. OCR features disabled.")

# Cloud Vision OCR
try:
    from utils.cloud_vision_ocr import CloudVisionOCR
    CLOUD_VISION_AVAILABLE = True
except ImportError:
    CLOUD_VISION_AVAILABLE = False
    print("Warning: Cloud Vision not available. Install google-cloud-vision for enhanced OCR.")

# NIM Client
from utils.nim_client import NimClient

# Import NormalizeInvoice for compatibility if needed, or define here.
# We will define a structure compatible with the rest of the pipeline.

@dataclass
class ExtractedNormalizedInvoice:
    """Combined Data Structure for Extraction + Normalization"""
    document_id: str
    summary_extraction: Dict[str, Any] # The raw/structured extraction from Gemma
    raw_text: str
    
    # Standard Fields expected by Agent 3
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    total_amount: Optional[float] = None
    currency: str = "INR"
    vendor_name: Optional[str] = None
    vendor_gstin: Optional[str] = None
    buyer_name: Optional[str] = None
    line_items: List[Dict[str, Any]] = None
    
    warnings: List[str] = None

class InvoiceProcessor:
    """
    Combined Agent 1 & Agent 2.
    - OCR (local + Google Cloud Vision)
    - Extraction + Normalization (Gemma via NIM)
    """
    
    def __init__(self, nim_api_key: Optional[str] = None, gcp_project_id: Optional[str] = None):
        self.nim_client = NimClient(api_key=nim_api_key)
        self.supported_formats = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff']
        
        # Initialize Cloud Vision if available
        self.cloud_vision = None
        if CLOUD_VISION_AVAILABLE:
            try:
                self.cloud_vision = CloudVisionOCR(project_id=gcp_project_id)
                print("Cloud Vision OCR initialized successfully")
            except Exception as e:
                print(f"Warning: Cloud Vision initialization failed: {e}")
        
    def process(self, file_path: str) -> Dict[str, Any]:
        """
        Main entry point: Extract text -> Detect auth elements -> Call Gemma -> Return Structured JSON
        """
        print(f"Processing invoice: {file_path}")
        
        # 1. OCR / Text Extraction (Cloud Vision first, fallback to local)
        raw_text = self._extract_text(file_path)
        if not raw_text or len(raw_text.strip()) < 10:
             raise ValueError("Could not extract sufficient text from document.")
             
        # Sanitize text (remove null bytes and control characters that break JSON)
        raw_text = self._sanitize_text(raw_text)
        
        # 2. Detect authentication elements using Cloud Vision
        auth_elements = self._detect_authentication(file_path)
        
        # 3. Gemma Extraction & Normalization
        structured_data = self._gemma_extraction(raw_text, auth_elements)
        
        if not isinstance(structured_data, dict):
             print(f"Warning: Gemma extraction returned non-dict: {type(structured_data)}")
             structured_data = {"error": "Invalid Output Format", "raw_output": str(structured_data)}
        
        # 4. Final cleanup / structuration
        # We ensure the output matches what Agent 3 expects
        final_output = self._standardize_output(structured_data, raw_text, file_path)
        
        return final_output

    def _extract_text(self, file_path: str) -> str:
        """Extract text using Cloud Vision (primary) or local OCR (fallback)"""
        ext = Path(file_path).suffix.lower()
        
        # Try Cloud Vision first
        if self.cloud_vision:
            try:
                text = self.cloud_vision.extract_text_from_file(file_path)
                if text and len(text.strip()) > 10:
                    print("Text extracted successfully using Cloud Vision")
                    return text
            except Exception as e:
                print(f"Cloud Vision extraction failed: {e}. Falling back to local OCR...")
        
        # Fallback to local extraction
        if ext == '.pdf':
            text = self._extract_pdf_text(file_path)
            return text
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff']:
            return self._extract_image_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _detect_authentication(self, file_path: str) -> Dict[str, Any]:
        """
        Detect authentication elements (signatures, QR codes, seals) using Cloud Vision
        """
        if not self.cloud_vision:
            return {
                "has_signature": False,
                "has_qr_code": False,
                "has_seal": False,
                "has_authentication": False,
                "confidence": 0.0
            }
        
        try:
            auth_result = self.cloud_vision.detect_authentication_elements(file_path)
            print(f"Authentication detection result: {auth_result['has_authentication']}")
            return auth_result
        except Exception as e:
            print(f"Authentication detection failed: {e}")
            return {
                "has_signature": False,
                "has_qr_code": False,
                "has_seal": False,
                "has_authentication": False,
                "confidence": 0.0,
                "error": str(e)
            }

    def _extract_pdf_text(self, file_path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            print(f"PDF extraction error: {e}")
        return text

    def _extract_image_text(self, file_path: str) -> str:
        if not OCR_AVAILABLE:
            return ""
        try:
            image = Image.open(file_path)
            return pytesseract.image_to_string(image)
        except Exception as e:
            print(f"Image OCR error: {e}")
            return ""

    def _sanitize_text(self, text: str) -> str:
        """Remove null bytes and non-printable characters."""
        if not text:
            return ""
        # Remove null bytes
        text = text.replace('\x00', '')
        # Remove other control characters (except newlines/tabs)
        # using a simple regex or just keeping it simple for now with strict replacement
        # C0 control set: 00-1F and 7F. We want to keep 09 (\t), 0A (\n), 0D (\r)
        import re
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text

    def _gemma_extraction(self, raw_text: str, auth_elements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Uses Gemma 3N via NIM to extract and normalize in one shot.
        Incorporates Cloud Vision authentication detection results.
        """
        # Build authentication context from Cloud Vision results
        auth_context = ""
        if auth_elements and auth_elements.get("has_authentication"):
            auth_context = f"""

AUTHENTICATION DETECTED BY CLOUD VISION OCR:
- Digital Signature Found: {auth_elements.get('has_signature', False)}
- QR Code Found: {auth_elements.get('has_qr_code', False)}
- Seal/Stamp Found: {auth_elements.get('has_seal', False)}
- Detection Confidence: {auth_elements.get('confidence', 0):.1%}
Details: {'; '.join(auth_elements.get('details', ['None found']))}

Use this information to validate and confirm authentication fields in your extraction."""
        
        prompt = f"""
        You are an expert Autonomous Invoice Auditor. Your job is to extract and normalize data from the provided invoice text.
        
        CRITICAL: Extract ALL fields present in the document, even if partially visible or unclear. Return null only if absolutely not found.
        
        Extract these fields exactly:
        1. invoice_number: The unique invoice identifier (NOT PO number). Look for "Invoice No", "Inv#", "Invoice ID"
        2. invoice_date: Date invoice was issued (YYYY-MM-DD format). Look for "Invoice Date", "Date of Invoice", "Dated"
        3. total_amount: Total invoice amount (numbers only, no currency). Include all taxes and amounts.
        4. tax_amount: Total tax amount (SGST + CGST + IGST + GST). Look for GST breakdown sections.
        5. vendor_name: Selling company name (the party issuing the invoice)
        6. vendor_gstin: GSTIN of vendor (15 chars, format: 29ABCDE1234F1Z5). Look in "From" or "Billed By" section
        7. vendor_pan: PAN extracted from GSTIN or from separate field
        8. buyer_name: Buying company name (the party receiving goods/services)
        9. buyer_gstin: GSTIN of buyer (15 chars). Look in "To" or "Bill To" section
        10. line_items: Array of line items. Each must have: description, quantity (or null), unit_price (or null), amount
        11. has_digital_signature: Boolean - true if ANY of these present:
            - Digital signature block
            - Signature line with name
            - QR code (indicates e-invoice)
            - IRN (Invoice Reference Number)
            - Digital certification mark
        12. has_seal: Boolean - true if ANY of these present:
            - Physical seal/stamp impression
            - Company seal
            - Watermark marking
            - Government certification seal
        13. has_authentication: Boolean - true if any authentication element exists
        
        IMPORTANT RULES:
        - Look carefully at the entire document for all fields
        - If you see text that looks like an invoice number (alphanumeric identifier), extract it
        - If you see a date near "Date" or "Dated" label, extract it
        - Even if fields are small or partially cut off, try to extract them
        - Return all numeric fields as numbers (not strings)
        - Do NOT return "Not specified" - return null instead
        
        Output ONLY valid JSON with no markdown or extra text.
        {auth_context}
        
        INVOICE TEXT TO ANALYZE:
        \"\"\"
        {raw_text[:5000]}
        \"\"\"
        """
        
        try:
            return self.nim_client.generate(prompt, json_mode=True)
        except Exception as e:
            print(f"Gemma extraction failed: {e}")
            return {"error": str(e), "raw_text_snippet": raw_text[:200]}

    def _standardize_output(self, gemma_data: Dict[str, Any], raw_text: str, file_path: str) -> Dict[str, Any]:
        """
        Map Gemma's output to the standard schema expected by the rest of the pipeline.
        Agent 3 expects:
        {
            "document": {
                "invoice_number": ...,
                "amounts": { ... },
                "vendor": { ... },
                ...
            }
        }
        """
        # Check for extraction errors
        if "error" in gemma_data:
            print(f"WARNING: Gemma extraction error: {gemma_data.get('error')}")
        
        # Ensure minimal fields exist with None as defaults (not empty strings)
        defaults = {
            "invoice_number": None,
            "invoice_date": None,
            "total_amount": 0.0,
            "tax_amount": 0.0,
            "vendor_name": None,
            "vendor_gstin": None,
            "vendor_pan": None,
            "buyer_name": None,
            "buyer_gstin": None,
            "has_digital_signature": False,
            "has_seal": False,
            "has_authentication": False
        }
        
        # Merge: defaults first, then Gemma's data (Gemma values take precedence)
        data = {**defaults, **gemma_data}
        
        # Deterministic Fixes
        # 1. Derive PAN from GSTIN if missing
        v_gstin = data.get("vendor_gstin")
        if not data.get("vendor_pan") and v_gstin and len(v_gstin) == 15:
             data["vendor_pan"] = v_gstin[2:12]
             
        # 2. Ensure line_items is a list
        if not isinstance(data.get("line_items"), list):
             data["line_items"] = []
             
        # 3. Calculate derived amounts
        total = float(data.get("total_amount") or 0.0)
        tax = float(data.get("tax_amount") or 0.0)
        taxable = total - tax if total > 0 else 0.0

        # 4. Consolidate authentication flags (if any is true, set has_authentication)
        has_any_auth = (data.get("has_digital_signature", False) or 
                       data.get("has_seal", False) or 
                       data.get("has_authentication", False))

        # Construct Nested Output
        return {
            "document": {
                "document_id": Path(file_path).name,
                "document_type": "INVOICE",
                "invoice_number": data.get("invoice_number"),
                "invoice_date": data.get("invoice_date"),
                "vendor": {
                    "name": data.get("vendor_name"),
                    "gstin": data.get("vendor_gstin"),
                    "pan": data.get("vendor_pan"),
                    "address": None # Gemma didn't extract this yet
                },
                "buyer": {
                    "name": data.get("buyer_name"),
                    "gstin": data.get("buyer_gstin"),
                    "address": None
                },
                "amounts": {
                    "total_amount": total,
                    "tax_amount": tax,
                    "taxable_amount": round(taxable, 2),
                    "currency": "INR"
                },
                "line_items": data.get("line_items", []),
                "authentication": {
                    "has_digital_signature": data.get("has_digital_signature", False),
                    "has_seal": data.get("has_seal", False),
                    "has_authentication": has_any_auth
                },
                "raw_text": raw_text, # Keep raw text for reference
                "extraction_method": "NIM_Gemma_Combined"
            }
        }
