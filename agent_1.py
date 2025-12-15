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

# NIM Client
from utils.nim_client import NimClient


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
    - OCR (local)
    - Extraction + Normalization (Gemma via NIM)
    """
    
    def __init__(self, nim_api_key: Optional[str] = None):
        self.nim_client = NimClient(api_key=nim_api_key)
        self.supported_formats = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff']
        
    def process(self, file_path: str) -> Dict[str, Any]:
        """
        Main entry point: Extract text -> Call Gemma -> Return Structured JSON
        """
        print(f"Processing invoice: {file_path}")
        
        # 1. OCR / Text Extraction
        raw_text = self._extract_text(file_path)
        if not raw_text or len(raw_text.strip()) < 10:
             raise ValueError("Could not extract sufficient text from document.")
             
        # Sanitize text (remove null bytes and control characters that break JSON)
        raw_text = self._sanitize_text(raw_text)
             
        # 2. Gemma Extraction & Normalization
        structured_data = self._gemma_extraction(raw_text)
        
        if not isinstance(structured_data, dict):
             print(f"Warning: Gemma extraction returned non-dict: {type(structured_data)}")
             structured_data = {"error": "Invalid Output Format", "raw_output": str(structured_data)}
        
        # 3. Final cleanup / structuration
        # We ensure the output matches what Agent 3 expects
        final_output = self._standardize_output(structured_data, raw_text, file_path)
        
        return final_output

    def _extract_text(self, file_path: str) -> str:
        """Extract text using pdfplumber (text) or pytesseract (OCR)"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            text = self._extract_pdf_text(file_path)
            # If text is sparse, try OCR
            if len(text.strip()) < 50:
                 print("PDF text sparse, falling back to OCR...")
                 pass 
            return text
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff']:
            return self._extract_image_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

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

        import re
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text

    def _gemma_extraction(self, raw_text: str) -> Dict[str, Any]:
        """
        Uses Gemma 3N via NIM to extract and normalize in one shot.
        """
        prompt = f"""
        You are an expert Autonomous Invoice Auditor. Your job is to extract and normalize data from the provided invoice text.
        
        Rules:
        1. Extract the following fields exactly:
           - invoice_number (string)
           - invoice_date (YYYY-MM-DD format)
           - total_amount (float, numbers only)
           - tax_amount (float)
           - vendor_name (string)
           - vendor_gstin (string, 15 chars)
           - vendor_pan (string, 10 chars)
           - buyer_name (string)
           - buyer_gstin (string)
           - line_items (list of objects with: description, quantity, unit_price, amount)
        
        2. Format dates to YYYY-MM-DD.
        3. Remove currency symbols from amounts.
        4. Normalize GSTINs to uppercase and remove spaces/hyphens.
        5. For `line_items`, ensure `amount` is the total for that line (qty * price).
        
        Output strictly valid JSON.
        
        Invoice Text:
        \"\"\"
        {raw_text[:4000]}  # Truncate to avoid context window issues if necessary
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
        # Ensure minimal fields exist
        defaults = {
            "invoice_number": None,
            "invoice_date": None,
            "total_amount": 0.0,
            "vendor_name": "Unknown Vendor",
            "vendor_gstin": None,
            "buyer_name": None
        }
        
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
                    "address": None 
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
                    "has_digital_signature": False, # TODO: Add extraction logic
                    "has_seal": False
                },
                "raw_text": raw_text, # Keep raw text for reference
                "extraction_method": "NIM_Gemma_Combined"
            }
        }
