"""
Google Cloud Vision API for OCR
Extracts text and detects authentication elements (signatures, QR codes) from images/PDFs
"""

import io
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from google.cloud import vision
import base64


class CloudVisionOCR:
    """Google Cloud Vision-based OCR for invoice documents"""
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize Cloud Vision client
        
        Args:
            project_id: GCP project ID (uses GOOGLE_CLOUD_PROJECT env var if not provided)
        """
        self.client = vision.ImageAnnotatorClient()
        self.project_id = project_id
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract all text from an image or PDF file using Cloud Vision OCR
        
        Args:
            file_path: Path to image or PDF file
            
        Returns:
            Extracted text from the document
        """
        with open(file_path, 'rb') as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        response = self.client.document_text_detection(image=image)
        
        # Extract text from response
        texts = response.text_annotations
        if not texts:
            return ""
        
        # First annotation contains full text
        return texts[0].description if texts else ""
    
    def detect_authentication_elements(self, file_path: str) -> Dict[str, Any]:
        """
        Detect authentication elements in document:
        - Handwritten/digital signatures
        - QR codes (e-invoice indicator)
        - Seals and stamps
        - Digital certification marks
        
        Args:
            file_path: Path to image or PDF file
            
        Returns:
            Dictionary with detection results:
            {
                "has_signature": bool,
                "has_qr_code": bool,
                "has_seal": bool,
                "has_stamp": bool,
                "confidence": float,
                "details": str
            }
        """
        with open(file_path, 'rb') as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        
        # Use multiple detection methods
        response = self.client.document_text_detection(image=image)
        text_response = response
        
        # Also run barcode detection for QR codes
        qr_response = self.client.detect_barcodes(image=image)
        
        # Extract results
        result = {
            "has_signature": False,
            "has_qr_code": False,
            "has_seal": False,
            "has_stamp": False,
            "has_authentication": False,
            "confidence": 0.0,
            "details": [],
            "qr_codes_found": []
        }
        
        # Check for QR codes
        if qr_response.barcode_annotations:
            for barcode in qr_response.barcode_annotations:
                if "QR_CODE" in barcode.bounding_poly.vertices or "qr" in barcode.description.lower():
                    result["has_qr_code"] = True
                    result["qr_codes_found"].append({
                        "format": barcode.format,
                        "raw_value": barcode.raw_value
                    })
                    result["details"].append("QR Code detected (e-invoice indicator)")
        
        # Check text for signature/seal indicators
        full_text = text_response.text_annotations[0].description if text_response.text_annotations else ""
        
        # Signature keywords
        signature_keywords = [
            "signature", "signed by", "authorized by", "signed", "signature line",
            "digital signature", "_____", "digitally signed", "დ ს", "✓"
        ]
        
        # Seal keywords
        seal_keywords = [
            "seal", "stamp", "official seal", "company seal", "round seal",
            "watermark", "certified", "verified seal", "certification"
        ]
        
        full_text_lower = full_text.lower()
        
        for keyword in signature_keywords:
            if keyword in full_text_lower:
                result["has_signature"] = True
                result["details"].append(f"Signature keyword found: '{keyword}'")
                break
        
        for keyword in seal_keywords:
            if keyword in full_text_lower:
                result["has_seal"] = True
                result["details"].append(f"Seal/stamp keyword found: '{keyword}'")
                break
        
        # Check for visual elements using safe_search
        safe_search = self.client.safe_search_detection(image=image).safe_search_annotation
        
        # Set overall authentication flag
        result["has_authentication"] = (
            result["has_signature"] or 
            result["has_qr_code"] or 
            result["has_seal"] or 
            result["has_stamp"]
        )
        
        # Calculate confidence based on detections
        detection_count = sum([
            result["has_signature"],
            result["has_qr_code"],
            result["has_seal"],
            result["has_stamp"]
        ])
        result["confidence"] = min(0.95, 0.3 + (detection_count * 0.2))
        
        return result
    
    def extract_structured_data(self, file_path: str) -> Dict[str, Any]:
        """
        Extract both text and authentication elements
        
        Args:
            file_path: Path to image or PDF file
            
        Returns:
            Combined extraction result
        """
        text = self.extract_text_from_file(file_path)
        auth = self.detect_authentication_elements(file_path)
        
        return {
            "extracted_text": text,
            "authentication": auth,
            "extraction_method": "google_cloud_vision"
        }
    
    def batch_extract(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Extract text from multiple files
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary mapping file paths to extracted data
        """
        results = {}
        for file_path in file_paths:
            try:
                results[file_path] = self.extract_structured_data(file_path)
            except Exception as e:
                results[file_path] = {"error": str(e)}
        
        return results
