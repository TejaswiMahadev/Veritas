import json
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()


class GeminiAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash"

    def analyze_invoice_semantics(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Deep semantic analysis for invoice fraud."""
        prompt = f"""Analyze this invoice for subtle fraud indicators. Look for:
1. Suspicious vendor/buyer relationships (same address, unusual names)
2. Suspicious payment terms (too fast, too slow relative to amount)
3. Unusual pricing patterns (too low, too high, round numbers)
4. Red flags in line items (missing descriptions, generic items)
5. Missing standard business info (PO references, delivery details)

DOCUMENT:
{json.dumps(doc, indent=2, default=str)}

Return JSON:
{{
    "risk_indicators": ["list of suspicious patterns"],
    "semantic_fraud_score": 0-1.0,
    "reasoning": "brief explanation",
    "recommended_action": "approve|review|investigate"
}}

Return ONLY valid JSON."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            return {
                "error": str(e),
                "semantic_fraud_score": 0.0,
                "risk_indicators": [],
                "reasoning": f"Analysis failed: {str(e)}"
            }

    def analyze(self, doc_type: str, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Route to appropriate semantic analyzer."""
        if doc_type == "INVOICE":
            return self.analyze_invoice_semantics(doc)
        elif doc_type in ("TENDER_DOCUMENT", "TENDER"):
            return self.analyze_tender_semantics(doc)
        elif doc_type == "CONTRACT":
            return self.analyze_contract_semantics(doc)
        else:
            return {"error": "Unknown document type", "semantic_fraud_score": 0.0}
