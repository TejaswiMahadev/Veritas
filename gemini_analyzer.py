import json
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()


class GeminiAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.use_vertex = os.getenv('USE_VERTEX_AI', 'false').lower() == 'true'
        
        if self.use_vertex:
            print("Using Vertex AI (Google Cloud)")
            self.client = genai.Client(
                vertexai=True,
                project=os.getenv('GCP_PROJECT_ID'),
                location=os.getenv('GCP_LOCATION', 'us-central1')
            )
        else:
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

    def cross_examine_flags(self, triggered_flags: List[Dict[str, Any]], doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enrich triggered flags with AI reasoning: Why, Auditor Question, Financial Risk.
        """
        if not triggered_flags:
            return []

        prompt = f"""You are an expert AI Invoice Auditor. For each triggered fraud/compliance flag, provide a deep cross-examination.
        
        INPUT FLAGS:
        {json.dumps(triggered_flags, indent=2)}
        
        DOCUMENT DATA (CONTEXT):
        {json.dumps(doc_data, indent=2, default=str)}
        
        OUTPUT FORMAT (JSON):
        Return a list of objects, one for each input flag (maintain the order). Each object MUST have:
        1. "id": (matching input flag id)
        2. "ai_audit_check": {{
            "why_is_wrong": "Deep technical/tax/compliance justification for why this flag is a risk",
            "auditor_question": "Strong tactical question to ask the vendor or finance team to resolve/validate this",
            "financial_risk": "Specific financial impact (ITC loss, penalty, litigation risk, or direct fraud loss)"
        }}

        Return ONLY valid JSON."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )
            enriched_data = json.loads(response.text)
            
            # Map enriched data back to flags
            # Expecting a list from AI
            if isinstance(enriched_data, list):
                # Create a lookup for enriched data
                enrich_map = {item['id']: item['ai_audit_check'] for item in enriched_data if 'id' in item and 'ai_audit_check' in item}
                
                for flag in triggered_flags:
                    fid = flag.get('id')
                    if fid in enrich_map:
                        flag['ai_audit_check'] = enrich_map[fid]
            
            return triggered_flags
            
        except Exception as e:
            # We don't raise here to allow the pipeline to continue with rule-based flags
            # but we'll return a clear indicator if needed.
            print(f"❌ AI Cross-Examination failed: {e}")
            return triggered_flags

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
