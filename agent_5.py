"""
Agent 6: Multi-Agent Disagreement Resolver
Resolves conflicts between rule-based and AI-based auditors through intelligent arbitration
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# DISAGREEMENT RESOLVER CONFIGURATION
# =============================================================================

DISAGREEMENT_THRESHOLDS = {
    "fraud_score_diff": 20,  # Difference in fraud scores to trigger resolution
    "risk_level_mismatch": True,  # Different risk levels trigger resolution
    "flag_count_diff": 3,  # Difference in number of flags
}

RESOLUTION_STRATEGIES = {
    "conservative": "favor_stricter",  # Choose the stricter assessment
    "balanced": "ai_arbitration",      # Use AI to arbitrate
    "evidence_based": "evidence_weight"  # Weight by evidence quality
}


# =============================================================================
# DISAGREEMENT RESOLVER
# =============================================================================

class MultiAgentDisagreementResolver:
    """
    Multi-Agent Audit Committee
    
    Coordinates two independent auditors:
    - Agent A: Rule-based auditor (deterministic, transparent)
    - Agent B: AI reasoning auditor (semantic, contextual)
    
    When they disagree, a 3rd agent (Resolver) arbitrates using:
    - Evidence quality assessment
    - Confidence scores
    - Historical accuracy
    - AI-powered reasoning
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        resolution_strategy: str = "balanced"
    ):
        """
        Initialize disagreement resolver
        
        Args:
            api_key: Gemini API key for AI arbitration
            resolution_strategy: How to resolve disagreements
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.resolution_strategy = RESOLUTION_STRATEGIES.get(
            resolution_strategy,
            RESOLUTION_STRATEGIES["balanced"]
        )
        
        self.use_vertex = os.getenv('USE_VERTEX_AI', 'false').lower() == 'true'
        self.client = None
        
        try:
            if self.use_vertex:
                print(" Using Vertex AI for arbitration (Google Cloud)")
                self.client = genai.Client(
                    vertexai=True,
                    project=os.getenv('GCP_PROJECT_ID'),
                    location=os.getenv('GCP_LOCATION', 'us-central1')
                )
            else:
                if self.api_key:
                    self.client = genai.Client(api_key=self.api_key)
            
            if self.client:
                self.model_name = "gemini-2.5-flash"
                print(" AI Arbitrator initialized")
            else:
                if not self.use_vertex:
                    print(" AI Arbitrator: No API key provided")
        except Exception as e:
            print(f"  AI Arbitrator initialization failed: {e}")
            self.client = None
    
    def analyze_with_committee(
        self,
        invoice_data: Dict[str, Any],
        rule_based_analysis: Dict[str, Any],
        ai_based_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run audit committee analysis
        
        Args:
            invoice_data: Normalized invoice data
            rule_based_analysis: Agent 3 (rule-based) output
            ai_based_analysis: Optional separate AI analysis
            
        Returns:
            Committee decision with resolution details
        """
        # Extract Agent A (Rule-based) assessment
        agent_a = {
            "name": "Rule-Based Auditor",
            "type": "deterministic",
            "fraud_score": rule_based_analysis.get('fraud_score', 0),
            "risk_level": rule_based_analysis.get('risk_level', 'UNKNOWN'),
            "triggered_flags": rule_based_analysis.get('triggered_flags', []),
            "flag_count": len(rule_based_analysis.get('triggered_flags', [])),
            "reasoning": "Deterministic rule evaluation",
            "confidence": 1.0  # Rules are always confident
        }
        
        # Extract Agent B (AI-based) assessment
        semantic = rule_based_analysis.get('semantic_analysis', {})
        if ai_based_analysis:
            semantic = ai_based_analysis
        
        agent_b = {
            "name": "AI Reasoning Auditor",
            "type": "semantic",
            "fraud_score": semantic.get('semantic_fraud_score', 0) * 100,
            "risk_level": self._semantic_to_risk_level(semantic.get('semantic_fraud_score', 0)),
            "risk_indicators": semantic.get('risk_indicators', []),
            "flag_count": len(semantic.get('risk_indicators', [])),
            "reasoning": semantic.get('reasoning', 'No AI analysis available'),
            "confidence": semantic.get('confidence', 0.5) if isinstance(semantic.get('confidence'), (int, float)) else 0.5
        }
        
        # Check for disagreement
        disagreement = self._detect_disagreement(agent_a, agent_b)
        
        if disagreement['has_disagreement']:
            print(f"  DISAGREEMENT DETECTED: {disagreement['reason']}")
            
            # Resolve disagreement
            resolution = self._resolve_disagreement(
                invoice_data,
                agent_a,
                agent_b,
                disagreement
            )
        else:
            print(" AGENTS AGREE")
            resolution = {
                "resolution_needed": False,
                "final_decision": "consensus",
                "final_fraud_score": agent_a['fraud_score'],
                "final_risk_level": agent_a['risk_level'],
                "reasoning": "Both agents reached similar conclusions"
            }
        
        # Build committee report
        return {
            "committee_decision": {
                "resolution_needed": disagreement['has_disagreement'],
                "disagreement_type": disagreement.get('reason'),
                "final_fraud_score": resolution['final_fraud_score'],
                "final_risk_level": resolution['final_risk_level'],
                "decision_basis": resolution.get('decision_basis', 'consensus'),
                "confidence": resolution.get('confidence', 1.0)
            },
            "agent_a_assessment": agent_a,
            "agent_b_assessment": agent_b,
            "resolution_details": resolution,
            "audit_trail": {
                "timestamp": datetime.now().isoformat(),
                "resolution_strategy": self.resolution_strategy,
                "arbitrator_used": resolution.get('arbitrator_used', False)
            },
            "recommendations": self._generate_committee_recommendations(
                agent_a,
                agent_b,
                resolution
            )
        }
    
    def _detect_disagreement(
        self,
        agent_a: Dict[str, Any],
        agent_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect if agents disagree significantly"""
        disagreements = []
        
        # Check fraud score difference
        score_diff = abs(agent_a['fraud_score'] - agent_b['fraud_score'])
        if score_diff >= DISAGREEMENT_THRESHOLDS['fraud_score_diff']:
            disagreements.append(f"Fraud score differs by {score_diff:.1f} points")
        
        # Check risk level mismatch
        if agent_a['risk_level'] != agent_b['risk_level']:
            disagreements.append(f"Risk level mismatch: {agent_a['risk_level']} vs {agent_b['risk_level']}")
        
        # Check flag count difference
        flag_diff = abs(agent_a['flag_count'] - agent_b['flag_count'])
        if flag_diff >= DISAGREEMENT_THRESHOLDS['flag_count_diff']:
            disagreements.append(f"Flag count differs by {flag_diff}")
        
        return {
            "has_disagreement": len(disagreements) > 0,
            "reason": "; ".join(disagreements) if disagreements else None,
            "severity": "high" if len(disagreements) >= 2 else "moderate" if disagreements else "none"
        }
    
    def _resolve_disagreement(
        self,
        invoice_data: Dict[str, Any],
        agent_a: Dict[str, Any],
        agent_b: Dict[str, Any],
        disagreement: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve disagreement between agents"""
        
        if self.resolution_strategy == "favor_stricter":
            return self._resolve_conservative(agent_a, agent_b)
        
        elif self.resolution_strategy == "evidence_weight":
            return self._resolve_by_evidence(agent_a, agent_b)
        
        elif self.resolution_strategy == "ai_arbitration" and self.client:
            return self._resolve_with_ai_arbitrator(invoice_data, agent_a, agent_b, disagreement)
        
        else:
            # Fallback to conservative
            return self._resolve_conservative(agent_a, agent_b)
    
    def _resolve_conservative(
        self,
        agent_a: Dict[str, Any],
        agent_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Conservative resolution: favor the stricter assessment"""
        
        # Choose higher fraud score
        if agent_a['fraud_score'] >= agent_b['fraud_score']:
            winner = agent_a
            decision = "Favored rule-based auditor (stricter)"
        else:
            winner = agent_b
            decision = "Favored AI auditor (stricter)"
        
        return {
            "resolution_needed": True,
            "final_decision": "conservative",
            "final_fraud_score": winner['fraud_score'],
            "final_risk_level": winner['risk_level'],
            "decision_basis": decision,
            "reasoning": "Conservative approach: selected stricter assessment to minimize risk",
            "confidence": 0.8,
            "arbitrator_used": False
        }
    
    def _resolve_by_evidence(
        self,
        agent_a: Dict[str, Any],
        agent_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evidence-based resolution: weight by evidence quality"""
        
        # Rule-based has concrete flags, AI has confidence
        rule_weight = min(1.0, agent_a['flag_count'] / 10) * agent_a['confidence']
        ai_weight = agent_b['confidence']
        
        # Weighted average
        total_weight = rule_weight + ai_weight
        if total_weight > 0:
            final_score = (
                (agent_a['fraud_score'] * rule_weight) +
                (agent_b['fraud_score'] * ai_weight)
            ) / total_weight
        else:
            final_score = (agent_a['fraud_score'] + agent_b['fraud_score']) / 2
        
        # Determine risk level from final score
        final_risk = self._score_to_risk_level(final_score)
        
        return {
            "resolution_needed": True,
            "final_decision": "evidence_weighted",
            "final_fraud_score": round(final_score, 2),
            "final_risk_level": final_risk,
            "decision_basis": f"Weighted by evidence (Rule: {rule_weight:.2f}, AI: {ai_weight:.2f})",
            "reasoning": "Balanced both assessments based on evidence quality and confidence",
            "confidence": (rule_weight + ai_weight) / 2,
            "arbitrator_used": False
        }
    
    def _resolve_with_ai_arbitrator(
        self,
        invoice_data: Dict[str, Any],
        agent_a: Dict[str, Any],
        agent_b: Dict[str, Any],
        disagreement: Dict[str, Any]
    ) -> Dict[str, Any]:
        """AI-powered arbitration"""
        
        try:
            # Prepare arbitration prompt
            prompt = self._build_arbitration_prompt(invoice_data, agent_a, agent_b, disagreement)
            
            # Get AI arbitration
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json"
                )
            )
            
            arbitration = json.loads(response.text)
            
            return {
                "resolution_needed": True,
                "final_decision": "ai_arbitration",
                "final_fraud_score": arbitration.get('final_fraud_score', agent_a['fraud_score']),
                "final_risk_level": arbitration.get('final_risk_level', agent_a['risk_level']),
                "decision_basis": arbitration.get('decision', 'AI arbitration'),
                "reasoning": arbitration.get('reasoning', 'AI-powered resolution'),
                "confidence": arbitration.get('confidence', 0.7),
                "arbitrator_used": True,
                "arbitrator_analysis": arbitration.get('analysis', '')
            }
            
        except Exception as e:
            print(f"  AI arbitration failed: {e}")
            # Fallback to evidence-based
            return self._resolve_by_evidence(agent_a, agent_b)
    
    def _build_arbitration_prompt(
        self,
        invoice_data: Dict[str, Any],
        agent_a: Dict[str, Any],
        agent_b: Dict[str, Any],
        disagreement: Dict[str, Any]
    ) -> str:
        """Build prompt for AI arbitrator"""
        
        return f"""You are an expert audit arbitrator resolving a disagreement between two auditors.

INVOICE SUMMARY:
- Vendor: {invoice_data.get('vendor_name', 'Unknown')}
- Amount: {invoice_data.get('amounts', {}).get('total_amount', 'Unknown')}
- Invoice Number: {invoice_data.get('invoice_number', 'Unknown')}

AGENT A (Rule-Based Auditor):
- Fraud Score: {agent_a['fraud_score']}/100
- Risk Level: {agent_a['risk_level']}
- Flags Triggered: {agent_a['flag_count']}
- Key Issues: {[f.get('id') for f in agent_a.get('triggered_flags', [])[:5]]}

AGENT B (AI Reasoning Auditor):
- Fraud Score: {agent_b['fraud_score']}/100
- Risk Level: {agent_b['risk_level']}
- Risk Indicators: {agent_b.get('risk_indicators', [])}
- Reasoning: {agent_b['reasoning'][:200]}

DISAGREEMENT:
{disagreement['reason']}

As the arbitrator, analyze both perspectives and provide your resolution:

Return JSON:
{{
    "final_fraud_score": 0-100,
    "final_risk_level": "APPROVED|NEEDS_REVIEW|HIGH_RISK|REJECT",
    "decision": "which_agent_to_favor_or_compromise",
    "reasoning": "detailed explanation of your decision",
    "confidence": 0.0-1.0,
    "analysis": "brief analysis of the disagreement"
}}

Consider:
1. Which agent has stronger evidence?
2. Are the rule-based flags legitimate concerns?
3. Does the AI reasoning reveal context the rules missed?
4. What's the appropriate balance between caution and efficiency?

Return ONLY valid JSON."""
    
    def _semantic_to_risk_level(self, semantic_score: float) -> str:
        """Convert semantic fraud score to risk level"""
        if semantic_score < 0.2:
            return "APPROVED"
        elif semantic_score < 0.5:
            return "NEEDS_REVIEW"
        elif semantic_score < 0.7:
            return "HIGH_RISK"
        else:
            return "REJECT"
    
    def _score_to_risk_level(self, fraud_score: float) -> str:
        """Convert fraud score to risk level"""
        if fraud_score < 20:
            return "APPROVED"
        elif fraud_score < 50:
            return "NEEDS_REVIEW"
        elif fraud_score < 80:
            return "HIGH_RISK"
        else:
            return "REJECT"
    
    def _generate_committee_recommendations(
        self,
        agent_a: Dict[str, Any],
        agent_b: Dict[str, Any],
        resolution: Dict[str, Any]
    ) -> List[str]:
        """Generate committee recommendations"""
        recommendations = []
        
        if resolution.get('resolution_needed'):
            recommendations.append(
                f" COMMITTEE RESOLUTION: {resolution.get('decision_basis', 'Resolved')}"
            )
        
        # Add specific recommendations based on final decision
        final_risk = resolution['final_risk_level']
        
        if final_risk == "REJECT":
            recommendations.append(" REJECT: Do not process this invoice")
            recommendations.append("Escalate to compliance team for investigation")
        elif final_risk == "HIGH_RISK":
            recommendations.append(" HIGH RISK: Require senior approval")
            recommendations.append("Request additional documentation from vendor")
        elif final_risk == "NEEDS_REVIEW":
            recommendations.append(" NEEDS REVIEW: Manual verification required")
            recommendations.append("Verify flagged items before processing")
        else:
            recommendations.append(" APPROVED: Standard processing")
        
        # Add agent-specific insights
        if agent_a['flag_count'] > 0:
            recommendations.append(
                f" Rule-based auditor flagged {agent_a['flag_count']} compliance issues"
            )
        
        if agent_b.get('risk_indicators'):
            recommendations.append(
                f" AI auditor identified: {', '.join(agent_b['risk_indicators'][:3])}"
            )
        
        return recommendations


# =============================================================================
# DEMO / CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-AGENT DISAGREEMENT RESOLVER - Demo")
    print("=" * 60)
    
    resolver = MultiAgentDisagreementResolver(resolution_strategy="balanced")
    
    # Sample data with disagreement
    sample_invoice = {
        "vendor_name": "XYZ Corp",
        "invoice_number": "INV-2025-001",
        "amounts": {"total_amount": 100000}
    }
    
    # Agent A (Rule-based) finds issues
    rule_based = {
        "fraud_score": 45,
        "risk_level": "NEEDS_REVIEW",
        "triggered_flags": [
            {"id": "INV_GSTIN_MISSING"},
            {"id": "INV_TAX_MISMATCH"},
            {"id": "INV_ROUND_AMOUNT"}
        ]
    }
    
    # Agent B (AI) is more lenient
    ai_based = {
        "semantic_fraud_score": 0.25,
        "risk_indicators": ["Minor formatting inconsistency"],
        "reasoning": "Invoice appears legitimate despite missing GSTIN. Vendor is known supplier.",
        "confidence": 0.75
    }
    
    # Add semantic analysis to rule-based
    rule_based['semantic_analysis'] = ai_based
    
    # Run committee analysis
    result = resolver.analyze_with_committee(sample_invoice, rule_based)
    
    print(f"\n COMMITTEE DECISION:")
    print(f"Final Fraud Score: {result['committee_decision']['final_fraud_score']}/100")
    print(f"Final Risk Level: {result['committee_decision']['final_risk_level']}")
    print(f"Resolution Needed: {result['committee_decision']['resolution_needed']}")
    print(f"Decision Basis: {result['committee_decision']['decision_basis']}")
    
    print(f"\n AGENT ASSESSMENTS:")
    print(f"Agent A (Rule-Based): {result['agent_a_assessment']['fraud_score']}/100 - {result['agent_a_assessment']['risk_level']}")
    print(f"Agent B (AI Reasoning): {result['agent_b_assessment']['fraud_score']:.1f}/100 - {result['agent_b_assessment']['risk_level']}")
    
    print(f"\n RECOMMENDATIONS:")
    for rec in result['recommendations']:
        print(f"  {rec}")
