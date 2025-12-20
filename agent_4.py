 import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict


TRUST_SCORE_WEIGHTS = {
    "document_integrity": 30,     
    "financial_behavior": 25,     
    "frequency_pattern": 20,      
    "metadata_anomalies": 15,      
    "ai_reasoning": 10             
}

TRUST_SIGNALS = {
    "valid_gstin": 10,
    "digital_signature_present": 8,
    "physical_seal_present": 5,
    "missing_gstin": -15,
    "invalid_gstin_format": -12,
    "missing_signature": -8,
    "missing_seal": -3,
    
    # Financial Behavior (0-25 points)
    "consistent_tax_rates": 10,
    "accurate_calculations": 8,
    "proper_hsn_codes": 7,
    "frequent_rounding": -10,
    "tax_calculation_errors": -12,
    "missing_hsn_codes": -5,
    "suspicious_discounts": -8,
    
    # Frequency Pattern (0-20 points)
    "regular_submission": 10,
    "appropriate_volume": 10,
    "burst_submissions": -15,
    "duplicate_invoices": -20,
    "stale_invoices": -8,
    
    # Metadata Anomalies (0-15 points)
    "recent_pdf_timestamp": 5,
    "standard_software": 5,
    "consistent_format": 5,
    "suspicious_timestamp": -10,
    "unusual_software": -5,
    "format_inconsistency": -8,
    
    # AI Reasoning (0-10 points)
    "ai_approved": 10,
    "ai_minor_concerns": 5,
    "ai_major_concerns": -5,
    "ai_fraud_detected": -15
}


@dataclass
class VendorProfile:
    """Vendor profile with historical data"""
    vendor_id: str
    vendor_name: str
    vendor_gstin: Optional[str]
    first_seen: str
    last_seen: str
    total_invoices: int
    total_amount: float
    trust_score: float
    trust_history: List[Dict[str, Any]]
    flags_triggered: Dict[str, int]  # flag_id -> count
    behavioral_patterns: Dict[str, Any]
    metadata: Dict[str, Any]
    # New E-Invoicing Fields
    aato: float = 0.0
    is_gov_supplier: bool = False
    category: str = "Regular"
    e_invoice_mandate_start: Optional[str] = None


# =============================================================================
# VENDOR TRUST ANALYZER
# =============================================================================

class VendorTrustAnalyzer:
    """
    Vendor Trustworthiness Scoring System
    
    Maintains vendor profiles and calculates dynamic trust scores based on:
    - Document integrity (signatures, GSTIN validity)
    - Financial behavior (tax consistency, calculation accuracy)
    - Frequency patterns (submission velocity, duplicates)
    - Metadata anomalies (PDF timestamps, software)
    - AI reasoning (semantic analysis results)
    """
    
    def __init__(self, vendor_db_path: str = "vendor_profiles.json"):
        """Initialize vendor trust analyzer"""
        self.vendor_db_path = Path(vendor_db_path)
        self.vendors: Dict[str, VendorProfile] = {}
        self._load_vendor_database()
    
    def _load_vendor_database(self) -> None:
        """Load vendor profiles from disk"""
        if self.vendor_db_path.exists():
            try:
                with open(self.vendor_db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for vendor_id, vendor_data in data.items():
                        self.vendors[vendor_id] = VendorProfile(**vendor_data)
                print(f"Loaded {len(self.vendors)} vendor profiles")
            except Exception as e:
                print(f"Error loading vendor database: {e}")
                self.vendors = {}
        else:
            print("Creating new vendor database")
            self.vendors = {}
    
    def _save_vendor_database(self) -> None:
        """Save vendor profiles to disk"""
        try:
            data = {vid: asdict(profile) for vid, profile in self.vendors.items()}
            with open(self.vendor_db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving vendor database: {e}")
    
    def analyze_vendor_trust(
        self,
        invoice_data: Dict[str, Any],
        fraud_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze vendor trustworthiness and update profile
        
        Args:
            invoice_data: Parsed and normalized invoice data
            fraud_analysis: Agent 3 fraud analysis results
            
        Returns:
            Trust score analysis with breakdown
        """
        # Extract vendor info
        vendor_info = self._extract_vendor_info(invoice_data)
        vendor_id = vendor_info['vendor_id']
        
        # Get or create vendor profile
        if vendor_id not in self.vendors:
            profile = self._create_vendor_profile(vendor_info)
        else:
            profile = self.vendors[vendor_id]
        
        # Calculate trust score components
        score_breakdown = {
            "document_integrity": self._score_document_integrity(invoice_data, fraud_analysis),
            "financial_behavior": self._score_financial_behavior(invoice_data, fraud_analysis),
            "frequency_pattern": self._score_frequency_pattern(profile, invoice_data),
            "metadata_anomalies": self._score_metadata(invoice_data),
            "ai_reasoning": self._score_ai_reasoning(fraud_analysis)
        }
        
        # Calculate weighted total score (0-100)
        total_score = sum(
            score_breakdown[category] * (TRUST_SCORE_WEIGHTS[category] / 100)
            for category in score_breakdown
        )
        
        # Normalize to 0-100 range
        trust_score = max(0, min(100, total_score))
        
        # Update vendor profile
        self._update_vendor_profile(profile, invoice_data, fraud_analysis, trust_score, score_breakdown)
        
        # Determine trust level
        trust_level = self._get_trust_level(trust_score)
        
        # Generate recommendations
        recommendations = self._generate_trust_recommendations(trust_score, score_breakdown, profile)
        
        return {
            "vendor_id": vendor_id,
            "vendor_name": vendor_info['vendor_name'],
            "vendor_gstin": vendor_info.get('vendor_gstin'),
            "trust_score": round(trust_score, 2),
            "trust_level": trust_level,
            "score_breakdown": score_breakdown,
            "historical_average": round(profile.trust_score, 2),
            "total_invoices_processed": profile.total_invoices,
            "total_amount_processed": profile.total_amount,
            "first_seen": profile.first_seen,
            "last_seen": profile.last_seen,
            "red_flags": self._get_active_red_flags(profile),
            "recommendations": recommendations,
            "trend": self._calculate_trust_trend(profile)
        }
    
    def _extract_vendor_info(self, invoice_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract vendor identification info"""
        vendor = invoice_data.get('vendor', {})
        
        # Try different possible locations for vendor data
        vendor_name = None
        vendor_gstin = None
        
        if isinstance(vendor, dict):
            vendor_name = vendor.get('name', {}).get('value') if isinstance(vendor.get('name'), dict) else vendor.get('name')
            vendor_gstin = vendor.get('gstin', {}).get('value') if isinstance(vendor.get('gstin'), dict) else vendor.get('gstin')
        else:
            vendor = {}
        
        # Fallback to top-level fields
        if not vendor_name:
            vendor_name = invoice_data.get('vendor_name', 'UNKNOWN_VENDOR')
        if not vendor_gstin:
            vendor_gstin = invoice_data.get('vendor_gstin')
        
        # Create vendor ID (prefer GSTIN, fallback to name hash)
        if vendor_gstin:
            vendor_id = f"GSTIN_{vendor_gstin}"
        else:
            vendor_id = f"NAME_{hash(vendor_name) % 1000000:06d}"
        
        return {
            "vendor_id": vendor_id,
            "vendor_name": vendor_name,
            "vendor_gstin": vendor_gstin
        }
    
    def _create_vendor_profile(self, vendor_info: Dict[str, str]) -> VendorProfile:
        """Create new vendor profile"""
        profile = VendorProfile(
            vendor_id=vendor_info['vendor_id'],
            vendor_name=vendor_info['vendor_name'],
            vendor_gstin=vendor_info.get('vendor_gstin'),
            first_seen=datetime.now().isoformat(),
            last_seen=datetime.now().isoformat(),
            total_invoices=0,
            total_amount=0.0,
            trust_score=50.0,  # Start neutral
            trust_history=[],
            flags_triggered={},
            behavioral_patterns={},
            metadata={}
        )
        self.vendors[vendor_info['vendor_id']] = profile
        return profile
    
    def _score_document_integrity(
        self,
        invoice_data: Dict[str, Any],
        fraud_analysis: Dict[str, Any]
    ) -> float:
        """Score document integrity (0-100)"""
        score = 50  # Start neutral
        
        # Check GSTIN
        vendor = invoice_data.get('vendor') or {}
        vendor_gstin = vendor.get('gstin')
        if vendor_gstin:
            score += TRUST_SIGNALS['valid_gstin']
        else:
            score += TRUST_SIGNALS['missing_gstin']
        
        # Check signatures and seals
        auth = invoice_data.get('authentication') or {}
        if auth.get('has_digital_signature'):
            score += TRUST_SIGNALS['digital_signature_present']
        else:
            score += TRUST_SIGNALS['missing_signature']
        
        if auth.get('has_physical_seal') or auth.get('has_seal'):
            score += TRUST_SIGNALS['physical_seal_present']
        
        # Check for GSTIN-related flags
        flags = fraud_analysis.get('triggered_flags', [])
        for flag in flags:
            if 'GSTIN' in flag.get('id', ''):
                if 'FORMAT' in flag.get('id', ''):
                    score += TRUST_SIGNALS['invalid_gstin_format']
                elif 'MISSING' in flag.get('id', ''):
                    score += TRUST_SIGNALS['missing_gstin']
        
        return max(0, min(100, score))
    
    def _score_financial_behavior(
        self,
        invoice_data: Dict[str, Any],
        fraud_analysis: Dict[str, Any]
    ) -> float:
        """Score financial behavior (0-100)"""
        score = 50  # Start neutral
        
        flags = fraud_analysis.get('triggered_flags', [])
        flag_ids = [f.get('id', '') for f in flags]
        
        # Tax calculation accuracy
        if 'INV_TAX_MISMATCH' not in flag_ids:
            score += TRUST_SIGNALS['consistent_tax_rates']
        else:
            score += TRUST_SIGNALS['tax_calculation_errors']
        
        # Amount accuracy
        if 'INV_AMOUNT_MISMATCH' not in flag_ids:
            score += TRUST_SIGNALS['accurate_calculations']
        else:
            score += TRUST_SIGNALS['tax_calculation_errors']
        
        # HSN codes
        if 'INV_HSN_MISSING' not in flag_ids:
            score += TRUST_SIGNALS['proper_hsn_codes']
        else:
            score += TRUST_SIGNALS['missing_hsn_codes']
        
        # Suspicious patterns
        if 'INV_ROUND_AMOUNT' in flag_ids:
            score += TRUST_SIGNALS['frequent_rounding']
        
        if 'INV_SUSPICIOUS_DISCOUNT' in flag_ids:
            score += TRUST_SIGNALS['suspicious_discounts']
        
        return max(0, min(100, score))
    
    def _score_frequency_pattern(
        self,
        profile: VendorProfile,
        invoice_data: Dict[str, Any]
    ) -> float:
        """Score submission frequency patterns (0-100)"""
        score = 50  # Start neutral
        
        # Check for regular submission pattern
        if profile.total_invoices > 5:
            # Analyze submission velocity
            recent_submissions = [
                h for h in profile.trust_history[-10:]
                if (datetime.now() - datetime.fromisoformat(h['timestamp'])).days <= 30
            ]
            
            if len(recent_submissions) > 0:
                if len(recent_submissions) <= 5:
                    score += TRUST_SIGNALS['regular_submission']
                elif len(recent_submissions) > 10:
                    score += TRUST_SIGNALS['burst_submissions']
        
        # Check for duplicates in flags
        if any('DUPLICATE' in f for f in profile.flags_triggered.keys()):
            score += TRUST_SIGNALS['duplicate_invoices']
        
        # Check for stale invoices
        invoice_date = invoice_data.get('invoice_date')
        if invoice_date:
            try:
                inv_date = datetime.fromisoformat(invoice_date) if isinstance(invoice_date, str) else invoice_date
                days_old = (datetime.now() - inv_date).days
                if days_old > 90:
                    score += TRUST_SIGNALS['stale_invoices']
            except:
                pass
        
        return max(0, min(100, score))
    
    def _score_metadata(self, invoice_data: Dict[str, Any]) -> float:
        """Score metadata anomalies (0-100)"""
        score = 50  # Start neutral
        
        # Check PDF metadata if available
        metadata = invoice_data.get('metadata') or {}
        
        # Timestamp checks
        if metadata.get('creation_date'):
            score += TRUST_SIGNALS['recent_pdf_timestamp']
        
        # Software checks
        if metadata.get('producer') or metadata.get('creator'):
            score += TRUST_SIGNALS['standard_software']
        
        # Format consistency
        if invoice_data.get('confidence_score', 0) > 0.8:
            score += TRUST_SIGNALS['consistent_format']
        
        return max(0, min(100, score))
    
    def _score_ai_reasoning(self, fraud_analysis: Dict[str, Any]) -> float:
        """Score based on AI reasoning (0-100)"""
        score = 50  # Start neutral
        
        # Check semantic analysis if available
        semantic = fraud_analysis.get('semantic_analysis') or {}
        if semantic:
            semantic_score = semantic.get('semantic_fraud_score', 0)
            if semantic_score < 0.3:
                score += TRUST_SIGNALS['ai_approved']
            elif semantic_score < 0.5:
                score += TRUST_SIGNALS['ai_minor_concerns']
            elif semantic_score < 0.7:
                score += TRUST_SIGNALS['ai_major_concerns']
            else:
                score += TRUST_SIGNALS['ai_fraud_detected']
        
        # Check AI validation status
        ai_signals = fraud_analysis.get('ai_signals') or {}
        ai_status = ai_signals.get('status')
        if ai_status == 'VALID':
            score += TRUST_SIGNALS['ai_approved']
        elif ai_status == 'INVALID':
            score += TRUST_SIGNALS['ai_major_concerns']
        
        return max(0, min(100, score))
    
    def _update_vendor_profile(
        self,
        profile: VendorProfile,
        invoice_data: Dict[str, Any],
        fraud_analysis: Dict[str, Any],
        trust_score: float,
        score_breakdown: Dict[str, float]
    ) -> None:
        """Update vendor profile with new invoice data"""
        # Update basic stats
        profile.total_invoices += 1
        profile.last_seen = datetime.now().isoformat()
        
        # Update total amount
        amounts = invoice_data.get('amounts') or {}
        total = amounts.get('total_amount', 0)
        if isinstance(total, dict):
            total = total.get('value', 0)
        profile.total_amount += float(total)
        
        # Update trust score (moving average)
        if profile.total_invoices == 1:
            profile.trust_score = trust_score
        else:
            # Weighted average: 70% historical, 30% current
            profile.trust_score = (profile.trust_score * 0.7) + (trust_score * 0.3)
        
        # Add to trust history
        profile.trust_history.append({
            "timestamp": datetime.now().isoformat(),
            "trust_score": trust_score,
            "fraud_score": fraud_analysis.get('fraud_score', 0),
            "score_breakdown": score_breakdown
        })
        
        # Keep only last 50 entries
        profile.trust_history = profile.trust_history[-50:]
        
        # Update flags triggered
        for flag in fraud_analysis.get('triggered_flags', []):
            flag_id = flag.get('id', 'UNKNOWN')
            profile.flags_triggered[flag_id] = profile.flags_triggered.get(flag_id, 0) + 1
        
        # Save to disk
        self._save_vendor_database()
    
    def _get_trust_level(self, trust_score: float) -> str:
        """Convert numeric score to trust level"""
        if trust_score >= 80:
            return "HIGHLY_TRUSTED"
        elif trust_score >= 60:
            return "TRUSTED"
        elif trust_score >= 40:
            return "NEUTRAL"
        elif trust_score >= 20:
            return "CAUTION"
        else:
            return "HIGH_RISK"
    
    def _get_active_red_flags(self, profile: VendorProfile) -> List[str]:
        """Get list of frequently triggered flags"""
        red_flags = []
        for flag_id, count in profile.flags_triggered.items():
            if count >= 3:  # Triggered 3+ times
                red_flags.append(f"{flag_id} (triggered {count}x)")
        return red_flags
    
    def _generate_trust_recommendations(
        self,
        trust_score: float,
        score_breakdown: Dict[str, float],
        profile: VendorProfile
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if trust_score < 40:
            recommendations.append(" HIGH RISK: Require additional verification for all invoices")
            recommendations.append("Consider vendor audit or contract review")
        elif trust_score < 60:
            recommendations.append(" CAUTION: Implement enhanced monitoring")
        
        # Category-specific recommendations
        if score_breakdown['document_integrity'] < 40:
            recommendations.append(" Request properly signed and sealed invoices")
        
        if score_breakdown['financial_behavior'] < 40:
            recommendations.append(" Verify tax calculations and request itemized breakdowns")
        
        if score_breakdown['frequency_pattern'] < 40:
            recommendations.append(" Investigate unusual submission patterns")
        
        if len(profile.flags_triggered) > 5:
            recommendations.append(" Multiple compliance issues detected - escalate to procurement")
        
        if not recommendations:
            recommendations.append(" Vendor shows consistent compliance - standard processing approved")
        
        return recommendations
    
    def _calculate_trust_trend(self, profile: VendorProfile) -> str:
        """Calculate trust score trend"""
        if len(profile.trust_history) < 2:
            return "NEW_VENDOR"
        
        recent_scores = [h['trust_score'] for h in profile.trust_history[-5:]]
        if len(recent_scores) < 2:
            return "STABLE"
        
        avg_recent = sum(recent_scores[-3:]) / 3
        avg_older = sum(recent_scores[:-3]) / max(1, len(recent_scores) - 3)
        
        diff = avg_recent - avg_older
        
        if diff > 10:
            return "IMPROVING"
        elif diff < -10:
            return "DECLINING"
        else:
            return "STABLE"
    
    def get_vendor_profile(self, vendor_id: str) -> Optional[Dict[str, Any]]:
        """Get vendor profile by ID"""
        if vendor_id in self.vendors:
            return asdict(self.vendors[vendor_id])
        return None
    
    def get_all_vendors(self) -> List[Dict[str, Any]]:
        """Get all vendor profiles"""
        return [asdict(profile) for profile in self.vendors.values()]


# =============================================================================
# DEMO / CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VENDOR TRUST ANALYZER - Demo")
    print("=" * 60)
    
    analyzer = VendorTrustAnalyzer()
    
    # Sample invoice data
    sample_invoice = {
        "vendor": {
            "name": "ABC Corp",
            "gstin": "29ABCDE1234F1Z5"
        },
        "amounts": {
            "total_amount": 50000
        },
        "authentication": {
            "has_digital_signature": True,
            "has_seal": True
        },
        "invoice_date": "2025-12-01"
    }
    
    sample_fraud_analysis = {
        "fraud_score": 25,
        "triggered_flags": [],
        "ai_signals": {"status": "VALID"}
    }
    
    result = analyzer.analyze_vendor_trust(sample_invoice, sample_fraud_analysis)
    
    print(f"\n Vendor Trust Analysis:")
    print(f"Vendor: {result['vendor_name']}")
    print(f"Trust Score: {result['trust_score']}/100 ({result['trust_level']})")
    print(f"\nScore Breakdown:")
    for category, score in result['score_breakdown'].items():
        print(f"   {category}: {score:.1f}/100")
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  {rec}")
