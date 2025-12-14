import json
import os
from typing import Dict, Any, List, Tuple, Optional

from rules import invoice_rules
from gemini_analyzer import GeminiAnalyzer


CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'risk_weights.json')

def load_config() -> Dict[str, Any]:
    """Load risk weights configuration with fallback defaults."""
    default_config = {
        "severity_weights": {
            "critical": 3.0,
            "high": 2.0,
            "medium": 1.0,
            "low": 0.5,
            "ai": 3.0
        },
        "max_weight_per_flag": 3.0,
        "score_blending": {
            "deterministic_weight": 0.7,
            "semantic_weight": 0.3
        },
        "risk_thresholds": {
            "safe": 20,
            "needs_review": 50,
            "high_risk": 75
        }
    }
    
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
            # Merge with defaults
            for key, value in default_config.items():
                if key not in loaded:
                    loaded[key] = value
            return loaded
    except FileNotFoundError:
        print(f"⚠️  Config not found at {CONFIG_PATH}, using defaults")
        return default_config
    except json.JSONDecodeError as e:
        print(f"⚠️  Config parse error: {e}, using defaults")
        return default_config


RISK_CONFIG = load_config()

# Invoice-specific flags collection
INVOICE_FLAGS = getattr(invoice_rules, 'ALL_FLAGS', [])



class InvoiceFraudAnalyzer:
    SUPPORTED_DOC_TYPE = "INVOICE"

    def __init__(
        self,
        history_index: Optional[Dict[str, Any]] = None,
        use_gemini: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Invoice Fraud Analyzer.
        
        Args:
            history_index: Historical data for duplicate/pattern detection (future use)
            use_gemini: Enable Gemini-powered semantic analysis
            config: Override default risk configuration
        """
        self.history = history_index or {}
        self.config = config or RISK_CONFIG
        
        # Extract config values
        self.severity_weights = self.config.get('severity_weights', {})
        self.max_weight_per_flag = self.config.get('max_weight_per_flag', 3.0)
        self.score_blending = self.config.get('score_blending', {})
        self.risk_thresholds = self.config.get('risk_thresholds', {})
        
        # Pre-compute max possible score for normalization
        self._max_score = self._compute_max_score()
        
        # Initialize Gemini analyzer if requested
        self.use_gemini = use_gemini
        self.gemini: Optional[GeminiAnalyzer] = None
        if use_gemini:
            self._init_gemini()

    def _init_gemini(self) -> None:
        """Initialize Gemini semantic analyzer with error handling."""
        try:
            self.gemini = GeminiAnalyzer()
            print("✅ Gemini semantic analyzer enabled for invoice analysis")
        except Exception as e:
            print(f"⚠️  Gemini initialization failed: {e}")
            self.use_gemini = False
            self.gemini = None

    def _compute_max_score(self) -> float:
        """Compute maximum possible weight from all invoice flags."""
        if not INVOICE_FLAGS:
            return self.max_weight_per_flag
        
        max_weight = sum(
            self.severity_weights.get(f.get('severity', 'medium'), 1.0)
            for f in INVOICE_FLAGS
        )
        return max(max_weight, self.max_weight_per_flag)

    def analyze(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an invoice document for fraud and compliance risks.
        
        Args:
            document: Normalized invoice document from Agent2
            
        Returns:
            Comprehensive risk assessment dictionary containing:
            - fraud_score: 0-100 risk score
            - risk_level: Human-readable risk classification
            - triggered_flags: List of detected issues
            - ai_signals: AI validation results (if present)
            - semantic_analysis: Gemini analysis (if enabled)
            - details: Score computation breakdown
        """
        # Validate document type
        validation_result = self._validate_document_type(document)
        if validation_result:
            return validation_result
        
        # Extract document data
        doc_data = document.get('document', document)
        
        # Step 1: Run deterministic rule-based detection
        flagged = self._detect_rule_based_flags(document)
        
        # Step 2: Extract and process AI validation signals
        ai_signals = self._extract_ai_signals(document)
        ai_flags = self._process_ai_signals(ai_signals)
        flagged.extend(ai_flags)
        
        # Step 3: Compute base fraud score
        base_score, score_details = self._compute_fraud_score(flagged)
        
        # Step 4: Optional semantic analysis via Gemini
        semantic_analysis = None
        final_score = base_score
        
        if self.use_gemini and self.gemini:
            semantic_analysis, final_score = self._apply_semantic_analysis(
                doc_data, base_score, score_details
            )
        
        # Step 5: Build comprehensive result
        return self._build_result(
            fraud_score=final_score,
            triggered_flags=flagged,
            ai_signals=ai_signals,
            semantic_analysis=semantic_analysis,
            details=score_details
        )

    def _validate_document_type(self, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate that the document is an invoice. Returns error dict if not."""
        doc_type = self._extract_document_type(document)
        
        if not doc_type:
            return {
                "error": "Document type not specified",
                "supported_type": self.SUPPORTED_DOC_TYPE,
                "fraud_score": None,
                "risk_level": "UNKNOWN"
            }
        
        if doc_type != self.SUPPORTED_DOC_TYPE:
            return {
                "error": f"Unsupported document type: {doc_type}",
                "supported_type": self.SUPPORTED_DOC_TYPE,
                "fraud_score": None,
                "risk_level": "UNSUPPORTED"
            }
        
        return None  # Validation passed

    def _extract_document_type(self, document: Dict[str, Any]) -> Optional[str]:
        """Extract and normalize document type from various possible locations."""
        doc_type = (
            document.get('document', {}).get('document_type') or
            document.get('document_type')
        )
        
        if isinstance(doc_type, str):
            return doc_type.upper().strip()
        return None

    def _detect_rule_based_flags(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run invoice-specific rule-based fraud detection."""
        try:
            return invoice_rules.detect_flags(document)
        except Exception as e:
            print(f"⚠️  Rule detection error: {e}")
            return [{
                "id": "RULE_DETECTION_ERROR",
                "description": f"Error during rule-based detection: {str(e)}",
                "severity": "medium",
                "evidence": {"error": str(e)}
            }]

    def _extract_ai_signals(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract AI validation signals from various possible locations."""
        ai_signals = (
            document.get('ai_validation') or
            document.get('validation', {}).get('ai_validation') or
            document.get('document', {}).get('ai_validation') or
            {}
        )
        return ai_signals if isinstance(ai_signals, dict) else {}

    def _process_ai_signals(self, ai_signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert AI validation signals into fraud flags."""
        flags = []
        
        if not ai_signals:
            return flags
        
        status = ai_signals.get('status', '').upper()
        checks = ai_signals.get('checks', {})
        errors = ai_signals.get('errors', [])
        warnings = ai_signals.get('warnings', [])
        
        # Flag: AI reports INVALID status
        if status == 'INVALID':
            flags.append({
                "id": "AI_INVALID_STATUS",
                "description": "AI validation marked document as INVALID",
                "severity": "ai",
                "category": "ai_validation",
                "evidence": {
                    "status": status,
                    "errors": errors[:5],  # Limit evidence size
                    "error_count": len(errors)
                }
            })
        
        # Flag: Format validation failed
        if checks.get('format_validation') is False:
            flags.append({
                "id": "AI_FORMAT_INVALID",
                "description": "AI detected format validation failures",
                "severity": "ai",
                "category": "ai_validation",
                "evidence": {
                    "check": "format_validation",
                    "result": False,
                    "related_errors": [e for e in errors if 'format' in str(e).lower()][:3]
                }
            })
        
        # Flag: Financial consistency check failed
        if checks.get('financial_consistency') is False:
            flags.append({
                "id": "AI_FINANCIAL_INCONSISTENCY",
                "description": "AI detected financial calculation inconsistencies",
                "severity": "ai",
                "category": "ai_validation",
                "evidence": {
                    "check": "financial_consistency",
                    "result": False,
                    "related_errors": [e for e in errors if 'amount' in str(e).lower() or 'total' in str(e).lower()][:3]
                }
            })
        
        # Flag: Required fields missing (AI detected)
        if checks.get('required_fields') is False:
            flags.append({
                "id": "AI_MISSING_REQUIRED_FIELDS",
                "description": "AI detected missing required invoice fields",
                "severity": "high",
                "category": "ai_validation",
                "evidence": {
                    "check": "required_fields",
                    "result": False
                }
            })
        
        # Flag: Suspicious patterns detected by AI
        if checks.get('suspicious_patterns') is True:
            flags.append({
                "id": "AI_SUSPICIOUS_PATTERNS",
                "description": "AI detected suspicious patterns in invoice data",
                "severity": "high",
                "category": "ai_validation",
                "evidence": {
                    "check": "suspicious_patterns",
                    "warnings": warnings[:5]
                }
            })
        
        return flags

    def _compute_fraud_score(
        self,
        flags: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute normalized fraud score (0-100) from triggered flags.
        
        Args:
            flags: List of triggered fraud flags
            
        Returns:
            Tuple of (score, details_dict)
        """
        if not flags:
            return 0.0, {
                "total_weight": 0.0,
                "max_possible": self._max_score,
                "flag_count": 0,
                "weight_breakdown": {}
            }
        
        # Calculate weights per severity
        weight_breakdown = {}
        total_weight = 0.0
        
        for flag in flags:
            severity = flag.get('severity', 'medium')
            weight = self.severity_weights.get(severity, 1.0)
            total_weight += weight
            
            # Track breakdown
            if severity not in weight_breakdown:
                weight_breakdown[severity] = {"count": 0, "weight": 0.0}
            weight_breakdown[severity]["count"] += 1
            weight_breakdown[severity]["weight"] += weight
        
        # Normalize to 0-100 scale
        if self._max_score <= 0:
            fraud_score = 0.0
        else:
            fraud_score = min((total_weight / self._max_score) * 100.0, 100.0)
        
        details = {
            "total_weight": round(total_weight, 2),
            "max_possible": round(self._max_score, 2),
            "flag_count": len(flags),
            "weight_breakdown": weight_breakdown,
            "normalization_factor": round(self._max_score, 2)
        }
        
        return round(fraud_score, 2), details

    def _apply_semantic_analysis(
        self,
        doc_data: Dict[str, Any],
        base_score: float,
        score_details: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Apply Gemini semantic analysis and blend with deterministic score.
        
        Args:
            doc_data: Invoice document data
            base_score: Deterministic fraud score
            score_details: Details dict to update
            
        Returns:
            Tuple of (semantic_analysis_result, blended_score)
        """
        try:
            semantic_analysis = self.gemini.analyze(self.SUPPORTED_DOC_TYPE, doc_data)
            
            if semantic_analysis:
                semantic_score = semantic_analysis.get('semantic_fraud_score', 0.0) * 100
                
                # Blend scores according to config
                det_weight = self.score_blending.get('deterministic_weight', 0.7)
                sem_weight = self.score_blending.get('semantic_weight', 0.3)
                
                blended_score = (base_score * det_weight) + (semantic_score * sem_weight)
                blended_score = round(min(blended_score, 100.0), 2)
                
                # Update details
                score_details['semantic_score'] = round(semantic_score, 2)
                score_details['base_deterministic_score'] = base_score
                score_details['blending_weights'] = {
                    'deterministic': det_weight,
                    'semantic': sem_weight
                }
                
                return semantic_analysis, blended_score
                
        except Exception as e:
            print(f"⚠️  Semantic analysis failed: {e}")
            score_details['semantic_error'] = str(e)
        
        return None, base_score

    def _determine_risk_level(self, score: float) -> str:
        """Map fraud score to human-readable risk level."""
        thresholds = self.risk_thresholds
        
        if score <= thresholds.get('safe', 20):
            return 'Safe'
        elif score <= thresholds.get('needs_review', 50):
            return 'Needs Review'
        elif score <= thresholds.get('high_risk', 75):
            return 'High Risk'
        else:
            return 'Likely Fraud / Reject'

    def _build_result(
        self,
        fraud_score: float,
        triggered_flags: List[Dict[str, Any]],
        ai_signals: Dict[str, Any],
        semantic_analysis: Optional[Dict[str, Any]],
        details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the final analysis result dictionary."""
        
        # Categorize flags
        flags_by_severity = {}
        flags_by_category = {}
        
        for flag in triggered_flags:
            # By severity
            sev = flag.get('severity', 'medium')
            if sev not in flags_by_severity:
                flags_by_severity[sev] = []
            flags_by_severity[sev].append(flag['id'])
            
            # By category
            cat = flag.get('category', 'general')
            if cat not in flags_by_category:
                flags_by_category[cat] = []
            flags_by_category[cat].append(flag['id'])
        
        return {
            "document_type": self.SUPPORTED_DOC_TYPE,
            "fraud_score": fraud_score,
            "risk_level": self._determine_risk_level(fraud_score),
            "summary": {
                "total_flags": len(triggered_flags),
                "critical_flags": len(flags_by_severity.get('critical', [])),
                "high_flags": len(flags_by_severity.get('high', [])),
                "ai_flags": len(flags_by_severity.get('ai', [])),
                "flags_by_category": flags_by_category
            },
            "triggered_flags": triggered_flags,
            "ai_signals": ai_signals if ai_signals else None,
            "semantic_analysis": semantic_analysis,
            "details": details,
            "recommendations": self._generate_recommendations(fraud_score, triggered_flags)
        }

    def _generate_recommendations(
        self,
        score: float,
        flags: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []
        
        risk_level = self._determine_risk_level(score)
        
        if risk_level == 'Safe':
            recommendations.append("Invoice appears compliant. Standard processing recommended.")
        elif risk_level == 'Needs Review':
            recommendations.append("Manual review recommended before processing.")
        elif risk_level == 'High Risk':
            recommendations.append("Detailed investigation required. Escalate to compliance team.")
        else:
            recommendations.append("REJECT: High fraud probability. Do not process without senior approval.")
        
        # Flag-specific recommendations
        flag_ids = {f['id'] for f in flags}
        
        if 'MISSING_GSTIN' in flag_ids or 'AI_MISSING_REQUIRED_FIELDS' in flag_ids:
            recommendations.append("Request vendor to provide valid GSTIN for GST compliance.")
        
        if 'TAX_CALCULATION_MISMATCH' in flag_ids or 'AI_FINANCIAL_INCONSISTENCY' in flag_ids:
            recommendations.append("Verify tax calculations manually. Cross-check with rate schedules.")
        
        if 'MISSING_SIGNATURE' in flag_ids:
            recommendations.append("Obtain digitally signed invoice copy for audit trail.")
        
        if 'ROUND_AMOUNT_SUSPICIOUS' in flag_ids:
            recommendations.append("Verify pricing against contract or purchase order.")
        
        if 'AI_SUSPICIOUS_PATTERNS' in flag_ids:
            recommendations.append("Cross-reference with historical vendor transactions.")
        
        return recommendations


# =============================================================================
# BACKWARD COMPATIBILITY ALIAS
# =============================================================================

# Alias for backward compatibility with existing code expecting 'Agent3'
Agent3 = InvoiceFraudAnalyzer


# =============================================================================
# DEMO / CLI RUNNER
# =============================================================================

def create_sample_invoice() -> Dict[str, Any]:
    """Create a sample invoice for testing."""
    return {
        "document": {
            "document_id": "INV-2024-001234",
            "document_type": "INVOICE",
            "invoice_number": "INV-2024-001234",
            "invoice_date": "2024-01-15",
            "due_date": "2024-02-15",
            "vendor": {
                "name": "ABC Suppliers Pvt Ltd",
                "gstin": "29AABCT2345F1Z3",
                "pan": "AABCT2345F",
                "address": "123 Business Park, Bangalore"
            },
            "buyer": {
                "name": "XYZ Corporation",
                "gstin": "29XYZC9876G1Z5",
                "address": "456 Corporate Tower, Mumbai"
            },
            "amounts": {
                "subtotal": 10000.00,
                "taxable_amount": 10000.00,
                "cgst": 900.00,
                "sgst": 900.00,
                "igst": 0.00,
                "tax_amount": 1800.00,
                "total_amount": 11800.00,
                "discount": 0.00
            },
            "line_items": [
                {
                    "description": "Office Supplies - Paper A4",
                    "hsn_code": "4802",
                    "quantity": 100,
                    "unit": "reams",
                    "unit_price": 50.00,
                    "total_amount": 5000.00,
                    "tax_rate": 18.0
                },
                {
                    "description": "Printer Cartridges",
                    "hsn_code": "8443",
                    "quantity": 10,
                    "unit": "pieces",
                    "unit_price": 500.00,
                    "total_amount": 5000.00,
                    "tax_rate": 18.0
                }
            ],
            "authentication": {
                "has_digital_signature": True,
                "has_seal": True,
                "signature_valid": True
            },
            "ai_validation": {
                "status": "VALID",
                "checks": {
                    "format_validation": True,
                    "financial_consistency": True,
                    "required_fields": True
                },
                "errors": [],
                "warnings": []
            }
        }
    }


def create_suspicious_invoice() -> Dict[str, Any]:
    """Create a suspicious invoice for testing fraud detection."""
    return {
        "document": {
            "document_id": "INV-SUSPECT-999",
            "document_type": "INVOICE",
            "invoice_number": "INV-SUSPECT-999",
            "invoice_date": "2024-01-20",
            "vendor": {
                "name": "Suspicious Vendor LLC",
                "gstin": None,  # Missing GSTIN
                "pan": None     # Missing PAN
            },
            "buyer": {
                "name": "Target Company",
                "gstin": "29TARG1234X1Z9"
            },
            "amounts": {
                "subtotal": 100000.00,  # Round number
                "taxable_amount": 100000.00,
                "tax_amount": 15000.00,  # Wrong tax (should be 18000 at 18%)
                "total_amount": 115000.00
            },
            "line_items": [
                {
                    "description": "Consulting Services",
                    "quantity": 1,
                    "total_amount": 100000.00
                }
            ],
            "authentication": {
                "has_digital_signature": False,  # Missing signature
                "has_seal": False
            },
            "ai_validation": {
                "status": "INVALID",
                "checks": {
                    "format_validation": False,
                    "financial_consistency": False,
                    "required_fields": False
                },
                "errors": [
                    "Missing vendor GSTIN",
                    "Tax calculation mismatch",
                    "Missing digital signature"
                ],
                "warnings": ["Unusually round amounts detected"]
            }
        }
    }


def print_analysis_report(report: Dict[str, Any], title: str = "INVOICE FRAUD ANALYSIS") -> None:
    """Pretty-print an analysis report."""
    print("\n" + "=" * 70)
    print(f"📊 {title}")
    print("=" * 70)
    
    # Check for errors
    if report.get('error'):
        print(f"\n❌ ERROR: {report['error']}")
        return
    
    # Main metrics
    print(f"\n📄 Document Type: {report['document_type']}")
    print(f"🎯 Fraud Score:   {report['fraud_score']:.2f} / 100")
    print(f"⚠️  Risk Level:    {report['risk_level']}")
    
    # Summary
    summary = report.get('summary', {})
    print(f"\n📈 FLAG SUMMARY:")
    print(f"   Total Flags:    {summary.get('total_flags', 0)}")
    print(f"   Critical:       {summary.get('critical_flags', 0)}")
    print(f"   High:           {summary.get('high_flags', 0)}")
    print(f"   AI-Detected:    {summary.get('ai_flags', 0)}")
    
    # Triggered flags
    flags = report.get('triggered_flags', [])
    if flags:
        print(f"\n🚩 TRIGGERED FLAGS ({len(flags)}):")
        for flag in flags[:10]:  # Show first 10
            severity = flag.get('severity', 'unknown').upper()
            print(f"   [{severity:8}] {flag['id']}")
            print(f"              └─ {flag['description']}")
    else:
        print("\n✅ No fraud flags triggered")
    
    # Semantic analysis
    semantic = report.get('semantic_analysis')
    if semantic:
        print(f"\n🧠 SEMANTIC ANALYSIS (Gemini):")
        print(f"   Score: {semantic.get('semantic_fraud_score', 0) * 100:.1f} / 100")
        indicators = semantic.get('risk_indicators', [])
        if indicators:
            print("   Risk Indicators:")
            for ind in indicators[:5]:
                print(f"      • {ind}")
    
    # Recommendations
    recommendations = report.get('recommendations', [])
    if recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Score details
    details = report.get('details', {})
    print(f"\n📐 SCORE COMPUTATION:")
    print(f"   Total Weight:     {details.get('total_weight', 0)}")
    print(f"   Max Possible:     {details.get('max_possible', 0)}")
    if 'semantic_score' in details:
        print(f"   Semantic Score:   {details.get('semantic_score', 0)}")
        print(f"   Base Score:       {details.get('base_deterministic_score', 0)}")
    
    print("\n" + "=" * 70)


def main():
    """Main demo function."""
    print("\n" + "=" * 70)
    print("🔍 INVOICE FRAUD ANALYZER - DEMO")
    print("=" * 70)
    
    # Initialize analyzer (set use_gemini=True to enable semantic analysis)
    analyzer = InvoiceFraudAnalyzer(use_gemini=False)
    
    # Test 1: Clean invoice
    print("\n\n📋 TEST 1: Analyzing CLEAN invoice...")
    clean_invoice = create_sample_invoice()
    clean_result = analyzer.analyze(clean_invoice)
    print_analysis_report(clean_result, "CLEAN INVOICE ANALYSIS")
    
    # Test 2: Suspicious invoice
    print("\n\n📋 TEST 2: Analyzing SUSPICIOUS invoice...")
    suspicious_invoice = create_suspicious_invoice()
    suspicious_result = analyzer.analyze(suspicious_invoice)
    print_analysis_report(suspicious_result, "SUSPICIOUS INVOICE ANALYSIS")
    
    # Test 3: Invalid document type
    print("\n\n📋 TEST 3: Testing invalid document type...")
    invalid_doc = {"document": {"document_type": "CONTRACT"}}
    invalid_result = analyzer.analyze(invalid_doc)
    print_analysis_report(invalid_result, "INVALID DOCUMENT TYPE")
    
    # Print full JSON for suspicious invoice
    print("\n\n📄 FULL JSON OUTPUT (Suspicious Invoice):")
    print("-" * 70)
    print(json.dumps(suspicious_result, indent=2, default=str))


if __name__ == '__main__':
    main()