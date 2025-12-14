import json
import os
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

REPORT_CONFIG = {
    "risk_levels": {
        "safe": {"label": "APPROVED", "icon": "✅", "action": "proceed"},
        "needs_review": {"label": "NEEDS_REVIEW", "icon": "⚠️", "action": "manual_review"},
        "high_risk": {"label": "HIGH_RISK", "icon": "🔶", "action": "escalate"},
        "reject": {"label": "REJECT", "icon": "❌", "action": "reject"}
    },
    "severity_icons": {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢",
        "ai": "🤖"
    },
    "gemini_model": "gemini-2.5-flash",
    "narrative_max_tokens": 1000,
    "narrative_temperature": 0.5
}

INVOICE_FLAG_EXPLANATIONS = {
    "INV_AMOUNT_MISMATCH": {
        "title": "AMOUNT MISMATCH",
        "icon": "💰",
        "description": "The line items total does not match the final invoice total.",
        "implication": "This discrepancy must be explained—it could be due to rounding, missing items, or calculation errors.",
        "action": "Request an itemized breakdown from the vendor. Reconcile all line items with the total."
    },
    "INV_GSTIN_FORMAT": {
        "title": "INVALID GSTIN FORMAT",
        "icon": "🔏",
        "description": "The GSTIN provided does not match the 15-character government standard.",
        "implication": "Invalid GSTIN may indicate unregistered vendor or data entry error.",
        "action": "Verify vendor GSTIN with the official GSTIN database (gstin.gov.in)."
    },
    "INV_GSTIN_MISSING": {
        "title": "MISSING GSTIN",
        "icon": "🔏",
        "description": "Vendor GSTIN is not provided on the invoice.",
        "implication": "Missing GSTIN violates GST compliance requirements for B2B transactions.",
        "action": "Request vendor to provide valid GSTIN. Do not process without GST details."
    },
    "INV_PAN_MISSING": {
        "title": "MISSING PAN",
        "icon": "📋",
        "description": "Vendor PAN (Permanent Account Number) is not provided.",
        "implication": "PAN is required for TDS compliance and vendor verification.",
        "action": "Request vendor PAN for TDS deduction purposes."
    },
    "INV_DUPLICATE_INVOICE": {
        "title": "DUPLICATE INVOICE",
        "icon": "⚠️",
        "description": "This invoice number has been submitted previously.",
        "implication": "Could indicate duplicate payment request or system error.",
        "action": "Verify this is not a duplicate payment request. Check payment history."
    },
    "INV_SUSPICIOUS_DISCOUNT": {
        "title": "UNUSUAL DISCOUNT",
        "icon": "💸",
        "description": "The discount applied appears unusually high for this transaction.",
        "implication": "May indicate unauthorized discounting or pricing manipulation.",
        "action": "Verify the discount is justified and has proper approval."
    },
    "INV_ROUND_AMOUNT": {
        "title": "SUSPICIOUSLY ROUND AMOUNT",
        "icon": "🎯",
        "description": "Invoice total is an unusually round number.",
        "implication": "Round amounts can indicate estimated billing rather than actual charges.",
        "action": "Request detailed breakdown. Verify against purchase order or contract."
    },
    "INV_TAX_MISMATCH": {
        "title": "TAX CALCULATION ERROR",
        "icon": "📊",
        "description": "Tax amount does not match expected calculation based on taxable amount.",
        "implication": "Incorrect tax collection affects GST filing and input credit.",
        "action": "Verify tax rates applied. Cross-check with HSN/SAC codes."
    },
    "INV_MISSING_SIGNATURE": {
        "title": "MISSING DIGITAL SIGNATURE",
        "icon": "📝",
        "description": "Invoice lacks a digital signature.",
        "implication": "Authenticity cannot be verified. Required for e-invoicing compliance.",
        "action": "Request digitally signed invoice from authorized signatory."
    },
    "INV_MISSING_SEAL": {
        "title": "MISSING PHYSICAL SEAL",
        "icon": "🔒",
        "description": "Invoice has no official seal from the vendor.",
        "implication": "Reduces document authenticity verification.",
        "action": "Request properly sealed document if organization policy requires it."
    },
    "INV_FUTURE_DATE": {
        "title": "FUTURE DATED INVOICE",
        "icon": "📅",
        "description": "Invoice date is in the future.",
        "implication": "Potential data entry error or attempt to manipulate accounting period.",
        "action": "Verify correct invoice date with vendor."
    },
    "INV_STALE_INVOICE": {
        "title": "STALE INVOICE",
        "icon": "📅",
        "description": "Invoice is unusually old.",
        "implication": "May affect GST input credit eligibility. Raises questions about delayed submission.",
        "action": "Verify reason for delayed submission. Check ITC eligibility."
    },
    "INV_HSN_MISSING": {
        "title": "MISSING HSN/SAC CODES",
        "icon": "🏷️",
        "description": "Line items missing HSN/SAC classification codes.",
        "implication": "Required for GST compliance and proper tax rate verification.",
        "action": "Request vendor to provide HSN/SAC codes for all line items."
    },
    "AI_INVALID_STATUS": {
        "title": "AI VALIDATION FAILED",
        "icon": "🤖",
        "description": "Automated validation systems flagged this document as invalid.",
        "implication": "There may be format, extraction, or consistency issues.",
        "action": "Review AI validation errors. Verify document quality and completeness."
    },
    "AI_FORMAT_INVALID": {
        "title": "AI FORMAT VALIDATION FAILED",
        "icon": "🤖",
        "description": "Document format does not match expected invoice structure.",
        "implication": "May indicate non-standard invoice or extraction errors.",
        "action": "Verify document is a valid invoice. Re-process if needed."
    },
    "AI_FINANCIAL_INCONSISTENCY": {
        "title": "AI FINANCIAL INCONSISTENCY",
        "icon": "🤖",
        "description": "AI detected that totals and calculations don't match.",
        "implication": "Could indicate extraction errors or genuine calculation problems.",
        "action": "Manually verify all amounts against source document."
    },
    "AI_MISSING_REQUIRED_FIELDS": {
        "title": "AI MISSING FIELDS",
        "icon": "🤖",
        "description": "AI detected missing required invoice fields.",
        "implication": "Incomplete invoice may not meet compliance requirements.",
        "action": "Request complete invoice with all mandatory fields."
    },
    "AI_SUSPICIOUS_PATTERNS": {
        "title": "AI SUSPICIOUS PATTERNS",
        "icon": "🤖",
        "description": "AI detected suspicious patterns in invoice data.",
        "implication": "Patterns may indicate fraud or manipulation.",
        "action": "Detailed manual review required. Cross-reference with historical data."
    }
}


class InvoiceAuditReportGenerator:
    """
    Invoice-Focused Audit Report Generator
    
    Generates comprehensive audit reports for invoice documents including:
    - Executive summary with risk assessment
    - Key invoice data extraction
    - Validation findings explanation
    - Issue identification and implications
    - Actionable recommendations
    - AI-powered narrative generation (optional)
    
    Usage:
        generator = InvoiceAuditReportGenerator(use_gemini=True)
        report = generator.generate_audit_report(agent3_output)
        generator.export_report(report, format='markdown')
    """

    SUPPORTED_DOC_TYPE = "INVOICE"

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_gemini: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Invoice Audit Report Generator.
        
        Args:
            api_key: Gemini API key (defaults to GOOGLE_API_KEY env var)
            use_gemini: Enable AI-powered narrative generation
            config: Override default report configuration
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.use_gemini = use_gemini
        self.config = config or REPORT_CONFIG
        self.client = None
        
        if self.use_gemini:
            self._init_gemini()

    def _init_gemini(self) -> None:
        """Initialize Gemini client with error handling."""
        if not self.api_key:
            print("⚠️  Gemini API key not found. Falling back to template-based reports.")
            self.use_gemini = False
            return
        
        try:
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = self.config.get('gemini_model', 'gemini-2.5-flash')
            print("✅ Gemini narrative generator enabled for invoice reports")
        except Exception as e:
            print(f"⚠️  Gemini initialization failed: {e}")
            self.use_gemini = False
            self.client = None

    def generate_audit_report(self, agent3_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive audit report from Agent3 fraud analysis output.
        
        Args:
            agent3_output: Output from InvoiceFraudAnalyzer (Agent3)
            
        Returns:
            Complete audit report dictionary
        """
        # Validate document type
        doc_type = agent3_output.get('document_type', 'UNKNOWN')
        if doc_type != self.SUPPORTED_DOC_TYPE:
            return self._generate_error_report(
                f"Unsupported document type: {doc_type}. Only INVOICE documents supported."
            )
        
        # Extract core data
        fraud_score = agent3_output.get('fraud_score', 0)
        risk_level = agent3_output.get('risk_level', 'UNKNOWN')
        triggered_flags = agent3_output.get('triggered_flags', [])
        ai_signals = agent3_output.get('ai_signals', {})
        semantic_analysis = agent3_output.get('semantic_analysis')
        summary = agent3_output.get('summary', {})
        
        # Extract invoice document data
        document_data = self._extract_document_data(agent3_output)
        
        # Build report sections
        report = {
            'document_type': self.SUPPORTED_DOC_TYPE,
            'generated_at': datetime.now().isoformat(),
            'fraud_score': fraud_score,
            'risk_level': self._normalize_risk_level(risk_level),
            'flag_summary': summary,
            'executive_summary': self._generate_executive_summary(fraud_score, risk_level),
            'invoice_details': self._extract_invoice_details(document_data),
            'validation_summary': self._generate_validation_summary(triggered_flags, ai_signals),
            'issues_identified': self._identify_issues(triggered_flags, document_data),
            'recommendations': self._generate_recommendations(triggered_flags, fraud_score),
            'semantic_insights': self._format_semantic_insights(semantic_analysis),
            'audit_narrative': None,
            'metadata': {
                'agent_version': '4.0-invoice-focused',
                'analysis_timestamp': datetime.now().isoformat(),
                'total_flags': len(triggered_flags),
                'gemini_enabled': self.use_gemini
            }
        }
        
        # Generate narrative
        if self.use_gemini and self.client:
            try:
                report['audit_narrative'] = self._generate_gemini_narrative(report, agent3_output)
            except Exception as e:
                print(f"⚠️  Narrative generation failed: {e}")
                report['audit_narrative'] = self._generate_template_narrative(report)
        else:
            report['audit_narrative'] = self._generate_template_narrative(report)
        
        return report

    def _generate_error_report(self, error_message: str) -> Dict[str, Any]:
        """Generate error report for invalid inputs."""
        return {
            'document_type': 'ERROR',
            'generated_at': datetime.now().isoformat(),
            'error': error_message,
            'fraud_score': None,
            'risk_level': 'ERROR',
            'executive_summary': f"❌ REPORT GENERATION FAILED\n\n{error_message}",
            'invoice_details': {},
            'validation_summary': "",
            'issues_identified': [error_message],
            'recommendations': ["Verify document type and resubmit."],
            'audit_narrative': f"Report generation failed: {error_message}"
        }

    def _extract_document_data(self, agent3_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract invoice document data from various possible locations."""
        # Try different possible locations for document data
        if 'document' in agent3_output and isinstance(agent3_output['document'], dict):
            return agent3_output['document']
        
        # Fallback: use agent3_output itself
        return agent3_output

    def _normalize_risk_level(self, risk_level: str) -> str:
        """Normalize risk level to standard format."""
        level_mapping = {
            'safe': 'APPROVED',
            'approved': 'APPROVED',
            'needs review': 'NEEDS_REVIEW',
            'needs_review': 'NEEDS_REVIEW',
            'high risk': 'HIGH_RISK',
            'high_risk': 'HIGH_RISK',
            'likely fraud / reject': 'REJECT',
            'likely_fraud': 'REJECT',
            'reject': 'REJECT'
        }
        return level_mapping.get(risk_level.lower(), risk_level.upper())

    def _generate_executive_summary(self, fraud_score: float, risk_level: str) -> str:
        """Generate executive summary section."""
        normalized_level = self._normalize_risk_level(risk_level)
        
        # Determine status message and icon
        status_messages = {
            'APPROVED': ("✅", "Invoice appears compliant with government standards and GST requirements."),
            'NEEDS_REVIEW': ("⚠️", "Invoice requires manual verification before payment processing."),
            'HIGH_RISK': ("🔶", "Invoice shows significant compliance issues. Escalate to compliance team."),
            'REJECT': ("❌", "Invoice flagged for potential fraud or serious non-compliance. Recommend rejection.")
        }
        
        icon, status = status_messages.get(normalized_level, ("❓", "Invoice status requires review."))
        
        # Risk score interpretation
        if fraud_score <= 20:
            score_interpretation = "LOW RISK - Document shows minimal compliance concerns."
        elif fraud_score <= 50:
            score_interpretation = "MODERATE RISK - Some issues require attention before approval."
        elif fraud_score <= 75:
            score_interpretation = "HIGH RISK - Multiple compliance failures detected."
        else:
            score_interpretation = "CRITICAL RISK - Strong indicators of fraud or non-compliance."
        
        return f"""
╔══════════════════════════════════════════════════════════════════════╗
║                         EXECUTIVE SUMMARY                             ║
╚══════════════════════════════════════════════════════════════════════╝

Document Type:        INVOICE
Risk Assessment Score: {fraud_score:.2f} / 100
Risk Classification:   {normalized_level}

{icon} STATUS: {status}

📊 RISK INTERPRETATION:
{score_interpretation}

This invoice has been analyzed against:
  • GST compliance requirements (GSTIN, HSN codes, tax calculations)
  • Financial accuracy (amount reconciliation, tax verification)
  • Document authenticity (signatures, seals, format)
  • Fraud indicators (duplicate detection, suspicious patterns)

Key findings and actionable recommendations follow.
"""

    def _extract_invoice_details(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format key invoice data points."""
        
        # Helper for safe nested access
        def safe_get(data: Dict, keys: List[str], default: Any = None) -> Any:
            current = data
            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key)
                else:
                    return default
            if isinstance(current, dict) and 'value' in current:
                return current.get('value')
            return current if current is not None else default
        
        # Extract amounts
        amounts = document_data.get('amounts', {})
        
        # Format line items
        line_items = document_data.get('line_items', [])
        items_formatted = self._format_line_items(line_items)
        
        # Format tax breakdown
        tax_breakdown = self._format_tax_breakdown(amounts)
        
        # Build invoice details
        details = {
            # Vendor Information
            'vendor_name': safe_get(document_data, ['vendor', 'name'], 'Not specified'),
            'vendor_gstin': safe_get(document_data, ['vendor', 'gstin'], '❌ Missing'),
            'vendor_pan': safe_get(document_data, ['vendor', 'pan'], 'Not specified'),
            'vendor_address': safe_get(document_data, ['vendor', 'address'], 'Not specified'),
            
            # Buyer Information
            'buyer_name': safe_get(document_data, ['buyer', 'name'], 'Not specified'),
            'buyer_gstin': safe_get(document_data, ['buyer', 'gstin'], 'Not specified'),
            'buyer_address': safe_get(document_data, ['buyer', 'address'], 'Not specified'),
            
            # Invoice Details
            'invoice_number': safe_get(document_data, ['invoice_number']) or 
                            safe_get(document_data, ['document_id'], 'Not specified'),
            'invoice_date': safe_get(document_data, ['invoice_date'], 'Not specified'),
            'due_date': safe_get(document_data, ['due_date']) or 
                       safe_get(document_data, ['payment_due_date'], 'Not specified'),
            
            # Financial Details
            'subtotal': self._format_currency(safe_get(amounts, ['subtotal'], 0)),
            'taxable_amount': self._format_currency(safe_get(amounts, ['taxable_amount'], 0)),
            'tax_breakdown': tax_breakdown,
            'total_tax': self._format_currency(safe_get(amounts, ['tax_amount'], 0)),
            'discount': self._format_currency(safe_get(amounts, ['discount'], 0)),
            'total_amount': self._format_currency(safe_get(amounts, ['total_amount'], 0)),
            
            # Line Items
            'line_items': items_formatted,
            'line_item_count': len(line_items),
            
            # Authentication
            'digital_signature': '✅ Present' if safe_get(document_data, ['authentication', 'has_digital_signature']) else '❌ Missing',
            'physical_seal': '✅ Present' if safe_get(document_data, ['authentication', 'has_seal']) or 
                                           safe_get(document_data, ['authentication', 'has_physical_seal']) else '❌ Missing',
            'signature_valid': '✅ Valid' if safe_get(document_data, ['authentication', 'signature_valid']) else '❓ Not verified',
            
            # Payment Terms
            'payment_terms': safe_get(document_data, ['payment_terms'], 'Not specified'),
            'bank_details': safe_get(document_data, ['bank_details'], 'Not specified'),
        }
        
        # Filter out None and empty values for cleaner output
        return {k: v for k, v in details.items() if v is not None and v != '' and v != '₹0.00'}

    def _format_currency(self, amount: Any) -> str:
        """Format amount as Indian Rupees."""
        try:
            if isinstance(amount, dict):
                amount = amount.get('value', 0)
            return f"₹{float(amount):,.2f}"
        except (ValueError, TypeError):
            return "₹0.00"

    def _format_line_items(self, line_items: List[Dict[str, Any]]) -> str:
        """Format line items as readable text."""
        if not line_items:
            return "No line items specified"
        
        formatted = []
        for i, item in enumerate(line_items, 1):
            desc = item.get('description', 'Unknown item')
            qty = item.get('quantity', 'N/A')
            unit = item.get('unit', 'units')
            unit_price = item.get('unit_price', 0)
            total = item.get('total_amount') or item.get('amount', 0)
            hsn = item.get('hsn_code', '')
            
            line = f"  {i}. {desc}"
            if hsn:
                line += f" [HSN: {hsn}]"
            line += f"\n     Qty: {qty} {unit} × ₹{unit_price:,.2f} = ₹{total:,.2f}"
            formatted.append(line)
        
        return "\n" + "\n".join(formatted)

    def _format_tax_breakdown(self, amounts: Dict[str, Any]) -> str:
        """Format tax breakdown."""
        taxes = []
        
        cgst = amounts.get('cgst', 0)
        sgst = amounts.get('sgst', 0)
        igst = amounts.get('igst', 0)
        cess = amounts.get('cess', 0)
        
        if cgst:
            taxes.append(f"CGST: ₹{cgst:,.2f}")
        if sgst:
            taxes.append(f"SGST: ₹{sgst:,.2f}")
        if igst:
            taxes.append(f"IGST: ₹{igst:,.2f}")
        if cess:
            taxes.append(f"Cess: ₹{cess:,.2f}")
        
        return " | ".join(taxes) if taxes else "Tax details not specified"

    def _generate_validation_summary(
        self,
        triggered_flags: List[Dict[str, Any]],
        ai_signals: Dict[str, Any]
    ) -> str:
        """Generate human-readable validation summary."""
        
        if not triggered_flags and not ai_signals:
            return "✅ VALIDATION PASSED\n\nNo compliance issues detected during automated validation."
        
        summary = """
╔══════════════════════════════════════════════════════════════════════╗
║                       VALIDATION SUMMARY                              ║
╚══════════════════════════════════════════════════════════════════════╝

"""
        
        # AI Validation Status
        if ai_signals:
            status = ai_signals.get('status', 'UNKNOWN')
            errors = ai_signals.get('errors', [])
            checks = ai_signals.get('checks', {})
            
            if status == 'INVALID':
                summary += "🤖 AI VALIDATION STATUS: ❌ INVALID\n"
                if errors:
                    summary += f"   Errors: {', '.join(errors[:3])}\n"
                summary += "\n"
            elif status == 'VALID':
                summary += "🤖 AI VALIDATION STATUS: ✅ VALID\n\n"
            
            # Individual checks
            check_results = []
            if 'format_validation' in checks:
                icon = "✅" if checks['format_validation'] else "❌"
                check_results.append(f"   {icon} Format Validation")
            if 'financial_consistency' in checks:
                icon = "✅" if checks['financial_consistency'] else "❌"
                check_results.append(f"   {icon} Financial Consistency")
            if 'required_fields' in checks:
                icon = "✅" if checks['required_fields'] else "❌"
                check_results.append(f"   {icon} Required Fields")
            
            if check_results:
                summary += "   Automated Checks:\n" + "\n".join(check_results) + "\n\n"
        
        # Triggered Flags Summary
        if triggered_flags:
            summary += f"📋 COMPLIANCE FLAGS TRIGGERED: {len(triggered_flags)}\n\n"
            
            # Group by severity
            by_severity = {}
            for flag in triggered_flags:
                sev = flag.get('severity', 'medium').upper()
                if sev not in by_severity:
                    by_severity[sev] = []
                by_severity[sev].append(flag)
            
            # Display in order of severity
            severity_order = ['CRITICAL', 'HIGH', 'AI', 'MEDIUM', 'LOW']
            for severity in severity_order:
                if severity in by_severity:
                    icon = self.config['severity_icons'].get(severity.lower(), '❓')
                    summary += f"   {icon} {severity}: {len(by_severity[severity])} flag(s)\n"
                    for flag in by_severity[severity]:
                        summary += f"      • {flag.get('id', 'Unknown')}\n"
            
            summary += "\n"
        
        return summary

    def _identify_issues(
        self,
        triggered_flags: List[Dict[str, Any]],
        document_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify and explain all issues found."""
        issues = []
        
        # Process triggered flags
        for flag in triggered_flags:
            flag_id = flag.get('id', 'UNKNOWN')
            severity = flag.get('severity', 'medium')
            evidence = flag.get('evidence', {})
            
            # Get explanation from our knowledge base
            explanation = INVOICE_FLAG_EXPLANATIONS.get(flag_id)
            
            if explanation:
                issue = {
                    'flag_id': flag_id,
                    'severity': severity.upper(),
                    'icon': explanation['icon'],
                    'title': explanation['title'],
                    'description': explanation['description'],
                    'implication': explanation['implication'],
                    'recommended_action': explanation['action'],
                    'evidence': evidence
                }
            else:
                # Generic handling for unknown flags
                issue = {
                    'flag_id': flag_id,
                    'severity': severity.upper(),
                    'icon': '⚠️',
                    'title': flag_id.replace('_', ' ').title(),
                    'description': flag.get('description', 'Issue detected during validation.'),
                    'implication': 'This issue may affect invoice processing or compliance.',
                    'recommended_action': 'Review and address before approval.',
                    'evidence': evidence
                }
            
            issues.append(issue)
        
        # Check for missing authentication (even if not flagged)
        auth = document_data.get('authentication', {})
        if not auth.get('has_digital_signature') and not any(i['flag_id'] == 'INV_MISSING_SIGNATURE' for i in issues):
            issues.append({
                'flag_id': 'OBSERVATION_NO_SIGNATURE',
                'severity': 'OBSERVATION',
                'icon': '📝',
                'title': 'Digital Signature Not Present',
                'description': 'Invoice does not have a digital signature.',
                'implication': 'Authenticity verification is limited without digital signature.',
                'recommended_action': 'Consider requesting signed invoice for high-value transactions.',
                'evidence': {}
            })
        
        return issues

    def _generate_recommendations(
        self,
        triggered_flags: List[Dict[str, Any]],
        fraud_score: float
    ) -> List[Dict[str, Any]]:
        """Generate prioritized, actionable recommendations."""
        recommendations = []
        
        # Primary recommendation based on risk level
        if fraud_score >= 80:
            recommendations.append({
                'priority': 1,
                'icon': '🛑',
                'category': 'IMMEDIATE_ACTION',
                'action': 'REJECT this invoice. Do not process payment.',
                'reason': 'Critical fraud indicators detected. Requires legal/compliance review.',
                'escalate_to': 'Compliance Officer / Legal Team'
            })
        elif fraud_score >= 50:
            recommendations.append({
                'priority': 1,
                'icon': '⚠️',
                'category': 'MANUAL_REVIEW',
                'action': 'Hold payment pending manual verification.',
                'reason': 'Multiple compliance issues require human review.',
                'escalate_to': 'Finance Supervisor'
            })
        elif fraud_score >= 20:
            recommendations.append({
                'priority': 1,
                'icon': '🔍',
                'category': 'VERIFY_AND_PROCEED',
                'action': 'Verify flagged items before approval.',
                'reason': 'Minor issues detected that should be addressed.',
                'escalate_to': None
            })
        else:
            recommendations.append({
                'priority': 1,
                'icon': '✅',
                'category': 'APPROVED',
                'action': 'Invoice may be processed for payment.',
                'reason': 'No significant compliance issues detected.',
                'escalate_to': None
            })
        
        # Flag-specific recommendations
        flag_ids = {f.get('id', '') for f in triggered_flags}
        
        if any('GSTIN' in fid for fid in flag_ids):
            recommendations.append({
                'priority': 2,
                'icon': '🔏',
                'category': 'GST_COMPLIANCE',
                'action': 'Verify vendor GSTIN at gstin.gov.in before processing.',
                'reason': 'Invalid or missing GSTIN affects GST input credit.',
                'escalate_to': None
            })
        
        if any('AMOUNT' in fid or 'FINANCIAL' in fid for fid in flag_ids):
            recommendations.append({
                'priority': 2,
                'icon': '🧮',
                'category': 'FINANCIAL_VERIFICATION',
                'action': 'Request itemized breakdown and reconcile all amounts.',
                'reason': 'Amount discrepancies must be resolved before payment.',
                'escalate_to': None
            })
        
        if any('DUPLICATE' in fid for fid in flag_ids):
            recommendations.append({
                'priority': 2,
                'icon': '🔄',
                'category': 'DUPLICATE_CHECK',
                'action': 'Check payment history for this invoice number.',
                'reason': 'Duplicate payment must be prevented.',
                'escalate_to': 'Accounts Payable'
            })
        
        if any('SIGNATURE' in fid for fid in flag_ids):
            recommendations.append({
                'priority': 3,
                'icon': '📝',
                'category': 'AUTHENTICATION',
                'action': 'Request digitally signed invoice for audit trail.',
                'reason': 'Digital signature required for e-invoicing compliance.',
                'escalate_to': None
            })
        
        if any('HSN' in fid for fid in flag_ids):
            recommendations.append({
                'priority': 3,
                'icon': '🏷️',
                'category': 'HSN_COMPLIANCE',
                'action': 'Request HSN/SAC codes for all line items.',
                'reason': 'HSN codes required for GST filing.',
                'escalate_to': None
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        
        return recommendations

    def _format_semantic_insights(
        self,
        semantic_analysis: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Format semantic analysis insights if available."""
        if not semantic_analysis:
            return None
        
        return {
            'semantic_fraud_score': semantic_analysis.get('semantic_fraud_score', 0) * 100,
            'risk_indicators': semantic_analysis.get('risk_indicators', []),
            'confidence': semantic_analysis.get('confidence', 'N/A'),
            'analysis_notes': semantic_analysis.get('notes', [])
        }

    def _generate_gemini_narrative(
        self,
        report: Dict[str, Any],
        agent3_output: Dict[str, Any]
    ) -> str:
        """Generate comprehensive audit narrative via Gemini."""
        
        # Format issues for prompt
        issues_text = "\n".join([
            f"- {issue['title']}: {issue['description']}"
            for issue in report['issues_identified'][:10]
        ])
        
        # Format recommendations for prompt
        rec_text = "\n".join([
            f"- [{rec['category']}] {rec['action']}"
            for rec in report['recommendations'][:5]
        ])
        
        prompt = f"""You are a government auditor writing a compliance review report for an invoice.

Generate a professional, audit-ready narrative report based on this analysis:

FRAUD RISK SCORE: {report['fraud_score']}/100
RISK CLASSIFICATION: {report['risk_level']}

INVOICE DETAILS:
- Vendor: {report['invoice_details'].get('vendor_name', 'Unknown')}
- Vendor GSTIN: {report['invoice_details'].get('vendor_gstin', 'Not specified')}
- Invoice Number: {report['invoice_details'].get('invoice_number', 'Unknown')}
- Invoice Date: {report['invoice_details'].get('invoice_date', 'Unknown')}
- Total Amount: {report['invoice_details'].get('total_amount', 'Unknown')}

ISSUES IDENTIFIED:
{issues_text if issues_text else "No significant issues found."}

RECOMMENDATIONS:
{rec_text}

Write a 300-400 word professional audit narrative that:
1. Opens with a clear assessment statement
2. Summarizes what was reviewed and key invoice details
3. Explains the issues found in business language (not technical jargon)
4. Describes the risk implications for payment processing
5. Concludes with clear next steps and approval recommendation

The narrative should be suitable for government procurement officers.
Use formal but clear language. Include specific amounts where relevant.
Format as plain text paragraphs. Do not use markdown or JSON formatting.

IMPORTANT: Write ONLY the narrative text. No headers, bullets, or formatting."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.config.get('narrative_temperature', 0.5),
                    max_output_tokens=self.config.get('narrative_max_tokens', 1000),
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"⚠️  Gemini narrative generation failed: {e}")
            return self._generate_template_narrative(report)

    def _generate_template_narrative(self, report: Dict[str, Any]) -> str:
        """Generate template-based narrative when Gemini is unavailable."""
        
        fraud_score = report['fraud_score']
        risk_level = report['risk_level']
        details = report['invoice_details']
        issues = report['issues_identified']
        recommendations = report['recommendations']
        
        # Determine assessment statement
        if risk_level == 'APPROVED':
            assessment = "This invoice has passed automated compliance verification and is recommended for payment processing."
        elif risk_level == 'NEEDS_REVIEW':
            assessment = "This invoice requires manual verification before payment authorization due to compliance concerns."
        elif risk_level == 'HIGH_RISK':
            assessment = "This invoice shows significant compliance issues and should be escalated to the compliance team."
        else:
            assessment = "This invoice is NOT recommended for payment. Critical compliance failures or fraud indicators detected."
        
        narrative = f"""AUDIT NARRATIVE REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generated: {report['generated_at']}

ASSESSMENT SUMMARY:
{assessment}

DOCUMENT REVIEWED:
This audit covers Invoice {details.get('invoice_number', 'N/A')} dated {details.get('invoice_date', 'N/A')}, 
submitted by {details.get('vendor_name', 'Unknown Vendor')} (GSTIN: {details.get('vendor_gstin', 'Not provided')}) 
for a total amount of {details.get('total_amount', 'N/A')}.

COMPLIANCE ANALYSIS:
The invoice was evaluated against government procurement standards including GST compliance, 
financial accuracy, document authenticity, and fraud detection rules. The analysis yielded 
a fraud risk score of {fraud_score:.1f} out of 100, resulting in a {risk_level} classification.

"""
        
        # Add issues section
        if issues:
            narrative += f"ISSUES IDENTIFIED ({len(issues)}):\n"
            for i, issue in enumerate(issues[:5], 1):
                narrative += f"{i}. {issue['icon']} {issue['title']}\n"
                narrative += f"   {issue['description']}\n"
                narrative += f"   Implication: {issue['implication']}\n\n"
        else:
            narrative += "ISSUES IDENTIFIED:\nNo material compliance issues were detected during review.\n\n"
        
        # Add recommendations
        narrative += "RECOMMENDED ACTIONS:\n"
        for i, rec in enumerate(recommendations[:4], 1):
            narrative += f"{i}. {rec['icon']} [{rec['category']}]\n"
            narrative += f"   {rec['action']}\n"
            if rec.get('escalate_to'):
                narrative += f"   Escalate to: {rec['escalate_to']}\n"
            narrative += "\n"
        
        # Conclusion
        narrative += f"""CONCLUSION:
Based on this comprehensive review, the invoice status is: {risk_level}

• APPROVED: Proceed with payment processing per standard procedures.
• NEEDS_REVIEW: Forward to supervisor with this report for verification.
• HIGH_RISK: Escalate to compliance team. Do not process without approval.
• REJECT: Do not process. Initiate investigation per fraud protocols.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This report was generated by the Invoice Audit System v4.0
Retain as part of official procurement records.
"""
        
        return narrative

    def export_report(
        self,
        report: Dict[str, Any],
        format: Literal['json', 'txt', 'markdown'] = 'json',
        output_path: Optional[str] = None
    ) -> str:
        """
        Export audit report to file.
        
        Args:
            report: Generated audit report
            format: Output format ('json', 'txt', 'markdown')
            output_path: Custom output path (auto-generated if not specified)
            
        Returns:
            Path to exported file
        """
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        invoice_num = report.get('invoice_details', {}).get('invoice_number', 'UNKNOWN')
        invoice_num = str(invoice_num).replace('/', '-').replace('\\', '-')[:20]
        base_name = f"invoice_audit_{invoice_num}_{timestamp}"
        
        if format == 'json':
            return self._export_json(report, output_path or f"{base_name}.json")
        elif format == 'txt':
            return self._export_txt(report, output_path or f"{base_name}.txt")
        elif format == 'markdown':
            return self._export_markdown(report, output_path or f"{base_name}.md")
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_json(self, report: Dict[str, Any], path: str) -> str:
        """Export as JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        return path

    def _export_txt(self, report: Dict[str, Any], path: str) -> str:
        """Export as plain text."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report['executive_summary'])
            f.write("\n\n")
            
            f.write("INVOICE DETAILS\n")
            f.write("─" * 60 + "\n")
            for key, value in report['invoice_details'].items():
                # Format key nicely
                display_key = key.replace('_', ' ').title()
                f.write(f"{display_key}: {value}\n")
            f.write("\n")
            
            f.write(report['validation_summary'])
            f.write("\n")
            
            f.write("ISSUES IDENTIFIED\n")
            f.write("─" * 60 + "\n")
            for issue in report['issues_identified']:
                f.write(f"\n{issue['icon']} [{issue['severity']}] {issue['title']}\n")
                f.write(f"   {issue['description']}\n")
                f.write(f"   Implication: {issue['implication']}\n")
                f.write(f"   Action: {issue['recommended_action']}\n")
            f.write("\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("─" * 60 + "\n")
            for rec in report['recommendations']:
                f.write(f"\n{rec['icon']} [{rec['category']}]\n")
                f.write(f"   {rec['action']}\n")
                if rec.get('reason'):
                    f.write(f"   Reason: {rec['reason']}\n")
            f.write("\n\n")
            
            f.write("AUDIT NARRATIVE\n")
            f.write("─" * 60 + "\n")
            f.write(report['audit_narrative'])
        
        return path

    def _export_markdown(self, report: Dict[str, Any], path: str) -> str:
        """Export as Markdown."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Invoice Audit Report\n\n")
            f.write(f"**Generated:** {report['generated_at']}\n\n")
            
            f.write("## Risk Assessment\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| **Fraud Score** | {report['fraud_score']}/100 |\n")
            f.write(f"| **Risk Level** | {report['risk_level']} |\n")
            f.write(f"| **Total Flags** | {report['flag_summary'].get('total_flags', 0)} |\n\n")
            
            f.write("## Invoice Details\n\n")
            f.write("| Field | Value |\n")
            f.write("|-------|-------|\n")
            for key, value in report['invoice_details'].items():
                display_key = key.replace('_', ' ').title()
                # Escape pipe characters in value
                value_str = str(value).replace('|', '\\|').replace('\n', '<br>')
                f.write(f"| {display_key} | {value_str} |\n")
            f.write("\n")
            
            f.write("## Issues Identified\n\n")
            for issue in report['issues_identified']:
                f.write(f"### {issue['icon']} {issue['title']}\n\n")
                f.write(f"**Severity:** {issue['severity']}\n\n")
                f.write(f"{issue['description']}\n\n")
                f.write(f"**Implication:** {issue['implication']}\n\n")
                f.write(f"**Recommended Action:** {issue['recommended_action']}\n\n")
            
            f.write("## Recommendations\n\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. **{rec['icon']} {rec['category']}**\n")
                f.write(f"   - {rec['action']}\n")
                if rec.get('reason'):
                    f.write(f"   - *Reason:* {rec['reason']}\n")
                if rec.get('escalate_to'):
                    f.write(f"   - *Escalate to:* {rec['escalate_to']}\n")
                f.write("\n")
            
            f.write("## Audit Narrative\n\n")
            f.write(report['audit_narrative'])
            f.write("\n\n---\n")
            f.write("*Report generated by Invoice Audit System v4.0*\n")
        
        return path


# =============================================================================
# BACKWARD COMPATIBILITY ALIAS
# =============================================================================

Agent4 = InvoiceAuditReportGenerator


# =============================================================================
# DEMO / CLI RUNNER
# =============================================================================

def create_sample_agent3_output() -> Dict[str, Any]:
    """Create sample Agent3 output for testing."""
    return {
        "document_type": "INVOICE",
        "fraud_score": 45.5,
        "risk_level": "Needs Review",
        "summary": {
            "total_flags": 4,
            "critical_flags": 0,
            "high_flags": 2,
            "ai_flags": 1
        },
        "triggered_flags": [
            {
                "id": "INV_GSTIN_MISSING",
                "description": "Vendor GSTIN is not provided",
                "severity": "high",
                "evidence": {"field": "vendor.gstin", "value": None}
            },
            {
                "id": "INV_MISSING_SIGNATURE",
                "description": "Invoice lacks digital signature",
                "severity": "medium",
                "evidence": {}
            },
            {
                "id": "INV_TAX_MISMATCH",
                "description": "Tax calculation does not match expected amount",
                "severity": "high",
                "evidence": {"expected": 1800, "actual": 1500, "difference": 300}
            },
            {
                "id": "AI_FINANCIAL_INCONSISTENCY",
                "description": "AI detected financial calculation issues",
                "severity": "ai",
                "evidence": {"check": "financial_consistency"}
            }
        ],
        "ai_signals": {
            "status": "INVALID",
            "checks": {
                "format_validation": True,
                "financial_consistency": False,
                "required_fields": False
            },
            "errors": ["Missing vendor GSTIN", "Tax calculation mismatch"]
        },
        "document": {
            "document_id": "INV-2024-00567",
            "document_type": "INVOICE",
            "invoice_number": "INV-2024-00567",
            "invoice_date": "2024-01-20",
            "due_date": "2024-02-20",
            "vendor": {
                "name": "Quick Supplies Trading Co.",
                "gstin": None,
                "pan": "AABCS1234F",
                "address": "45 Market Street, Delhi"
            },
            "buyer": {
                "name": "Government Department XYZ",
                "gstin": "07AAAGD0123E1Z5",
                "address": "Sector 12, New Delhi"
            },
            "amounts": {
                "subtotal": 50000.00,
                "taxable_amount": 50000.00,
                "cgst": 4500.00,
                "sgst": 4500.00,
                "tax_amount": 9000.00,
                "discount": 0,
                "total_amount": 59000.00
            },
            "line_items": [
                {
                    "description": "Office Stationery Kit",
                    "hsn_code": "4820",
                    "quantity": 50,
                    "unit": "sets",
                    "unit_price": 600.00,
                    "total_amount": 30000.00
                },
                {
                    "description": "Printer Paper A4 (500 sheets)",
                    "hsn_code": "4802",
                    "quantity": 100,
                    "unit": "reams",
                    "unit_price": 200.00,
                    "total_amount": 20000.00
                }
            ],
            "authentication": {
                "has_digital_signature": False,
                "has_seal": True,
                "signature_valid": False
            }
        }
    }


def print_report_summary(report: Dict[str, Any]) -> None:
    """Print a summary of the generated report."""
    print("\n" + "=" * 70)
    print("📋 INVOICE AUDIT REPORT GENERATED")
    print("=" * 70)
    
    if report.get('error'):
        print(f"\n❌ Error: {report['error']}")
        return
    
    print(f"\n📊 Risk Score: {report['fraud_score']}/100")
    print(f"⚠️  Risk Level: {report['risk_level']}")
    print(f"🚩 Issues Found: {len(report['issues_identified'])}")
    print(f"💡 Recommendations: {len(report['recommendations'])}")
    
    print("\n📄 Invoice Details:")
    details = report.get('invoice_details', {})
    print(f"   Vendor: {details.get('vendor_name', 'N/A')}")
    print(f"   Invoice #: {details.get('invoice_number', 'N/A')}")
    print(f"   Amount: {details.get('total_amount', 'N/A')}")
    print(f"   GSTIN: {details.get('vendor_gstin', 'N/A')}")
    
    print("\n🚩 Top Issues:")
    for issue in report['issues_identified'][:3]:
        print(f"   {issue['icon']} [{issue['severity']}] {issue['title']}")
    
    print("\n💡 Primary Recommendation:")
    if report['recommendations']:
        rec = report['recommendations'][0]
        print(f"   {rec['icon']} {rec['action']}")


def main():
    """Main demo function."""
    import sys
    
    print("\n" + "=" * 70)
    print("🔍 INVOICE AUDIT REPORT GENERATOR - DEMO")
    print("=" * 70)
    
    # Check for command line input
    if len(sys.argv) > 1:
        agent3_output_path = sys.argv[1]
        if os.path.exists(agent3_output_path):
            print(f"\n📂 Loading Agent3 output from: {agent3_output_path}")
            with open(agent3_output_path, 'r', encoding='utf-8') as f:
                agent3_output = json.load(f)
        else:
            print(f"\n❌ File not found: {agent3_output_path}")
            print("Using sample data instead...")
            agent3_output = create_sample_agent3_output()
    else:
        print("\n📝 No input file provided. Using sample invoice data...")
        print("   Usage: python agent_4.py <agent3_output.json>")
        agent3_output = create_sample_agent3_output()
    
    # Initialize generator
    generator = InvoiceAuditReportGenerator(use_gemini=True)
    
    # Generate report
    print("\n⚙️  Generating audit report...")
    report = generator.generate_audit_report(agent3_output)
    
    # Print summary
    print_report_summary(report)
    
    # Export in multiple formats
    print("\n📁 Exporting reports...")
    try:
        json_path = generator.export_report(report, format='json')
        txt_path = generator.export_report(report, format='txt')
        md_path = generator.export_report(report, format='markdown')
        
        print(f"   ✅ JSON: {json_path}")
        print(f"   ✅ Text: {txt_path}")
        print(f"   ✅ Markdown: {md_path}")
    except Exception as e:
        print(f"   ⚠️  Export failed: {e}")
    
    # Print executive summary
    print("\n" + report['executive_summary'])
    
    # Print narrative
    print("\n📝 AUDIT NARRATIVE:")
    print("-" * 70)
    print(report['audit_narrative'][:1500] + "..." if len(report['audit_narrative']) > 1500 else report['audit_narrative'])


if __name__ == '__main__':

    main()
