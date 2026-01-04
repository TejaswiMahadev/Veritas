# Veritas: Agentic Invoice Auditor

Veritas is a state-of-the-art, multi-agent AI system designed to automate the auditing of invoices, tenders, and contracts. It leverages **Nvidia NIM (Gemma)** for high-speed extraction and **Google Gemini** for deep semantic analysis and audit reasoning.

---

## 🚀 Key Features

### 1. Government E-Invoicing Compliance (April 2025 Mandates)
Automated validation of Indian GST E-Invoicing requirements:
- **IRN Mandate**: Automatically flags invoices missing an IRN for vendors with AATO > ₹5 crore.
- **30-Day Upload Limit**: Monitors the 30-day window for IRP uploads for entities above ₹10 crore turnover.
- **Exemption Tracking**: Smart bypass for exempt categories (Banking, GTA, Insurance, etc.).
- **AATO Registry**: Maintains historical turnover data to ensure once-in-always-in compliance.

### 2. AI Audit Cross-Examination
Moves beyond simple flags to provide actionable insights:
- **What is wrong**: Clear description of the violation.
- **Why it is wrong**: Deep technical and regulatory justification.
- **Auditor Question**: Targeted questions to ask the vendor or finance team.
- **Financial Risk**: Quantification of ITC loss, penalties, and litigation risk.

### 3. Robust Rule Engine
- **Tax Slab Validation**: Detects anomalies in GST rates (5, 12, 18, 28%).
- **Financial Integrity**: Matches line items to totals and taxable + tax to gross amounts.
- **Date Validation**: detects future dates and due-date mismatches.
- **Vendor Trust**: Real-time PAN extraction from GSTIN and lookup against vendor profiles.


## 📜 Regulatory Background
This system is configured to handle the **GSTN Notifications** effective through **April 1, 2025**, specifically focusing on e-invoicing thresholds and time-limit regulations for IRP reporting.

---
*Built for the Agentathon - Advanced Agentic Coding.*
