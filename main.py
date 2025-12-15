import os
import shutil
import uuid
import json
from typing import Dict, Any, Optional
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import Configs
load_dotenv()

from agent_1 import InvoiceProcessor, ExtractedNormalizedInvoice

from agent_2 import InvoiceFraudAnalyzer
from agent_3 import InvoiceAuditReportGenerator

app = FastAPI(title="Agentic Invoice Auditor API (NIM Powered)", version="2.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory storage for file mapping (file_id -> file_path)
FILE_STORE: Dict[str, str] = {}

# --- Pydantic Models for Input/Output ---

class FileIdRequest(BaseModel):
    file_id: str

class Agent2Request(BaseModel):
    document: Dict[str, Any]

class Agent3Request(BaseModel):
    document: Dict[str, Any]

class Agent4Request(BaseModel):
    fraud_analysis: Dict[str, Any]


processor = InvoiceProcessor()

# Analytical Agents
agent3 = InvoiceFraudAnalyzer(use_gemini=False)
agent4 = InvoiceAuditReportGenerator(use_gemini=True) 

# --- Utility Functions ---

def get_file_path(file_id: str) -> str:
    path = FILE_STORE.get(file_id)
    if not path:
        raise HTTPException(status_code=404, detail="File ID not found")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File content not found on server")
    return path

# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "Agentic Invoice Auditor API (NIM) is running", "endpoints": [
        "/api/upload", "/api/agent1/extract", "/api/agent2/normalize", 
        "/api/agent3/analyze", "/api/agent4/report", "/api/complete"
    ]}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    🟦 Step 0: Upload Invoice/Tender/Contract
    """
    try:
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        file_name = f"{file_id}{file_extension}"
        file_path = UPLOAD_DIR / file_name

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        FILE_STORE[file_id] = str(file_path)

        return {"file_id": file_id, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/agent1/extract")
async def extract_invoice(request: FileIdRequest):
    """
    🟧 Step 1 & 🟨 Step 2 Combined: Document Extraction & Normalization via NIM
    (Exposed as Agent 1/2 endpoint for compatibility)
    """
    try:
        file_path = get_file_path(request.file_id)
        
        # New Combined Processor
        # output is already normalized dict
        normalized_data = processor.process(file_path)
        
        return normalized_data

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.post("/api/agent2/normalize")
async def normalize_invoice(request: Agent2Request):
    """
    🟨 Step 2: Clean & Validate
    Note: In new architecture, extraction (Step 1) already returns normalized data.
    This endpoint acts as a pass-through or re-validator if needed.
    """
    return request.document

@app.post("/api/agent3/analyze")
async def analyze_fraud(request: Agent3Request):
    """
    🟥 Step 3: Fraud & Risk Analysis (Agent 3)
    """
    try:
        # Agent 3 expects normalized data
        normalized_data = request.document
    
        analysis_result = agent3.analyze(normalized_data)
        
        return analysis_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fraud analysis failed: {str(e)}")

@app.post("/api/agent4/report")
async def generate_report(request: Agent4Request):
    """
    🟩 Step 4: Full Audit Report (Agent 4)
    """
    try:
        analysis_data = request.fraud_analysis
        
        # Process with Agent 4
        report = agent4.generate_audit_report(analysis_data)
        
        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.post("/api/complete")
async def complete_pipeline(file: UploadFile = File(...), enable_semantic_ai: bool = True):
    """
    🟪 One-Shot Pipeline (Demo Mode)
    Powered by Nvidia NIM + Gemini
    """
    try:
        # 1. Save File
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        file_name = f"{file_id}{file_extension}"
        file_path = UPLOAD_DIR / file_name
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        FILE_STORE[file_id] = str(file_path)

        # 2. Combined Agent 1 & 2: Extraction + Normalization (NIM)
        print(f"[{file_id}] Starting NIM Extraction...")
        normalized_output = processor.process(str(file_path))
        
        # 3. Agent 3: Analysis (Rules + Deterministic)
        print(f"[{file_id}] Starting Fraud Analysis...")
        
        if enable_semantic_ai:
             # Create a stronger analyzer for the complete pipeline if requested
             analyzer = InvoiceFraudAnalyzer(use_gemini=True)
             analysis_result = analyzer.analyze(normalized_output)
        else:
             analysis_result = agent3.analyze(normalized_output)

        # 4. Agent 4: Reporting
        print(f"[{file_id}] Generating Report...")
        report = agent4.generate_audit_report(analysis_result)

        return {
            "status": "success",
            "file_id": file_id,
            "pipeline_stages": {
                "extraction_normalization": "completed (NIM)",
                "fraud_analysis": "completed",
                "reporting": "completed"
            },
            "final_report": report
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
