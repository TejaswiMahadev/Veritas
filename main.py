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

# Import Agents
from agent_1 import InvoiceParsingAgent, InvoiceData
from agent_2 import InvoiceNormalizer, NormalizedInvoice
from agent_3 import InvoiceFraudAnalyzer
from agent_4 import InvoiceAuditReportGenerator

# Load environment variables
load_dotenv()

app = FastAPI(title="Agentic Invoice Auditor API", version="1.0.0")

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

# --- Initialize Agents ---
# We initialize them once to reuse connections/configs if applicable
agent1 = InvoiceParsingAgent()
agent2 = InvoiceNormalizer(use_gemini="when_needed")
agent3 = InvoiceFraudAnalyzer(use_gemini=True)  # Enable Semantic Analysis
agent4 = InvoiceAuditReportGenerator(use_gemini=True) # Enable Narrative Generation

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
    return {"message": "Agentic Invoice Auditor API is running", "endpoints": [
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
    🟧 Step 1: Document Extraction (Agent 1)
    """
    try:
        file_path = get_file_path(request.file_id)
        
        # Process with Agent 1
        invoice_data: InvoiceData = agent1.process_invoice(file_path)
        
        # Return as JSON
        # InvoiceData can be complex, so we rely on its internal export capability or dict conversion
        # agent_1.py has export_to_json that returns a string, we parse it back to dict for API JSON response
        json_str = agent1.export_to_json(invoice_data)
        return json.loads(json_str)

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.post("/api/agent2/normalize")
async def normalize_invoice(request: Agent2Request):
    """
    🟨 Step 2: Clean & Validate (Agent 2)
    """
    try:
        # Agent 2 expects a dictionary (which comes from Agent 1 output)
        raw_doc = request.document
        
        # Process with Agent 2
        normalized_doc: NormalizedInvoice = agent2.process_invoice(raw_doc)
        
        # Get structured output
        final_output = agent2.get_final_output(normalized_doc)
        
        return final_output

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Normalization failed: {str(e)}")

@app.post("/api/agent3/analyze")
async def analyze_fraud(request: Agent3Request):
    """
    🟥 Step 3: Fraud & Risk Analysis (Agent 3)
    """
    try:
        # Agent 3 expects the output of Agent 2
        normalized_data = request.document
        
        # Process with Agent 3
        # Agent 3's analyze method expects a dict
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
    """
    cleanup_path = None
    try:
        # 1. Save File
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        file_name = f"{file_id}{file_extension}"
        file_path = UPLOAD_DIR / file_name
        cleanup_path = file_path # Mark for cleanup if needed, though we usually keep uploads

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        FILE_STORE[file_id] = str(file_path)

        # 2. Agent 1: Extraction
        print(f"[{file_id}] Starting Extraction...")
        extraction_data = agent1.process_invoice(str(file_path))
        extraction_json = json.loads(agent1.export_to_json(extraction_data))

        # 3. Agent 2: Normalization
        print(f"[{file_id}] Starting Normalization...")
        normalized_doc = agent2.process_invoice(extraction_json)
        normalized_output = agent2.get_final_output(normalized_doc)

        # 4. Agent 3: Analysis
        print(f"[{file_id}] Starting Analysis...")
        # Agent 3 can optionally use Gemini based on initialization. We initialized it with use_gemini=True.
        # If the user disables it via query param, we could technically re-init or just let it run.
        # Ideally we'd pass a config override, but the class structure initializes it in __init__.
        # For this demo, we'll stick to the global agent unless strictly required.
        # Re-initializing for custom config if needed:
        if not enable_semantic_ai and agent3.use_gemini:
             # Temp disable for this request if we wanted to be strict, but for now we'll just run as is
             # or create a local instance. Let's create local to respect the flag if false.
             local_agent3 = InvoiceFraudAnalyzer(use_gemini=False)
             analysis_result = local_agent3.analyze(normalized_output)
        else:
             analysis_result = agent3.analyze(normalized_output)

        # 5. Agent 4: Reporting
        print(f"[{file_id}] Generating Report...")
        report = agent4.generate_audit_report(analysis_result)

        return {
            "status": "success",
            "file_id": file_id,
            "pipeline_stages": {
                "extraction": "completed",
                "normalization": "completed",
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
