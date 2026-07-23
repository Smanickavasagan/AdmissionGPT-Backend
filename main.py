import os
import json
import re
import base64
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from db import documents_collection
from tools import save_to_mongodb, query_documents
from agent import llm_chat, processing_system_prompt, chat_system_prompt

load_dotenv()

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5174")
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8000))

app = FastAPI(title="Form2Database API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"},
        headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "*", "Access-Control-Allow-Headers": "*"}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "*", "Access-Control-Allow-Headers": "*"}
    )



# ----------------- Models -----------------

class SaveRequest(BaseModel):
    filename: str
    structured_json: dict | None = None
    text: str | None = None

class ChatRequest(BaseModel):
    message: str

# ----------------- Helpers -----------------

def extract_json_from_text(text: str) -> dict:
    """Extract a JSON object from LLM output, even if wrapped in markdown."""
    # Try stripping markdown fences
    clean = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Try finding the first {...} block
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"No valid JSON found in LLM response: {text[:200]}")

# ----------------- Endpoints -----------------

@app.get("/health")
def health_check():
    return {"status": "ok"}

from rapidocr_onnxruntime import RapidOCR

ocr_engine = RapidOCR()

def paddle_ocr_extract(image_bytes: bytes) -> str:
    """Extract text from image bytes using PaddleOCR (RapidOCR ONNX engine)."""
    result, elapse = ocr_engine(image_bytes)
    if result:
        lines = [line[1] for line in result]
        return "\n".join(lines)
    return ""

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

async def process_ocr_with_qwen(ocr_text: str, filename: str) -> dict:
    """Send extracted OCR text to Qwen model to structure into JSON format."""
    model_name = os.getenv("CHAT_MODEL", "qwen2.5:7b")
    print(f"[Qwen AI] Formatting extracted text using model '{model_name}'...")
    payload = {
        "model": model_name,
        "prompt": f"{processing_system_prompt}\n\nOCR text from '{filename}':\n\n{ocr_text}",
        "stream": False,
        "options": {"temperature": 0}
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        resp.raise_for_status()
        raw = resp.json().get("response", "")
    
    return extract_json_from_text(raw)

@app.post("/upload_and_ocr")
async def upload_and_ocr(file: UploadFile = File(...)):
    print(f"[API /upload_and_ocr] Received file: {file.filename}, type: {file.content_type}")
    try:
        content = await file.read()
        ocr_text = paddle_ocr_extract(content)
        print(f"[PaddleOCR] Extracted {len(ocr_text)} characters from {file.filename}")
        
        # Process extracted OCR text using Qwen model into structured JSON
        structured_json = {}
        if ocr_text:
            try:
                structured_json = await process_ocr_with_qwen(ocr_text, file.filename)
                print(f"[Qwen AI] Successfully structured data into JSON for {file.filename}")
            except Exception as qwen_err:
                print(f"[Qwen AI Warning] Could not parse JSON with Qwen: {qwen_err}")
                structured_json = {"raw_text": ocr_text}
        
        return {
            "filename": file.filename,
            "ocr_text": ocr_text,
            "structured_json": structured_json
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload & OCR processing failed: {str(e)}")

@app.post("/process_and_save")
@app.post("/save_document")
async def process_and_save(req: SaveRequest):
    print(f"[API /process_and_save] Request received for file: '{req.filename}'")
    try:
        structured_data = None
        if req.structured_json and isinstance(req.structured_json, dict) and len(req.structured_json) > 0:
            print(f"[API /process_and_save] Using pre-extracted structured JSON ({len(req.structured_json)} fields)")
            structured_data = req.structured_json
        elif req.text:
            print(f"[Qwen AI] Structuring raw OCR text into JSON for '{req.filename}'...")
            structured_data = await process_ocr_with_qwen(req.text, req.filename)

        if not structured_data:
            raise ValueError("No valid structured JSON or text provided to save.")

        print(f"[MongoDB] Saving document for '{req.filename}' into collection...")
        result = save_to_mongodb(req.filename, structured_data)
        print(f"[MongoDB Success] Saved document ID: {result['inserted_id']}")

        return {
            "id": result["inserted_id"],
            "filename": result["stored_filename"],
            "status": "complete",
            "extractedData": result["structured_data"]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")


@app.post("/chat")
async def chat_with_agent(req: ChatRequest):
    try:
        # Fetch limited DB docs to keep context small (fits in num_ctx=2048)
        db_docs = query_documents()[:5]
        
        if db_docs:
            # Only send key fields to save tokens
            slim_docs = [
                {
                    "filename": d.get("filename"),
                    "studentName": d.get("studentName", {}).get("value"),
                    "admissionNo": d.get("admissionNo", {}).get("value"),
                    "phone": d.get("phone", {}).get("value"),
                    "dateOfBirth": d.get("dateOfBirth", {}).get("value"),
                }
                for d in db_docs
            ]
            db_context = f"Database records (latest 5):\n{json.dumps(slim_docs, indent=2)}"
        else:
            db_context = "The database is currently empty."
        
        messages = [
            ("system", chat_system_prompt),
            ("human", f"{db_context}\n\nUser question: {req.message}")
        ]
        
        response_msg = llm_chat.invoke(messages)
        return {"response": response_msg.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/documents")
def get_documents():
    try:
        docs = list(documents_collection.find().sort("_id", -1))
        formatted = []
        for doc in docs:
            oid = doc.get("_id")
            try:
                upload_date = str(oid.generation_time)
            except AttributeError:
                upload_date = None
            formatted.append({
                "id": str(oid),
                "filename": doc.get("filename", "Unknown"),
                "status": "complete",
                "extractedData": {k: v for k, v in doc.items() if k not in ["_id", "filename"]},
                "uploadDate": upload_date
            })
        return formatted
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB fetch failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT)