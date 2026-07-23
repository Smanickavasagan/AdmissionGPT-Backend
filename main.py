import os
import json
import re
import base64
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
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
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Models -----------------

class ProcessRequest(BaseModel):
    filename: str
    text: str

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

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

async def ollama_ocr(image_bytes: bytes, content_type: str) -> str:
    """Send image to deepseek-ocr via Ollama and return extracted text."""
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    model_name = os.getenv("OCR_MODEL", "deepseek-ocr:3b")
    print(f"[OCR] Sending {len(image_bytes)} bytes image to Ollama ({model_name}) without timeout limit...")
    payload = {
        "model": model_name,
        "prompt": (
            "You are an OCR engine. Extract ALL text from this image exactly as it appears. "
            "Return only the raw extracted text — no commentary, no formatting, no explanation."
        ),
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0}
    }
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        print(f"[OCR] Successfully extracted {len(text)} characters of text from image.")
        return text

@app.post("/upload_and_ocr")
async def upload_and_ocr(file: UploadFile = File(...)):
    print(f"[API /upload_and_ocr] Received file: {file.filename}, type: {file.content_type}")
    try:
        content = await file.read()
        ocr_text = await ollama_ocr(content, file.content_type or "image/jpeg")
        return {"filename": file.filename, "ocr_text": ocr_text}
    except httpx.HTTPStatusError as e:
        print(f"[API /upload_and_ocr Error] Ollama HTTP Error: {e.response.text}")
        raise HTTPException(status_code=500, detail=f"Ollama OCR failed: {e.response.text}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

@app.post("/process_and_save")
async def process_and_save(req: ProcessRequest):
    print(f"[API /process_and_save] Processing request for file: {req.filename}")
    try:
        payload = {
            "model": os.getenv("OCR_MODEL", "deepseek-ocr:3b"),
            "prompt": f"{processing_system_prompt}\n\nOCR text from '{req.filename}':\n\n{req.text}",
            "stream": False,
            "options": {"temperature": 0}
        }
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            resp.raise_for_status()
            raw = resp.json().get("response", "")

        structured_data = extract_json_from_text(raw)
        result = save_to_mongodb(req.filename, structured_data)

        return {
            "id": result["inserted_id"],
            "filename": req.filename,
            "status": "complete",
            "extractedData": result["structured_data"]
        }
    except httpx.HTTPStatusError as e:
        print(f"[API /process_and_save Error] Ollama HTTP Error: {e.response.text}")
        raise HTTPException(status_code=500, detail=f"Ollama processing failed: {e.response.text}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

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