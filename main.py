import os
import json
import re
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pytesseract
from PIL import Image
import io
from dotenv import load_dotenv

from db import documents_collection
from tools import save_to_mongodb, query_documents
from agent import llm_ocr, llm_chat, processing_system_prompt, chat_system_prompt

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

@app.post("/upload_and_ocr")
async def upload_and_ocr(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(image)
        return {"filename": file.filename, "ocr_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

@app.post("/process_and_save")
async def process_and_save(req: ProcessRequest):
    try:
        messages = [
            ("system", processing_system_prompt),
            ("human", f"OCR text from '{req.filename}':\n\n{req.text}")
        ]
        
        response_msg = llm_ocr.invoke(messages)
        structured_data = extract_json_from_text(response_msg.content)
        result = save_to_mongodb(req.filename, structured_data)

        return {
            "id": result["inserted_id"],
            "filename": req.filename,
            "status": "complete",
            "extractedData": result["structured_data"]
        }
    except Exception as e:
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
            formatted.append({
                "id": str(doc["_id"]),
                "filename": doc.get("filename", "Unknown"),
                "status": "complete",
                "extractedData": {k: v for k, v in doc.items() if k not in ["_id", "filename"]},
                "uploadDate": str(doc["_id"].generation_time) if "_id" in doc else None
            })
        return formatted
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB fetch failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT)