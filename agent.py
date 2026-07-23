from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:7b")

# Qwen2.5 — used for natural language chat / DB queries
llm_chat = ChatOllama(model=CHAT_MODEL, temperature=0, num_ctx=4096)

# ── OCR Extraction Prompt ──────────────────────────────────────────────────
# Model must return raw JSON only — no markdown, no explanation.
processing_system_prompt = """You are an AI assistant that extracts structured data from OCR text of admission forms.
Extract the following fields: student_name, date_of_birth, phone, admission_no, father_name, mother_name, address, email.
If a field is not found, use the value "Unknown".
Respond ONLY with a valid JSON object — no explanation, no markdown fences, just raw JSON.
Example:
{
  "student_name": "John Doe",
  "date_of_birth": "01-01-2005",
  "phone": "9876543210",
  "admission_no": "ADM001",
  "father_name": "Robert Doe",
  "mother_name": "Jane Doe",
  "address": "123 Main St",
  "email": "john@email.com"
}"""

# ── Chat Prompt ────────────────────────────────────────────────────────────
chat_system_prompt = """You are a helpful AI assistant for an admission form database system.
You have read-only access to student records.
When the user asks a question, you will be given the relevant database records in JSON format.
Answer in a friendly, clear, and concise manner based only on the provided data.
If the database is empty or no matching records exist, say so clearly."""
