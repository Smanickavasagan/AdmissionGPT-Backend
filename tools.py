import json
from db import documents_collection

def save_to_mongodb(filename: str, structured_data: dict) -> dict:
    """Save structured admission form data to MongoDB."""
    doc_data = {
        "filename": filename,
        "studentName":   {"value": structured_data.get("student_name", "Unknown"),   "confidence": 0.95},
        "dateOfBirth":   {"value": structured_data.get("date_of_birth", "Unknown"),  "confidence": 0.90},
        "phone":         {"value": structured_data.get("phone", "Unknown"),           "confidence": 0.95},
        "admissionNo":   {"value": structured_data.get("admission_no", "Unknown"),   "confidence": 0.90},
        "fatherName":    {"value": structured_data.get("father_name", "Unknown"),    "confidence": 0.90},
        "motherName":    {"value": structured_data.get("mother_name", "Unknown"),    "confidence": 0.90},
        "address":       {"value": structured_data.get("address", "Unknown"),        "confidence": 0.85},
        "email":         {"value": structured_data.get("email", "Unknown"),          "confidence": 0.95},
    }
    result = documents_collection.insert_one(doc_data)
    return {
        "success": True,
        "inserted_id": str(result.inserted_id),
        "structured_data": doc_data
    }

def query_documents(query_filter: dict = None) -> list:
    """Query MongoDB documents collection."""
    if query_filter is None:
        query_filter = {}
    docs = list(documents_collection.find(query_filter).limit(10))
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs
