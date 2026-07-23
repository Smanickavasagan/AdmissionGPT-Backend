import json
from db import documents_collection

def save_to_mongodb(filename: str, structured_data: dict) -> dict:
    """Save structured admission form data to MongoDB."""
    def get_val(*keys, default="Unknown"):
        for k in keys:
            if k in structured_data and structured_data[k] is not None and str(structured_data[k]).strip() != "":
                val = structured_data[k]
                if isinstance(val, dict) and "value" in val:
                    return val["value"]
                return str(val).strip()
        return default

    doc_data = {
        "filename": filename,
        "studentName":   {"value": get_val("student_name", "studentName", "student", "name"), "confidence": 0.95},
        "dateOfBirth":   {"value": get_val("date_of_birth", "dateOfBirth", "dob", "birth_date"), "confidence": 0.90},
        "phone":         {"value": get_val("phone", "phone_number", "phoneNo", "mobile", "contact"), "confidence": 0.95},
        "admissionNo":   {"value": get_val("admission_no", "admissionNo", "admission_number", "adm_no", "roll_no"), "confidence": 0.90},
        "fatherName":    {"value": get_val("father_name", "fatherName", "father"), "confidence": 0.90},
        "motherName":    {"value": get_val("mother_name", "motherName", "mother"), "confidence": 0.90},
        "address":       {"value": get_val("address", "residential_address"), "confidence": 0.85},
        "email":         {"value": get_val("email", "email_id"), "confidence": 0.95},
    }
    # Store any extra fields extracted by Qwen
    standard_keys = {
        "student_name", "studentName", "student", "name",
        "date_of_birth", "dateOfBirth", "dob", "birth_date",
        "phone", "phone_number", "phoneNo", "mobile", "contact",
        "admission_no", "admissionNo", "admission_number", "adm_no", "roll_no",
        "father_name", "fatherName", "father",
        "mother_name", "motherName", "mother",
        "address", "residential_address",
        "email", "email_id"
    }
    for k, v in structured_data.items():
        if k not in standard_keys and k not in doc_data:
            doc_data[k] = {"value": str(v), "confidence": 0.85}
            
    result = documents_collection.insert_one(doc_data)
    doc_data["_id"] = str(result.inserted_id)
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

