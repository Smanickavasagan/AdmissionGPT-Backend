# Algorithms and AI Methods Used

Form2Database combines traditional OCR, Large Language Models, structured-data extraction, and document-database querying.

## 1. Optical Character Recognition (OCR)

**Algorithm/Technology:** RapidOCR

RapidOCR is used to recognize text from uploaded admission-form images.

```text
Input Image
     ↓
RapidOCR
     ↓
Detected Text Lines
     ↓
Combined OCR Text
```

The application initializes the OCR engine with `RapidOCR()` and combines the recognized text lines into a single text string.

### Purpose

OCR converts information from a visual admission form into machine-readable text so that the AI model can process it.

---

## 2. LLM-Based Information Extraction

**Model:** Qwen 2.5
**Default model:** `qwen2.5:7b`
**Runtime:** Ollama

The system uses Qwen to transform unstructured OCR text into structured JSON.

```text
OCR Text
   ↓
Qwen 2.5
   ↓
Prompt-Based Information Extraction
   ↓
Structured JSON
```

The extraction prompt specifies the fields that should be extracted:

```text
student_name
date_of_birth
phone
admission_no
father_name
mother_name
address
email
```

If a field cannot be found, the model is instructed to return `"Unknown"`. The model is also instructed to return JSON without Markdown or additional explanation.

### Important Note

This part is **LLM-based information extraction**, not a traditional machine-learning classification algorithm. The provided source code does not implement or train a custom neural-network algorithm.

---

## 3. Prompt-Based Structured Data Extraction

The application uses a predefined system prompt to control how Qwen interprets OCR text.

The prompt defines:

1. Which fields to extract.
2. What to return when a field is missing.
3. The required JSON structure.
4. The required output format.

This creates a structured extraction pipeline:

```text
Unstructured OCR Text
        ↓
Extraction Prompt
        ↓
Qwen 2.5
        ↓
Structured JSON
```

The backend sends the OCR text together with the extraction prompt to the Ollama `/api/generate` endpoint.

---

## 4. Regex-Based JSON Recovery

**Algorithm:** Regular Expression (Regex) pattern matching

The application contains a fallback mechanism for extracting JSON when the Qwen response contains extra text or Markdown formatting.

The process is:

```text
Qwen Response
     ↓
Remove Markdown Code Fences
     ↓
Try JSON Parsing
     ↓
If Parsing Fails
     ↓
Regex Search for {...}
     ↓
Parse Extracted JSON
```

The implementation first removes possible Markdown code fences and attempts `json.loads()`. If that fails, a regular expression searches for the first `{...}` block before attempting JSON parsing again.

---

## 5. Field Normalization Algorithm

The application normalizes different possible names for the same piece of information.

For example, student name can be recognized from several possible keys:

```text
student_name
studentName
student
name
```

Similarly, admission number can be obtained from:

```text
admission_no
admissionNo
admission_number
adm_no
roll_no
```

The `get_val()` function checks these alternatives and returns the first valid value it finds. If none is available, it returns `"Unknown"`.

This can be described as a **rule-based field normalization/mapping algorithm**.

---

## 6. Confidence Assignment

The database representation includes a confidence value for extracted fields.

For example:

```json
{
  "studentName": {
    "value": "John Doe",
    "confidence": 0.95
  }
}
```

The current implementation assigns predefined confidence values to standard fields rather than calculating confidence dynamically from OCR or Qwen probabilities.

Therefore, these values should be considered **application-defined confidence scores**, not model-generated confidence probabilities.

---

## 7. Natural-Language Database Querying

**Method:** Context-based LLM question answering

The `/chat` endpoint retrieves database records and supplies selected fields to Qwen together with the user's question.

```text
User Question
      ↓
MongoDB Records
      ↓
Select Relevant Fields
      ↓
Create Context
      ↓
Qwen 2.5
      ↓
Natural-Language Answer
```

The implementation currently sends up to the latest five records to the model and includes fields such as filename, student name, admission number, phone number, and date of birth.

The chat prompt instructs the model to answer only from the provided database records.

---

# Algorithm Summary

| Component          | Algorithm / Method                       | Purpose                                        |
| ------------------ | ---------------------------------------- | ---------------------------------------------- |
| Image → Text       | **OCR using RapidOCR**                   | Extract text from admission forms              |
| Text → Information | **LLM-based information extraction**     | Identify student information                   |
| Information → JSON | **Prompt-based structured extraction**   | Produce standardized JSON                      |
| JSON Recovery      | **Regex + JSON parsing**                 | Recover JSON from imperfect LLM output         |
| Field Mapping      | **Rule-based normalization**             | Map alternative field names to standard fields |
| Confidence         | **Rule-based fixed scoring**             | Attach application-defined confidence values   |
| Database           | **MongoDB document querying**            | Store and retrieve records                     |
| User Questions     | **Context-based LLM question answering** | Answer questions about stored records          |

## Overall Processing Algorithm

The complete system can be represented as:

```text
              ┌─────────────────────┐
              │ Admission Form Image│
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │     RapidOCR        │
              │   OCR Processing    │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │     OCR Text        │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │     Qwen 2.5        │
              │ Information Extract │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │   Structured JSON   │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │ Field Normalization │
              │ + Confidence Values │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │      MongoDB        │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │ Natural-Language    │
              │      Queries        │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │     Qwen 2.5        │
              │    Final Answer     │
              └─────────────────────┘
```

### Key Technologies

* **Python**
* **FastAPI**
* **RapidOCR**
* **Qwen 2.5**
* **Ollama**
* **MongoDB**
* **PyMongo**
* **LangChain Ollama**
* **HTTPX**
* **Pydantic**

These technologies are directly reflected in the supplied source code.
