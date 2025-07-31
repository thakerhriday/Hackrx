from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import importlib.util
import sys
import os

# Ensure NEONDB_CONN is set in environment for NeonDB access
if not os.getenv("NEONDB_CONN"):
    raise RuntimeError("NEONDB_CONN environment variable must be set to your NeonDB connection string.")

# Dynamically import 'testing-pyt.py' as a module
module_path = os.path.join(os.path.dirname(__file__), 'testing-pyt.py')
spec = importlib.util.spec_from_file_location('testing_pyt', module_path)
testing_pyt = importlib.util.module_from_spec(spec)
sys.modules['testing_pyt'] = testing_pyt
spec.loader.exec_module(testing_pyt)

extract_pdf_from_url = testing_pyt.extract_pdf_from_url
create_retriever_from_result = testing_pyt.create_retriever_from_result
process_pdf_queries = testing_pyt.process_pdf_queries

app = FastAPI()

class HackrxRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def run_hackrx(
    req: HackrxRequest,
    authorization: Optional[str] = Header(None)
):
    # Check for Bearer token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    groq_api_key = authorization.split(" ", 1)[1]
    if not groq_api_key:
        raise HTTPException(status_code=401, detail="Missing API key in Authorization header")

    # Set the API key in environment for downstream code
    os.environ["GROQ_API_KEY"] = groq_api_key

    # Prepare queries JSON
    queries_json = json.dumps({"queries": req.questions})
    
    # Process queries using the existing logic
    result_json = process_pdf_queries(req.documents, queries_json, groq_api_key)
    try:
        result = json.loads(result_json)
    except Exception:
        raise HTTPException(status_code=500, detail="Internal error in processing queries")

    # If error in result, propagate
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    # Convert answers dict to list in the order of questions
    answers_dict = result.get("answers", {})
    answers = [answers_dict.get(q, "Information not available in policy documents") for q in req.questions]
    return JSONResponse(content={"answers": answers})
