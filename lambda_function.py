import os

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"
os.environ["HF_METRICS_CACHE"] = "/tmp/huggingface/metrics"
os.makedirs("/tmp/huggingface", exist_ok=True)

import json
import logging

import requests
import io
import faiss

from sentence_transformers import SentenceTransformer
import PyPDF2
from langchain_groq import ChatGroq

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---- Core code for retriever, extraction, and question answering ----

class FAISSRetriever:
    def __init__(self, text, embedding_model='all-MiniLM-L6-v2', chunk_size=500, chunk_overlap=50):
        self.text = text
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunks = self._chunk_text(text)
        self.embeddings = self._create_embeddings(self.chunks)
        self.index = self._create_faiss_index(self.embeddings)

    def _chunk_text(self, text):
        if len(text) <= self.chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            # Try to end at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                boundary = max(last_period, last_newline)
                if boundary > start + self.chunk_size // 2:
                    chunk = text[start:start + boundary + 1]
                    end = start + boundary + 1
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        return chunks

    def _create_embeddings(self, chunks):
        return self.embedding_model.encode(chunks, convert_to_numpy=True)
    
    def _create_faiss_index(self, embeddings):
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return index

    def retrieve(self, query, top_k=2):
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results

# PDF extraction
def extract_pdf_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    pdf_file = io.BytesIO(response.content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Main QA logic
def process_pdf_queries(pdf_url, questions, groq_api_key):
    # Initialize LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=150
    )
    # Extract PDF once
    result = extract_pdf_from_url(pdf_url)
    retriever = FAISSRetriever(result, chunk_size=500, chunk_overlap=50)
    answers = []
    examples = """Examples of concise policy answers:

Q: What is the grace period for premium payment?
A: 30 days grace period after due date.

Q: What is the waiting period for pre-existing diseases?
A: 36 months continuous coverage required.

Q: Does this policy cover maternity expenses?
A: Yes, after 24 months continuous coverage. Limited to 2 deliveries per policy period.

Q: What is the waiting period for cataract surgery?
A: 2 years waiting period."""
    for query in questions:
        relevant_chunks = retriever.retrieve(query, top_k=2)
        context = "\n\n".join([chunk for chunk, score in relevant_chunks])
        prompt = f"""{examples}

Based on the context below, provide a concise, specific answer:

Context:
{context}

Q: {query}
A:"""
        response = llm.invoke(prompt)
        answer = response.content.strip()
        # Clean output
        if answer.startswith("Q:"):
            a_index = answer.find("A:")
            if a_index != -1:
                answer = answer[a_index + 2:].strip()
        elif answer.startswith("A:"):
            answer = answer[2:].strip()
        answers.append(answer)
    return answers

# ---- AWS Lambda entrypoint ----

def lambda_handler(event, context):
    try:
        if event.get("body"):
            body = event["body"]
            if event.get("isBase64Encoded"):
                import base64
                body = base64.b64decode(body).decode()
            data = json.loads(body)
        else:
            data = event

        # Input validation
        pdf_url = data.get("documents")
        questions = data.get("questions", [])
        groq_api_key = "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"
        if not groq_api_key:
            return {
                'statusCode': 500,
                'body': json.dumps({"error": "GROQ_API_KEY not configured"})
            }
        if not pdf_url or not questions or not isinstance(questions, list):
            return {
                'statusCode': 400,
                'body': json.dumps({"error": "Both 'documents' URL and 'questions' list are required."})
            }

        # Answer questions
        answers = process_pdf_queries(pdf_url, questions, groq_api_key)
        response = {"answers": answers}

        return {
            'statusCode': 200,
            'body': json.dumps(response),
            'headers': {"Content-Type": "application/json"}
        }
    except Exception as e:
        logger.exception("Unhandled Lambda exception:")
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)}),
            'headers': {"Content-Type": "application/json"}
        }
