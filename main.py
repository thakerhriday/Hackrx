from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional
import json
import os
import hashlib
import mimetypes
import logging
from dotenv import load_dotenv

# Import your existing classes and functions
import pinecone
from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2
import requests
import io
import faiss
from groq import Groq
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HackRX PDF Query API",
    description="API for processing PDF documents and answering questions using RAG",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# Pydantic models for request/response
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# Your existing classes (FAISSRetriever, etc.) - keeping them as is
class FAISSRetriever:
    def __init__(self, text: str, embedding_model: str = 'all-MiniLM-L6-v2', chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize FAISS retriever with extracted text
        
        Args:
            text: The extracted text from PDF (your 'result' variable)
            embedding_model: Sentence transformer model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.text = text
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Process text and create index
        self.chunks = self._chunk_text(text)
        self.embeddings = self._create_embeddings(self.chunks)
        self.index = self._create_faiss_index(self.embeddings)
        
        logger.info(f"Created FAISS index with {len(self.chunks)} chunks")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
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
    
    def _create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for all chunks"""
        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)
        return embeddings
    
    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create FAISS index from embeddings"""
        dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (after normalization)
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        index.add(embeddings)
        
        return index
    
    def retrieve(self, query: str, top_k: int = 3) -> List[tuple]:
        """
        Retrieve most relevant chunks for a query
        
        Args:
            query: User query string
            top_k: Number of top chunks to retrieve
        
        Returns:
            List of tuples (chunk_text, similarity_score)
        """
        # Create query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Return results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):  # Valid index
                results.append((self.chunks[idx], float(score)))
        
        return results

# Helper functions
def extract_pdf_from_url(url: str) -> str:
    """Extract text from PDF URL"""
    try:
        # Download PDF from URL
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Create a file-like object from the response content
        pdf_file = io.BytesIO(response.content)

        # Extract text using PyPDF2
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF from URL: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract PDF: {str(e)}")

def create_retriever_from_result(result: str) -> FAISSRetriever:
    """
    Create FAISS retriever from your extracted PDF text
    
    Args:
        result: Your extracted text variable
    
    Returns:
        FAISSRetriever instance
    """
    retriever = FAISSRetriever(
        text=result,
        chunk_size=1000,  # Adjust as needed
        chunk_overlap=100  # Adjust as needed
    )
    return retriever

def process_pdf_queries_optimized(pdf_url: str, questions: List[str], groq_api_key: str) -> List[str]:
    """
    Process PDF from URL and answer queries one by one using vector search + ChatGroq
    Returns concise, point-to-point answers
    """
    try:
        # Initialize ChatGroq with optimized settings for faster response
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=150  # Reduced for shorter answers
        )

        # Extract PDF content (done once)
        result = extract_pdf_from_url(pdf_url)
        if not result:
            raise HTTPException(status_code=400, detail="Failed to extract PDF content")

        # Create retriever (done once)
        retriever = create_retriever_from_result(result)

        answers = []

        # Optimized examples for concise responses
        examples = """Examples of concise policy answers:

Q: What is the grace period for premium payment?
A: 30 days grace period after due date.

Q: What is the waiting period for pre-existing diseases?
A: 36 months continuous coverage required.

Q: Does this policy cover maternity expenses?
A: Yes, after 24 months continuous coverage. Limited to 2 deliveries per policy period.

Q: What is the waiting period for cataract surgery?
A: 2 years waiting period."""

        # Process each query individually
        for query in questions:
            # Get relevant chunks for this specific query
            relevant_chunks = retriever.retrieve(query, top_k=2)  # Reduced to 2 for faster processing

            # Combine chunks for context
            context = "\n\n".join([chunk for chunk, score in relevant_chunks])

            # Create focused prompt for concise answers
            prompt = f"""{examples}

Based on the context below, provide a concise, specific answer:

Context:
{context}

Q: {query}
A:"""

            # Get answer from ChatGroq
            response = llm.invoke(prompt)
            answer = response.content.strip()

            # Clean the answer - remove any Q: and A: prefixes if present
            if answer.startswith("Q:"):
                # Find the A: part and extract only the answer
                a_index = answer.find("A:")
                if a_index != -1:
                    answer = answer[a_index + 2:].strip()
            elif answer.startswith("A:"):
                # Remove A: prefix
                answer = answer[2:].strip()

            # Store only the clean answer
            answers.append(answer)

        return answers

    except Exception as e:
        logger.error(f"Error processing queries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing queries: {str(e)}")

# Authentication dependency
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify the API key from Authorization header
    """
    expected_api_key = os.getenv("API_KEY")
    if not expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured on server"
        )
    
    if credentials.credentials != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return credentials.credentials

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "HackRX PDF Query API is running"}

# Main endpoint
@app.post(
    "/hackrx/run",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)
async def process_document_queries(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Process PDF document and answer questions using RAG
    
    Args:
        request: QueryRequest containing document URL and questions
        api_key: API key from Authorization header
    
    Returns:
        QueryResponse with answers to all questions
    """
    try:
        # Validate environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GROQ API key not configured"
            )

        # Validate request
        if not request.questions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one question is required"
            )

        if len(request.questions) > 20:  # Reasonable limit
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Too many questions. Maximum 20 allowed per request"
            )

        logger.info(f"Processing {len(request.questions)} questions for document: {request.documents}")

        # Process the queries
        answers = process_pdf_queries_optimized(
            pdf_url=str(request.documents),
            questions=request.questions,
            groq_api_key=groq_api_key
        )

        logger.info(f"Successfully processed {len(answers)} answers")

        return QueryResponse(answers=answers)

    except HTTPException:
        # Re-raise HTTP exceptions as they are
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {"error": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {"error": "Internal server error", "detail": str(exc)}

if __name__ == "__main__":
    import uvicorn
    
    # Check if required environment variables are set
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY environment variable is required")
        exit(1)
    
    if not os.getenv("API_KEY"):
        logger.warning("API_KEY environment variable not set. Authentication will be disabled.")
    
    uvicorn.run(
        "main:app",  # Replace 'main' with your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
