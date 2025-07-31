import os
import hashlib
import mimetypes
from typing import List, Dict, Any, Optional
import logging

# Core libraries
import pinecone
from sentence_transformers import SentenceTransformer
import numpy as np

# File processing libraries
import PyPDF2
from docx import Document
import openpyxl
from pptx import Presentation
from PIL import Image
import pytesseract
import json
import csv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import re

class FAISSRetriever:
    def __init__(self, text: str, embedding_model: str = 'all-MiniLM-L6-v2', chunk_size: int = 500, chunk_overlap: int = 50):
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
        
        print(f"✅ Created FAISS index with {len(self.chunks)} chunks")
    
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
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
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
    
    def retrieve_with_context(self, query: str, top_k: int = 3) -> List[dict]:
        """
        Retrieve chunks with additional context information
        
        Returns:
            List of dictionaries with chunk info
        """
        results = self.retrieve(query, top_k)
        
        detailed_results = []
        for i, (chunk, score) in enumerate(results):
            detailed_results.append({
                'rank': i + 1,
                'chunk': chunk,
                'similarity_score': score,
                'chunk_length': len(chunk),
                'preview': chunk[:100] + "..." if len(chunk) > 100 else chunk
            })
        
        return detailed_results

# Usage example:
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
        chunk_size=500,  # Adjust as needed
        chunk_overlap=50  # Adjust as needed
    )
    return retriever

import numpy as np
import faiss

# file_path = "/content/BAJHLIP23020V012223.pdf"
import requests
import io

def extract_pdf_from_url(url: str) -> str:
   """Extract text from PDF URL"""
   try:
       # Download PDF from URL
       response = requests.get(url)
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
       print(f"Error extracting PDF from URL: {str(e)}")
       return ""

# Usage:
pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

# result = extract_pdf_from_url(pdf_url)
# retriever = create_retriever_from_result(result)
def extract_pdf_content(file_path: str) -> str:
        """Extract text from PDF files"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting PDF content: {str(e)}")
            return ""

# Process the file
#result = extract_pdf_content(file_path)
result = extract_pdf_from_url(pdf_url)
print("PDF content extracted successfully.")
retriever = create_retriever_from_result(result)
print("FAISS retriever created successfully.")

# Step 2: Query for relevant chunks
user_query = "will I get the cover if my wife died while delivery of the child"
relevant_chunks = retriever.retrieve(user_query, top_k=3)

# Step 3: Use the results
for i, (chunk, score) in enumerate(relevant_chunks, 1):
    print(f"Chunk {i} (Score: {score:.3f}):")
    print(chunk)
    print("-" * 50)

#llama-3.1-8b-instant
# For reasoning models
# llama-4-maverick-17b-128e-instruct

# reasoning model - setting the max reasoning tokens and all for efficiency

import json
from groq import Groq

from langchain_groq import ChatGroq
import json

def process_pdf_queries(pdf_url: str, queries_json: str, groq_api_key: str) -> str:
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
            return json.dumps({"error": "Failed to extract PDF content"})

        # Create retriever (done once)
        retriever = create_retriever_from_result(result)

        # Parse queries
        queries_data = json.loads(queries_json)
        queries = queries_data.get("queries", [])

        answers = {}

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
        for query in queries:
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
            answers[query] = answer

        # Return compiled answers in clean JSON structure
        return json.dumps({"answers": answers}, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})

# Usage example:
pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

queries_json = json.dumps({
    "queries": [
        "will I get the cover if my wife died while delivery of the child",
        "what is the premium amount",
        "what are the exclusions in this policy"
    ]
})

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Get the API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    print("Error: GROQ_API_KEY not found in .env file.")
else:
    # Execute
    result = process_pdf_queries(pdf_url, queries_json, groq_api_key)
    print(result)
