"""
RAG System for PDF Processing and Query Answering
Converted from Jupyter notebook for standalone execution
"""

import os
import json
import io
from typing import List, Tuple

# Core libraries
import numpy as np
import requests

# Vector search and embeddings
import faiss
from sentence_transformers import SentenceTransformer

# NeonDB (Postgres) support
import psycopg2
import base64

# Document processing libraries
import PyPDF2

# LLM integration
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present

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

    @staticmethod
    def from_db(chunks, embeddings, embedding_model='all-MiniLM-L6-v2', chunk_size=500, chunk_overlap=50):
        """
        Create FAISSRetriever from chunks and embeddings loaded from DB
        """
        obj = FAISSRetriever.__new__(FAISSRetriever)
        obj.text = None
        obj.chunk_size = chunk_size
        obj.chunk_overlap = chunk_overlap
        obj.embedding_model = SentenceTransformer(embedding_model)
        obj.chunks = chunks
        obj.embeddings = embeddings
        obj.index = obj._create_faiss_index(embeddings)
        return obj

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

def neon_connect():
    # Get NeonDB connection string from environment
    conn_str = os.getenv("NEONDB_CONN")
    if not conn_str:
        raise Exception("NEONDB_CONN environment variable not set")
    return psycopg2.connect(conn_str)

def ensure_table():
    conn = neon_connect()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS pdf_embeddings (
            pdf_url TEXT PRIMARY KEY,
            chunks_json TEXT,
            embeddings_bytea BYTEA
        )
    ''')
    conn.commit()
    cur.close()
    conn.close()

def store_embeddings(pdf_url, chunks, embeddings):
    ensure_table()
    conn = neon_connect()
    cur = conn.cursor()
    # Serialize chunks and embeddings
    chunks_json = json.dumps(chunks)
    embeddings_bytes = embeddings.tobytes()
    cur.execute('''
        INSERT INTO pdf_embeddings (pdf_url, chunks_json, embeddings_bytea)
        VALUES (%s, %s, %s)
        ON CONFLICT (pdf_url) DO UPDATE SET chunks_json=EXCLUDED.chunks_json, embeddings_bytea=EXCLUDED.embeddings_bytea
    ''', (pdf_url, chunks_json, psycopg2.Binary(embeddings_bytes)))
    conn.commit()
    cur.close()
    conn.close()

def load_embeddings(pdf_url):
    ensure_table()
    conn = neon_connect()
    cur = conn.cursor()
    cur.execute('SELECT chunks_json, embeddings_bytea FROM pdf_embeddings WHERE pdf_url=%s', (pdf_url,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        chunks = json.loads(row[0])
        # Infer shape from number of chunks and embedding dim
        embeddings_bytes = row[1]
        # Need to get embedding dimension from model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        dim = embedding_model.get_sentence_embedding_dimension()
        embeddings = np.frombuffer(embeddings_bytes, dtype=np.float32).reshape((len(chunks), dim))
        return chunks, embeddings
    return None, None

def create_retriever_from_result(result: str, pdf_url: str = None) -> FAISSRetriever:
    """
    Create FAISS retriever from extracted PDF text, using NeonDB for caching
    """
    if pdf_url:
        chunks, embeddings = load_embeddings(pdf_url)
        if chunks and embeddings is not None:
            return FAISSRetriever.from_db(chunks, embeddings)
    retriever = FAISSRetriever(
        text=result,
        chunk_size=1000,
        chunk_overlap=100
    )
    if pdf_url:
        store_embeddings(pdf_url, retriever.chunks, retriever.embeddings)
    return retriever


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
        # Create retriever (uses NeonDB for caching)
        retriever = create_retriever_from_result(result, pdf_url)
        
        # Parse queries
        queries_data = json.loads(queries_json)
        queries = queries_data.get("queries", [])
        
        answers = {}
        
        # Optimized examples for concise responses
        examples = """You are a policy expert providing precise, actionable answers. Follow these exact formats:

## Format Patterns:

**Time-based queries:**
Q: What is the grace period for premium payment?
A: 30 days grace period after due date.

Q: What is the waiting period for pre-existing diseases?  
A: 36 months continuous coverage required.

Q: What is the waiting period for cataract surgery?
A: 2 years waiting period.

**Coverage queries:**
Q: Does this policy cover maternity expenses?
A: Yes, after 24 months continuous coverage. Limited to 2 deliveries per policy period.

Q: Are dental procedures covered?
A: Yes, up to ₹10,000 annually after 6 months waiting period.

**Claim-related queries:**
Q: What documents are required for claim submission?
A: Original bills, discharge summary, diagnostic reports, and claim form within 30 days.

Q: What is the claim settlement timeframe?
A: 15-30 days after complete documentation received.

**Exclusion queries:**
Q: Is cosmetic surgery covered?
A: No, cosmetic procedures are excluded unless medically necessary.

**Limit queries:**
Q: What is the annual coverage limit?
A: ₹5 lakhs per policy year with ₹1 lakh sub-limit for OPD expenses."""

        
        # Process each query individually
        for query in queries:
            # Get relevant chunks for this specific query
            relevant_chunks = retriever.retrieve(query, top_k=2)  # Reduced to 2 for faster processing
            
            # Combine chunks for context
            context = "\n\n".join([chunk for chunk, score in relevant_chunks])
            
            # Create focused prompt for concise answers
            prompt = f"""{examples}

Based on the context below, provide a concise, specific answer:
## Instructions:
- Provide direct, specific answers using the exact format shown above
- Include relevant numbers, timeframes, and conditions
- Keep responses under 20-50 words when possible
- If information is not in context, state "Information not available in policy documents"
- Use active voice and clear terminology

## Context:
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


# def main():
#     """
#     Main function to run the RAG system
#     """
#     # Configuration
#     pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
#     # Get GROQ API key from environment variable
#     groq_api_key = os.getenv("GROQ_API_KEY")
#     if not groq_api_key:
#         print("Error: GROQ_API_KEY environment variable not set")
#         return

#     # Sample queries
#     queries_json = json.dumps({
#         "queries": [
#             "will I get the cover if my wife died while delivery of the child",
#             "what is the premium amount",
#             "what are the exclusions in this policy"
#         ]
#     })

#     print("🚀 Starting RAG System...")
#     print("📄 Extracting PDF content...")
    
#     # Extract PDF content
#     result = extract_pdf_from_url(pdf_url)
#     if not result:
#         print("❌ Failed to extract PDF content")
#         return
    
#     print("✅ PDF content extracted successfully.")
    
#     # Create retriever
#     print("🔍 Creating FAISS retriever...")
#     retriever = create_retriever_from_result(result)
#     print("✅ FAISS retriever created successfully.")

#     # Test single query retrieval
#     print("\n🔍 Testing retrieval for sample query...")
#     user_query = "will I get the cover if my wife died while delivery of the child"
#     relevant_chunks = retriever.retrieve(user_query, top_k=3)

#     print(f"\nRelevant chunks for: '{user_query}'")
#     for i, (chunk, score) in enumerate(relevant_chunks, 1):
#         print(f"\nChunk {i} (Score: {score:.3f}):")
#         print(chunk)
#         print("-" * 50)

#     # Process queries with LLM
#     print("\n🤖 Processing queries with LLM...")
#     result_json = process_pdf_queries(pdf_url, queries_json, groq_api_key)
#     print("\n📋 Query Results:")
#     print(result_json)


# if __name__ == "__main__":
#     main()
