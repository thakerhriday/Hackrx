import os

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"
os.environ["HF_METRICS_CACHE"] = "/tmp/huggingface/metrics"
os.makedirs("/tmp/huggingface", exist_ok=True)

import json
import logging
import re
import requests
import io
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2
from langchain_groq import ChatGroq

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class AdvancedFAISSRetriever:
    def __init__(self, text, embedding_model='all-MiniLM-L6-v2', chunk_size=1000, chunk_overlap=200):
        self.text = text
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Preprocess text for better chunking
        self.preprocessed_text = self._preprocess_text(text)
        self.chunks = self._smart_chunk_text(self.preprocessed_text)
        
        # Create embeddings and index
        self.embeddings = self._create_embeddings(self.chunks)
        self.index = self._create_faiss_index(self.embeddings)
        
        logger.info(f"Created {len(self.chunks)} chunks for retrieval")

    def _preprocess_text(self, text):
        """Clean and preprocess text for better chunking"""
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'(\.)([A-Z])', r'\1 \2', text)     # Add space after period
        
        return text.strip()

    def _smart_chunk_text(self, text):
        """Improved chunking strategy that respects document structure"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        
        # First, try to split by major sections (headers, numbered items, etc.)
        sections = re.split(r'\n(?=\d+\.|\b[A-Z][A-Z\s]{3,}:|\b(?:SECTION|ARTICLE|CHAPTER)\b)', text)
        
        for section in sections:
            if len(section) <= self.chunk_size:
                if section.strip():
                    chunks.append(section.strip())
            else:
                # Further split large sections
                sub_chunks = self._split_large_section(section)
                chunks.extend(sub_chunks)
        
        return chunks

    def _split_large_section(self, text):
        """Split large sections while maintaining context"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk + sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Add overlap between chunks for continuity
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0 and len(chunks) > 1:
                # Add overlap from previous chunk
                prev_words = chunks[i-1].split()[-20:]  # Last 20 words
                overlap = " ".join(prev_words)
                chunk = overlap + " " + chunk
            overlapped_chunks.append(chunk)
        
        return overlapped_chunks

    def _create_embeddings(self, chunks):
        return self.embedding_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    
    def _create_faiss_index(self, embeddings):
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return index

    def retrieve(self, query, top_k=5):
        """Enhanced retrieval with relevance scoring"""
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, min(top_k * 2, len(self.chunks)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks) and score > 0.3:  # Relevance threshold
                results.append((self.chunks[idx], float(score)))
        
        # Return top_k most relevant chunks
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

def extract_pdf_from_url(url):
    """Enhanced PDF extraction with better text processing"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():  # Only add non-empty pages
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text
        
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        raise

def clean_answer(answer):
    """Clean and format the answer"""
    # Remove common prefixes
    prefixes_to_remove = ["ANSWER:", "A:", "Answer:", "Based on the context,", "According to the document,"]
    
    for prefix in prefixes_to_remove:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    # Remove quotes if the entire answer is quoted
    if answer.startswith('"') and answer.endswith('"'):
        answer = answer[1:-1]
    
    return answer

def process_pdf_queries(pdf_url, questions, groq_api_key):
    """Main QA processing with improved prompting"""
    
    # Initialize LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0,  # More deterministic
        max_tokens=200
    )
    
    # Extract PDF and create retriever
    pdf_text = extract_pdf_from_url(pdf_url)
    retriever = AdvancedFAISSRetriever(pdf_text, chunk_size=1000, chunk_overlap=200)
    
    answers = []
    
    for query in questions:
        try:
            # Retrieve relevant chunks
            relevant_chunks = retriever.retrieve(query, top_k=3)
            
            if not relevant_chunks:
                answers.append("I couldn't find relevant information in the document to answer this question.")
                continue
            
            # Prepare context from retrieved chunks
            context_parts = []
            for i, (chunk, score) in enumerate(relevant_chunks):
                context_parts.append(f"[Relevant Section {i+1}]:\n{chunk}")
            
            context = "\n\n".join(context_parts)
            
            # Improved prompt without examples that could bias answers
            prompt = f"""You are an expert document analyst. Based ONLY on the provided context, answer the question accurately and concisely.

IMPORTANT GUIDELINES:
- Use only information explicitly stated in the context
- If the answer is not in the context, say "This information is not available in the provided document"
- Be specific with numbers, dates, and percentages when mentioned
- Keep answers focused and direct
- Do not make assumptions or add external knowledge

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

            # Get response from LLM
            response = llm.invoke(prompt)
            answer = response.content.strip()
            
            # Clean up the answer - FIXED: removed self reference
            answer = clean_answer(answer)
            answers.append(answer)
            
        except Exception as e:
            logger.error(f"Error processing question '{query}': {str(e)}")
            answers.append("An error occurred while processing this question.")
    
    return answers

# ---- AWS Lambda entrypoint ----

def lambda_handler(event, context):
    try:
        # Parse input
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
        groq_api_key = "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"  # Consider using environment variable
        
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

        # Process questions
        answers = process_pdf_queries(pdf_url, questions, groq_api_key)
        
        response = {
            "answers": answers,
            "document_processed": True,
            "total_questions": len(questions)
        }

        return {
            'statusCode': 200,
            'body': json.dumps(response),
            'headers': {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        }
        
    except Exception as e:
        logger.exception("Unhandled Lambda exception:")
        return {
            'statusCode': 500,
            'body': json.dumps({"error": f"Processing failed: {str(e)}"}),
            'headers': {"Content-Type": "application/json"}
        }
