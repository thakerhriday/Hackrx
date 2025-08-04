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
import google.generativeai as genai

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

def parse_multi_answer_response(response_text, num_questions):
    """Parse multiple answers from a single API response"""
    answers = []
    
    # Try to split by numbered format first
    lines = response_text.strip().split('\n')
    current_answer = ""
    answer_started = False
    
    for line in lines:
        line = line.strip()
        
        # Check if line starts with a number (1., 2., etc.)
        if re.match(r'^\d+\.?\s+', line):
            if current_answer and answer_started:
                answers.append(clean_answer(current_answer.strip()))
            current_answer = re.sub(r'^\d+\.?\s+', '', line)
            answer_started = True
        elif answer_started:
            current_answer += " " + line
    
    # Add the last answer
    if current_answer and answer_started:
        answers.append(clean_answer(current_answer.strip()))
    
    # If parsing failed, try alternative splitting methods
    if len(answers) != num_questions:
        # Try splitting by double newlines
        alt_answers = [clean_answer(ans.strip()) for ans in response_text.split('\n\n') if ans.strip()]
        if len(alt_answers) == num_questions:
            return alt_answers
        
        # Try splitting by single newlines and filtering
        alt_answers = []
        for line in response_text.split('\n'):
            line = line.strip()
            if line and not re.match(r'^(ANSWERS?:?|Based on|According to)', line, re.IGNORECASE):
                # Remove number prefixes
                line = re.sub(r'^\d+\.?\s*', '', line)
                if line:
                    alt_answers.append(clean_answer(line))
        
        if len(alt_answers) >= num_questions:
            return alt_answers[:num_questions]
    
    # Ensure we have the right number of answers
    while len(answers) < num_questions:
        answers.append("Answer not found in response.")
    
    return answers[:num_questions]

def process_pdf_queries_optimized(pdf_url, questions, gemini_api_key):
    """Optimized QA processing that sends all questions in one API call"""
    
    # Initialize Gemini
    genai.configure(api_key=gemini_api_key)
    
    # Configure generation parameters
    generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 1000,  # Increased for multiple answers
    }
    
    # Safety settings
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]
    
    # Initialize the model
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-lite",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    # Extract PDF and create retriever
    pdf_text = extract_pdf_from_url(pdf_url)
    retriever = AdvancedFAISSRetriever(pdf_text, chunk_size=1000, chunk_overlap=200)
    
    # Get all unique relevant chunks for ALL questions
    all_relevant_chunks = set()
    combined_query = " ".join(questions)  # Combine all questions for better retrieval
    
    # Retrieve for combined query
    combined_chunks = retriever.retrieve(combined_query, top_k=8)
    for chunk, score in combined_chunks:
        if score > 0.3:
            all_relevant_chunks.add(chunk)
    
    # Also retrieve for individual questions to ensure coverage
    for query in questions:
        chunks = retriever.retrieve(query, top_k=3)
        for chunk, score in chunks:
            if score > 0.4:  # Higher threshold for individual queries
                all_relevant_chunks.add(chunk)
    
    # Create single context from all relevant chunks
    context_parts = []
    for i, chunk in enumerate(all_relevant_chunks):
        context_parts.append(f"[Section {i+1}]:\n{chunk}")
    
    context = "\n\n".join(context_parts)
    
    # Format all questions
    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    
    # Create comprehensive prompt
    prompt = f"""You are an expert document analyst. Based ONLY on the provided context, answer ALL questions below accurately and concisely.

IMPORTANT GUIDELINES:
- Use only information explicitly stated in the context
- If any answer is not in the context, say "This information is not available in the provided document"
- Be specific with numbers, dates, and percentages when mentioned
- Answer each question in 1-2 sentences for simple answers, or 2-3 sentences when more detail is essential
- Format your response as numbered answers (1., 2., 3., etc.)
- Do not make assumptions or add external knowledge

CONTEXT:
{context}

QUESTIONS:
{questions_text}

Please provide numbered answers in the following format:
1. [Answer to question 1]
2. [Answer to question 2]
3. [Answer to question 3]
...and so on.

ANSWERS:"""

    try:
        # Get response from Gemini
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Parse individual answers from the response
        answers = parse_multi_answer_response(response_text, len(questions))
        
        logger.info(f"Successfully processed {len(questions)} questions in one API call")
        return answers
        
    except Exception as e:
        logger.error(f"Error processing questions: {str(e)}")
        # Fallback: return error message for all questions
        return [f"An error occurred while processing the questions: {str(e)}"] * len(questions)

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
        gemini_api_key = "YOUR_GEMINI_API_KEY_HERE" 
        
        if not gemini_api_key or gemini_api_key == "YOUR_GEMINI_API_KEY_HERE":
            return {
                'statusCode': 500,
                'body': json.dumps({"error": "GEMINI_API_KEY not configured"})
            }
        
        if not pdf_url or not questions or not isinstance(questions, list):
            return {
                'statusCode': 400,
                'body': json.dumps({"error": "Both 'documents' URL and 'questions' list are required."})
            }

        # Process questions with optimized approach
        answers = process_pdf_queries_optimized(pdf_url, questions, gemini_api_key)
        
        response = {
            "answers": answers,
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
