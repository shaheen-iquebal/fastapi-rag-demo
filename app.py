import os
import json
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# Load environment variables
load_dotenv()

# --- Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-nano-9b-v2:free")
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))

# --- Initialize Clients ---
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- FastAPI App ---
app = FastAPI(
    title="RAG Book Query API",
    description="API for querying educational books using RAG",
    version="1.0.0"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "https://your-frontend-domain.com",  # Replace with your actual domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class QuestionRequest(BaseModel):
    question: str
    book_title: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

class BookSuggestion(BaseModel):
    book_title: str
    similarity: float
    is_current_book: bool = False

class SearchResult(BaseModel):
    book_title: str
    chapter: str
    topic: str
    content: str
    similarity: Optional[float] = None

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    suggestions: Optional[List[BookSuggestion]] = None
    current_book: Optional[str] = None
    similarity_score: Optional[float] = None

class BooksResponse(BaseModel):
    books: List[str]

class HealthResponse(BaseModel):
    status: str
    message: str

# --- Helper Functions ---

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a text string."""
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[
                {
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ],
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

def search_in_book(question: str, book_title: Optional[str] = None) -> List[Dict]:
    """Search for content in a specific book (or all books)."""
    try:
        # Generate embedding for the question
        question_embedding = generate_embedding(question)
        
        # Try using the RPC function first
        results = supabase.rpc(
            'match_books',
            {
                'query_embedding': question_embedding,
                'match_threshold': 0.0,
                'match_count': MAX_RESULTS,
                'filter_book': book_title
            }
        ).execute()
        return results.data
    except Exception as e:
        print(f"RPC function error: {e}")
        # Fallback: direct query
        query = supabase.table('books').select(
            'book_title, chapter, topic, content'
        )
        if book_title:
            query = query.eq('book_title', book_title)
        return query.limit(MAX_RESULTS).execute().data

def suggest_book(question: str, current_book: str) -> tuple[Optional[str], float]:
    """Find the best book for a question."""
    try:
        all_results = search_in_book(question, book_title=None)
        
        if not all_results:
            return None, 0
        
        # Find the best match
        best_match = max(all_results, key=lambda x: x.get('similarity', 0))
        best_similarity = best_match.get('similarity', 0)
        best_book = best_match.get('book_title', 'Unknown')
        
        return best_book, best_similarity
    except Exception as e:
        print(f"Error suggesting book: {e}")
        return None, 0

def build_retrieval_prompt(question: str, context_results: List[Dict]) -> str:
    """Build the prompt for the LLM."""
    context_parts = []
    
    for i, result in enumerate(context_results, 1):
        context_parts.append(f"""
[Source {i}]
Book: {result.get('book_title', 'Unknown')}
Chapter: {result.get('chapter', 'N/A')}
Topic: {result.get('topic', 'N/A')}

Content:
{result.get('content', '')}
""")
    
    context = "\n".join(context_parts)
    
    return f"""
You are an educational assistant.

Answer ONLY from the supplied context.

Rules:
- Never use outside knowledge.
- If the answer isn't present, say so.
- Write a natural, easy-to-read explanation.
- At the end mention the sources you used.

Context:

{context}

Question:
{question}

Answer:
"""

def generate_answer(question: str, retrieved_results: List[Dict]) -> str:
    """Generate an answer using the LLM."""
    try:
        prompt = build_retrieval_prompt(question, retrieved_results)
        
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You answer ONLY using the retrieved context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")

def get_all_books() -> List[str]:
    """Fetch all unique book titles."""
    try:
        response = supabase.table('books').select('book_title').execute()
        books = list(set([row['book_title'] for row in response.data]))
        return sorted(books)
    except Exception as e:
        print(f"Error fetching books: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch books: {str(e)}")

# --- API Endpoints ---

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "RAG Book Query API is running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint."""
    try:
        # Test Supabase connection
        supabase.table('books').select('count').limit(1).execute()
        return {
            "status": "healthy",
            "message": "All services are operational"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Service check failed: {str(e)}"
        }

@app.get("/books", response_model=BooksResponse)
async def get_books():
    """Get all available books."""
    try:
        books = get_all_books()
        return {"books": books}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QuestionResponse)
async def query_books(request: QuestionRequest):
    """
    Query the RAG system with a question.
    
    - If book_title is provided, search only in that book.
    - If book_title is None, search across all books.
    - Returns answer with sources and book suggestions if relevance is low.
    """
    try:
        question = request.question
        book_title = request.book_title
        
        # 1. Search in the selected book (or all books)
        results = search_in_book(question, book_title)
        
        if not results:
            return QuestionResponse(
                answer="I couldn't find any relevant information in the available books.",
                sources=[],
                suggestions=None,
                current_book=book_title,
                similarity_score=0.0
            )
        
        # 2. Check if results are relevant
        best_similarity = results[0].get('similarity', 0) if results else 0
        
        suggestions = None
        current_book_used = book_title
        
        # 3. If relevance is low and we have a specific book, suggest alternatives
        if best_similarity < SIMILARITY_THRESHOLD and book_title:
            suggested_book, suggested_similarity = suggest_book(question, book_title)
            
            if suggested_book and suggested_book != book_title and suggested_similarity > SIMILARITY_THRESHOLD:
                suggestions = [
                    BookSuggestion(
                        book_title=suggested_book,
                        similarity=suggested_similarity,
                        is_current_book=False
                    )
                ]
        
        # 4. Generate answer using top results
        answer = generate_answer(question, results[:3])
        
        # 5. Format sources
        sources = []
        for result in results[:3]:
            source = {
                "book_title": result.get('book_title', 'Unknown'),
                "chapter": result.get('chapter', 'N/A'),
                "topic": result.get('topic', 'N/A'),
                "content_preview": result.get('content', '')[:200] + "...",
                "similarity": result.get('similarity', None)
            }
            sources.append(source)
        
        return QuestionResponse(
            answer=answer,
            sources=sources,
            suggestions=suggestions,
            current_book=current_book_used,
            similarity_score=best_similarity
        )
        
    except Exception as e:
        print(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/search")
async def search_only(request: QuestionRequest):
    """
    Search for content without generating an answer.
    Returns raw search results.
    """
    try:
        results = search_in_book(request.question, request.book_title)
        
        if not results:
            return {"results": [], "message": "No results found"}
        
        return {
            "results": results[:MAX_RESULTS],
            "total": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggest")
async def suggest_book_endpoint(request: QuestionRequest):
    """
    Suggest the best book for a question.
    """
    try:
        suggested_book, similarity = suggest_book(
            request.question, 
            request.book_title or ""
        )
        
        if suggested_book:
            return {
                "suggested_book": suggested_book,
                "similarity": similarity,
                "current_book": request.book_title
            }
        else:
            return {
                "suggested_book": None,
                "similarity": 0,
                "current_book": request.book_title,
                "message": "No suitable book found"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Optional: Add logging middleware ---

@app.middleware("http")
async def log_requests(request, call_next):
    import time
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    print(f"Request: {request.method} {request.url.path} - {process_time:.2f}s")
    
    return response

# --- Run with Uvicorn (for local development) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
