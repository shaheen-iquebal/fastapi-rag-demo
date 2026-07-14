import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from supabase import create_client
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import logging
import secrets

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-nano-9b-v2:free")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
API_KEY = os.getenv("API_KEY")  # The secret API key

# --- API Key Security ---
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def validate_api_key(api_key: str = Security(api_key_header)):
    """Validate the API key from the request header."""
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key missing. Please provide X-API-Key header."
        )
    
    # Use secrets.compare_digest for constant-time comparison (prevents timing attacks)
    if not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key. Access denied."
        )
    
    return api_key

# --- Initialize Clients ---
logger.info("Connecting to Supabase...")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

logger.info("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

logger.info("Initializing OpenRouter...")
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- FastAPI App ---
app = FastAPI(title="Plexus AI Backend", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    workspace: Optional[str] = None
    conversation_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    success: bool
    message: str
    sources: List[Dict[str, Any]]
    has_content: bool

# --- Helper Functions ---

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text using all-MiniLM-L6-v2."""
    try:
        return embedding_model.encode(text).tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding")

def search_similar_books(query_embedding: List[float], workspace: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search for similar content using pgvector.
    If workspace is provided, filter by that subject.
    """
    try:
        # If workspace is "All" or None, don't filter
        filter_subject = workspace if workspace and workspace != "All" else None
        
        # Call the match_books RPC function
        response = supabase.rpc(
            'match_books',
            {
                'query_embedding': query_embedding,
                'match_threshold': 0.0,
                'match_count': MAX_RESULTS,
                'filter_subject': filter_subject  # Now filters by subject
            }
        ).execute()
        
        results = response.data
        
        # Filter by threshold
        filtered_results = [
            r for r in results 
            if r.get('similarity', 0) >= SIMILARITY_THRESHOLD
        ]
        
        return filtered_results[:MAX_RESULTS]
        
    except Exception as e:
        logger.error(f"Error searching books: {e}")
        return []

def build_retrieval_prompt(question: str, context_results: List[Dict[str, Any]]) -> str:
    """Build prompt that strictly uses ONLY the retrieved content."""
    
    # Build context from results
    context_parts = []
    for i, result in enumerate(context_results, 1):
        context_parts.append(f"""
[Source {i}]
Book/Workspace: {result.get('book_title', 'Unknown')}
Chapter: {result.get('chapter', 'Unknown')}
Topic: {result.get('topic', 'Unknown')}
Content: {result.get('content', '')}
""")
    
    context = "\n".join(context_parts)
    
    prompt = f"""You are an assistant that ONLY summarizes the provided content.
CRITICAL RULES:
1. ONLY use information from the provided content below
2. DO NOT add any information from outside the provided text
3. DO NOT use your training data or general knowledge
4. If the content doesn't fully answer the question, clearly state that
5. Cite your sources by mentioning which source provided the information

PROVIDED CONTENT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer using ONLY the content above
- If the content has the answer, summarize it clearly
- If the content doesn't have the answer, say "I couldn't find relevant information in your workspace"
- Always mention which source(s) you're using

ANSWER:"""

    return prompt

def call_openrouter(prompt: str) -> str:
    """Call OpenRouter API with the prompt."""
    try:
        response = openrouter_client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": "You are a retrieval-only assistant. You ONLY answer based on the provided content. You NEVER use external knowledge."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error calling OpenRouter: {e}")
        raise HTTPException(status_code=500, detail="Failed to get response from AI")

def format_sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format search results as sources for the frontend."""
    sources = []
    for r in results:
        sources.append({
            "book": r.get('book_title', 'Unknown'),
            "chapter": r.get('chapter', 'Unknown'),
            "topic": r.get('topic', 'Unknown'),
            "similarity": round(r.get('similarity', 0), 3),
            "content_preview": r.get('content', '')[:200] + "..." if r.get('content', '') else ""
        })
    return sources

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"status": "healthy", "service": "Plexus AI Backend"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    api_key: str = Depends(validate_api_key)  # Protected endpoint
):
    """
    Main chat endpoint with strict retrieval-only approach.
    Requires API key in X-API-Key header.
    """
    try:
        logger.info(f"Received chat request: workspace={request.workspace}, message={request.message[:50]}...")
        
        # Step 1: Generate embedding for the question
        query_embedding = generate_embedding(request.message)
        
        # Step 2: Search for similar content
        results = search_similar_books(query_embedding, request.workspace)
        logger.info(f"Found {len(results)} relevant results")
        
        # Step 3: Check if we found any content
        if not results:
            return ChatResponse(
                success=True,
                message="I couldn't find any relevant information in your workspace. Please try rephrasing your question or selecting a different workspace.",
                sources=[],
                has_content=False
            )
        
        # Step 4: Build prompt with ONLY the retrieved content
        prompt = build_retrieval_prompt(request.message, results)
        
        # Step 5: Call OpenRouter to format the answer
        ai_response = call_openrouter(prompt)
        
        # Step 6: Format sources
        sources = format_sources(results)
        
        # Return response
        return ChatResponse(
            success=True,
            message=ai_response,
            sources=sources,
            has_content=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/workspaces")
async def get_workspaces(
    api_key: str = Depends(validate_api_key)  # Protected endpoint
):
    """Get all available workspaces from the database."""
    try:
        response = supabase.table('books').select('subject').execute()
        
        # Get unique subjects
        subjects = list(set([row['subject'] for row in response.data]))
        
        # Add "All" option
        workspaces = [
            {"id": "all", "title": "All Workspaces", "subject": "All", "description": "Search across all content"}
        ]
        
        for subject in sorted(subjects):
            workspaces.append({
                "id": subject.lower().replace(' ', '_'),
                "title": subject,
                "subject": subject,
                "description": f"Content from {subject}"
            })
        
        return {"workspaces": workspaces}
        
    except Exception as e:
        logger.error(f"Error fetching workspaces: {e}")
        return {"workspaces": []}

# --- Run the app ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)