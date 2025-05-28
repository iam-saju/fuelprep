from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, AsyncGenerator, Set
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import datetime
import urllib.parse
import hashlib
import asyncio
import requests # Kept for PPLX if used
import uuid
import random
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Constants ---
TRACKER_RESET_INTERVAL_SECONDS: int = 86400  # 24 hours
MODEL_NAME: str = "models/gemini-1.5-flash-latest"
MAX_RETRIES: int = 3
BASE_DELAY_SECONDS: int = 5
MAX_CONTENT_LENGTH_FOR_PROMPT_SNIPPET: int = 400 # For truncating content in prompts
MAX_TITLES_TO_SHOW_IN_EXCLUSION_PROMPT: int = 20

# Card Types and Refresh Types
CARD_TYPE_GEMINI_BATCH = "prelims_gemini_batch"
REFRESH_TYPE_CONTENT = "content"
REFRESH_TYPE_TOPIC = "topic"
REFRESH_TYPE_RELATED = "related"

# Example focus styles for card generation, customize as needed
CARD_FOCUS_STYLES = [
    "analytical depth", "factual accuracy", "conceptual clarity", 
    "interlinking topics", "current affairs relevance", 
    "historical perspective", "critical evaluation", "application-based understanding"
]

# Initialize FastAPI app
app = FastAPI(
    title="UPSC Prep Hub API",
    version="1.4.1",
    description="API for generating UPSC study materials using Gemini AI"
)

# Add GZip middleware for compressing responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS Configuration
origins = [
    "http://localhost", "http://localhost:8000", "http://localhost:5500",
    "http://127.0.0.1:5500", "http://localhost:3000",
    "https://40y4vocdk7fvboc3gfdfrldo2d5qkmnsaqj3t11tdne4i2iau7-h761100114.scf.usercontent.goog",
    "https://50yku0b3e8kza2l8bp1yf7f8c76guqsqua7c4rt8jikpd7afjp-h761100114.scf.usercontent.goog",
    "https://01b698t7nhyxxjzistw4ycu3ryv04y7ow63ci3yyaun1h7h277-h761100114.scf.usercontent.goog",
    "https://4p9ly7moi3l52hwhzaezn5tfqy6mlklial4f5lwhd3rd528pd2-h761100114.scf.usercontent.goog",
    "https://4u2zjqv4r7h8rp8kx860os8aqaylwr1basvvngeo6a9tizim66-h761100114.scf.usercontent.goog",
    "https://1m1e377x4s8m2hu09oihuqwgr0o4bsvgueay23ked2hdlwrgjl-h761100114.scf.usercontent.goog",
    "https://5fwm7ytn8szik1cg5fjv1lddco9789ue2fl07z4am9fe4z6olj-h761100114.scf.usercontent.goog",
    "https://5i0ujx7wab0yq5800jl98dx7epe1nlfp9mf3c6wrvyhkowmtt5-h761100114.scf.usercontent.goog",
]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Global error handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error for request {request.method} {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "error_type": "Validation Error"}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP exception for request {request.method} {request.url}: Status {exc.status_code}, Detail: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_type": "API Error"}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global unhandled exception for request {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal server error occurred.", "error_type": "Internal Server Error"}
    )

# --- Global State & Tracker ---
model: Optional[genai.GenerativeModel] = None
generated_content_tracker = {
    "titles": set(),
    "content_hashes": set(),
    "pyq_ids_for_topic": {}, # Key: topic_title_hash, Value: set of PYQ IDs
    "last_reset": datetime.datetime.now(datetime.timezone.utc)
}

def reset_tracker_if_needed():
    now = datetime.datetime.now(datetime.timezone.utc)
    if (now - generated_content_tracker["last_reset"]).total_seconds() > TRACKER_RESET_INTERVAL_SECONDS:
        generated_content_tracker["titles"].clear()
        generated_content_tracker["content_hashes"].clear()
        generated_content_tracker["pyq_ids_for_topic"].clear()
        generated_content_tracker["last_reset"] = now
        logger.info(f"Generated content tracker was reset at {now.isoformat()}.")

# --- Perplexity API Client Configuration (Kept for potential future use) ---
PPLX_API_KEY_ENV = os.getenv("PPLX_API_KEY")
# COMBINED_UPSC_PROMPTS and PPLX_PROMPTS are kept for reference as per original code.
# If not actively used, consider moving to a separate utils/config file or removing.
COMBINED_UPSC_PROMPTS = { "Polity": "...", "History": "...", "Economy": "...", "Geography": "...", "Science_Tech": "...", "Environment": "...", "Current Affairs": "...", "International": "...", "Internal_Security": "..."}
PPLX_PROMPTS = COMBINED_UPSC_PROMPTS

def call_pplx_api_sync(prompt: str, api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    # (Implementation remains as provided, for potential PPLX use)
    if not api_key: logger.error("PPLX_API_KEY not provided for call_pplx_api_sync!"); return None
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "sonar-pro", "messages": [{"role": "user", "content": prompt}], "max_tokens": 8000, "temperature": 0.3 }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=240)
        response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: logger.error(f"PPLX API Request error: {e}"); return None
    except json.JSONDecodeError as e: logger.error(f"PPLX API JSON decode error: {e}"); return None

def extract_pplx_content_sync(result: Optional[Dict[str, Any]]) -> Optional[str]:
    # (Implementation remains as provided)
    if result and 'choices' in result and result['choices'] and 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
        return result['choices'][0]['message']['content']
    logger.warning(f"PPLX API Unexpected response structure. Result: {result}"); return None
# --- End of Perplexity API Client Configuration ---

# --- Syllabus and Schema Definitions ---
SYLLABUS_CATEGORIES = [
    "Current Events (National & Intl.)", "History & Indian National Movement",
    "Indian & World Geography", "Indian Polity & Governance", "Economic & Social Development",
    "Environment, Biodiversity & Climate Change", "General Science"
]

PRELIMS_CARD_JSON_SCHEMA = {
    "type": "OBJECT", "properties": {
        "subject": {"type": "STRING", "enum": SYLLABUS_CATEGORIES},
        "title": {"type": "STRING"}, "summary": {"type": "STRING"},
        "content": {"type": "STRING"},
        "related_links": {"type": "ARRAY", "items": {"type": "STRING"}},
        "news_links": {
            "type": "ARRAY", "items": {
                "type": "OBJECT", "properties": {
                    "url": {"type": "STRING"}, "title": {"type": "STRING"},
                    "source": {"type": "STRING"}, "publishedAt": {"type": "STRING"}
                }, "required": ["url", "title"]
            }
        }
    }, "required": ["subject", "title", "summary", "content"]
}

PYQ_OPTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "key": {"type": "STRING"},
        "text": {"type": "STRING"}
    },
    "required": ["key", "text"]
}

PYQ_ITEM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "question": {"type": "STRING"},
        "options": {"type": "ARRAY", "items": PYQ_OPTION_SCHEMA},
        "answer_key": {"type": "STRING"},
        "relevance_explanation": {"type": "STRING"}
    },
    "required": ["question", "options", "answer_key", "relevance_explanation"]
}
RELATED_PYQS_RESPONSE_SCHEMA = {"type": "ARRAY", "items": PYQ_ITEM_SCHEMA}

# --- Pydantic Models ---
class QuestionPrompt(BaseModel): prompt: str = Field(..., min_length=3)
class RefreshCardRequest(BaseModel): card_id: str; current_title: str; current_subject: str; refresh_type: str = Field(default=REFRESH_TYPE_CONTENT)
class NewsArticle(BaseModel): url: str; title: str; source: Optional[str] = None; publishedAt: Optional[str] = None
class GeneratedPrelimsCard(BaseModel):
    id: str; type: str = CARD_TYPE_GEMINI_BATCH; subject: str; title: str; summary: str; content: str
    related_links: List[str] = Field(default_factory=list)
    news_links: List[NewsArticle] = Field(default_factory=list)
    coaching_notes_links: List[str] = Field(default_factory=list) # Kept for PPLX flow if used elsewhere
    generated_at: str; refreshed_at: Optional[str] = None; refresh_type_used: Optional[str] = None
class PYQOption(BaseModel): key: str; text: str
class GeneratedPYQItem(BaseModel): id: str = Field(default_factory=lambda: str(uuid.uuid4())); question: str; options: List[PYQOption]; answer_key: str; relevance_explanation: str
class RelatedPYQsRequest(BaseModel): topic_title: str; topic_content: str; subject: str; offset: int = 0; count: int = Field(default=3, ge=1, le=5); exclude_ids: Optional[List[str]] = None
class RelatedPYQsResponse(BaseModel): pyqs: List[GeneratedPYQItem]
class StudyCardsRequest(BaseModel):
    category: str = Field(..., description=f"UPSC Syllabus category. Must be one of: {', '.join(SYLLABUS_CATEGORIES + ['All'])}")
    count: int = Field(default=6, ge=1, le=10, description="Number of distinct cards to generate.")
    exclude_titles: Optional[List[str]] = Field(default_factory=list, description="List of titles to exclude to ensure new cards.")
class SuccessResponse(BaseModel): message: str; timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
class ErrorResponse(BaseModel): detail: str; error_type: Optional[str] = None; timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
class ContentStatsResponse(BaseModel): total_unique_titles_generated_session: int; total_unique_content_hashes_session: int; tracker_last_reset_utc: str; recent_titles_example: List[str]

# --- Startup Script ---
@app.on_event("startup")
async def startup_event():
    logger.info("--- UPSC Prep Hub API Startup ---")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logger.critical("CRITICAL: GOOGLE_API_KEY not found. API will not function correctly.")
    else:
        logger.info("GOOGLE_API_KEY found.")
        try:
            genai.configure(api_key=google_api_key)
            global model
            model = genai.GenerativeModel(MODEL_NAME)
            logger.info(f"Successfully initialized Gemini Model: '{MODEL_NAME}'")
        except Exception as e:
            logger.error(f"Error during Gemini model initialization: {e}", exc_info=True)
            model = None

    if not PPLX_API_KEY_ENV:
        logger.warning("PPLX_API_KEY not found. Perplexity-based features might be limited or disabled.")
    else:
        logger.info("PPLX_API_KEY found.")
    logger.info("---------------------------------")

# --- Helper Functions ---
def _ensure_model_initialized():
    if model is None:
        logger.error("Gemini model accessed before successful initialization.")
        raise HTTPException(status_code=503, detail="Gemini model is not available or failed to initialize.")
    if not isinstance(model, genai.GenerativeModel): # Should not happen if None check passes
        logger.error(f"Gemini model object is invalid type: {type(model)}")
        raise HTTPException(status_code=503, detail="Gemini model object is invalid.")

def is_valid_external_url(url: Optional[str]) -> bool:
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.scheme not in ["http", "https"]:
            return False
        hostname = parsed_url.hostname
        if not hostname:
            return False

        if hostname == "localhost" or hostname == "127.0.0.1": return False
        if hostname.startswith("192.168.") or hostname.startswith("10."): return False
        if hostname.startswith("172."):
            try: second_octet = int(hostname.split('.')[1]);
            except (IndexError, ValueError): return False # Malformed IP
            if 16 <= second_octet <= 31: return False
        
        # Heuristic: external URLs usually have a dot for TLD and are not just numeric IPs (unless public)
        # This is a basic check; more sophisticated validation might be needed for edge cases.
        if '.' not in hostname: return False # Likely an internal hostname if no dot
        if hostname.replace('.', '').isdigit() and not any(hostname.startswith(ip_block) for ip_block in ["192.168.", "10.", "172."]):
             # It's a numeric IP not caught by private IP checks - could be public, allow for now.
             # More specific public IP validation is complex.
             pass
        return True
    except (ValueError, AttributeError, IndexError): # Catch potential errors during parsing or splitting
        logger.debug(f"URL validation error for '{url}'", exc_info=False)
        return False

async def fetch_news_for_topic(topic_title: str, subject: str, news_api_key_to_use: Optional[str], num_articles: int = 3) -> List[Dict[str, Any]]:
    logger.info(f"NewsAPI fetching is currently disabled by design. Skipping NewsAPI call for '{topic_title}'.")
    return [] # NewsAPI is disabled as per original code's intent

async def _call_gemini_model_json(prompt_text: str, response_schema: Dict[str, Any], temperature: float = 0.7) -> str:
    _ensure_model_initialized()
    chat_history = [{"role": "user", "parts": [{"text": prompt_text}]}]
    generation_config = genai.types.GenerationConfig(response_mime_type="application/json", response_schema=response_schema, temperature=temperature)
    
    current_retry = 0
    while current_retry < MAX_RETRIES:
        try:
            logger.info(f"Calling Gemini for JSON (attempt {current_retry + 1}). Prompt snippet: {prompt_text[:150]}...")
            response = await model.generate_content_async(chat_history, generation_config=generation_config)
            if not response.text:
                logger.warning(f"Gemini returned empty text for JSON (attempt {current_retry + 1}).")
                current_retry += 1
                if current_retry >= MAX_RETRIES: raise HTTPException(status_code=500, detail="Gemini returned empty JSON response after all retries.")
                await asyncio.sleep(BASE_DELAY_SECONDS * (2 ** (current_retry -1)))
                continue
            logger.info(f"Successfully received JSON response from Gemini (attempt {current_retry + 1}).")
            return response.text
        except Exception as e:
            current_retry += 1; error_str = str(e).lower(); retry_after_seconds = BASE_DELAY_SECONDS * (2 ** (current_retry - 1))
            log_msg_prefix = f"Gemini JSON call error (attempt {current_retry}/{MAX_RETRIES})"
            if any(term in error_str for term in ["rate limit", "quota", "resourceexhausted", "429"]):
                logger.warning(f"{log_msg_prefix} - Rate limit. Retrying in {retry_after_seconds}s. Error: {e}")
                if current_retry >= MAX_RETRIES: raise HTTPException(status_code=429, detail="Gemini API rate limit exceeded after retries.")
            elif "api key not valid" in error_str or "permission_denied" in error_str:
                logger.error(f"{log_msg_prefix} - API key/permission error: {e}")
                raise HTTPException(status_code=401, detail="Gemini API key invalid or permission denied.")
            elif any(term in error_str for term in ["schema", "field", "json", "candidate"]): # "candidate" can appear in safety blocks
                logger.error(f"{log_msg_prefix} - Schema/JSON/Safety error: {e}. Prompt: {prompt_text[:200]}")
                raise HTTPException(status_code=500, detail=f"Gemini schema, JSON, or safety filter error: {str(e)}.")
            else:
                logger.error(f"{log_msg_prefix} - Generic error. Retrying in {retry_after_seconds}s. Error: {e}", exc_info=True)
                if current_retry >= MAX_RETRIES: raise HTTPException(status_code=500, detail=f"Failed Gemini JSON call after retries. Last error: {str(e)}")
            await asyncio.sleep(retry_after_seconds)
    raise HTTPException(status_code=500, detail="Failed Gemini JSON call after all retries (exhausted loop).") # Should be caught by inner logic

async def _stream_gemini_model_text(prompt_text: str) -> AsyncGenerator[str, None]:
    try: _ensure_model_initialized()
    except HTTPException as e: yield f"Error: {e.detail}\n"; return

    chat_history = [{"role": "user", "parts": [{"text": prompt_text}]}]
    try:
        logger.info(f"Streaming Gemini text. Prompt snippet: {prompt_text[:150]}...")
        response_stream = await model.generate_content_async(chat_history, stream=True)
        async for chunk in response_stream:
            if chunk.text: yield chunk.text
        logger.info("Finished streaming Gemini text.")
    except Exception as e:
        error_message = str(e); logger.error(f"Error streaming Gemini text: {error_message}", exc_info=True)
        if any(term in error_message.lower() for term in ["rate limit", "quota", "429"]): yield f"Error: Gemini API rate limit. Please try again later.\nDetails: {error_message}\n"
        else: yield f"Error generating content stream: {error_message}\n"

async def process_card_data_with_news(card_data_from_ai: dict, news_api_key_val: Optional[str]) -> dict:
    title = card_data_from_ai.get("title", "Untitled Topic")
    logger.debug(f"Processing card data for title: '{title}'")
    
    raw_news_links = card_data_from_ai.get("news_links", [])
    valid_news_articles: List[NewsArticle] = []
    if isinstance(raw_news_links, list):
        for item in raw_news_links:
            url, news_title_val, src, pub_at = None, "Untitled News", None, None
            if isinstance(item, dict):
                url, news_title_val, src, pub_at = item.get("url"), item.get("title", "Untitled News"), item.get("source"), item.get("publishedAt")
            if is_valid_external_url(url):
                valid_news_articles.append(NewsArticle(url=str(url).strip(), title=str(news_title_val).strip(), source=str(src).strip() if src else None, publishedAt=str(pub_at).strip() if pub_at else None))
            else: logger.warning(f"Filtered invalid/internal news URL from AI: '{url}' for card title '{title}'")
        card_data_from_ai["news_links"] = valid_news_articles
    else:
        logger.warning(f"AI provided 'news_links' is not a list for card title '{title}'. Type: {type(raw_news_links)}. Setting to empty list.")
        card_data_from_ai["news_links"] = []

    ai_suggested_links = card_data_from_ai.get("related_links", [])
    final_related_links = []
    seen_urls_for_related = {news.url for news in valid_news_articles if news.url}

    if isinstance(ai_suggested_links, list):
        for link_str in ai_suggested_links:
            if isinstance(link_str, str) and is_valid_external_url(link_str.strip()) and link_str.strip() not in seen_urls_for_related:
                final_related_links.append(link_str.strip())
                seen_urls_for_related.add(link_str.strip())
            elif not is_valid_external_url(str(link_str).strip()): logger.warning(f"Filtered invalid/internal related_link from AI: '{link_str}' for card title '{title}'")
    else:
        logger.warning(f"AI provided 'related_links' is not a list for card title '{title}'. Type: {type(ai_suggested_links)}. Setting to empty list.")
    
    if not final_related_links and not valid_news_articles and title != "Untitled Topic":
        google_search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(f'{title} UPSC relevant information')}"
        if is_valid_external_url(google_search_url): final_related_links.append(google_search_url)
    
    card_data_from_ai["related_links"] = final_related_links[:2] # Limit to 2 related links
    return card_data_from_ai

# --- API Endpoints ---
@app.get("/", response_model=SuccessResponse, tags=["General"])
async def root():
    return SuccessResponse(message="UPSC Prep Hub Backend is running!")

@app.post("/api/generate_question_streaming", tags=["UPSC Cards - Gemini Interactive"])
async def generate_question_streaming_endpoint(question_data: QuestionPrompt):
    if not question_data.prompt or not question_data.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    prompt_text = f"""
    Generate a UPSC Civil Services Prelims style Multiple Choice Question (MCQ) based on the following topic: "{question_data.prompt}".
    The question should be challenging and require analytical thinking, typical of UPSC standards.
    Provide:
    1. The question stem.
    2. Four distinct options, labeled (A), (B), (C), (D).
    3. The correct answer key (e.g., "Answer: C").
    4. A brief (1-2 sentences) explanation for the correct answer, highlighting why it's correct and others are not.
    Format the output clearly.
    """
    return StreamingResponse(_stream_gemini_model_text(prompt_text), media_type="text/plain")

@app.post("/api/generate_study_cards", response_model=List[GeneratedPrelimsCard], tags=["UPSC Cards - Gemini Batch"])
async def generate_study_cards_endpoint(request_data: StudyCardsRequest):
    reset_tracker_if_needed(); _ensure_model_initialized()
    logger.info(f"Received request to generate {request_data.count} study cards for category: '{request_data.category}'. Excluding {len(request_data.exclude_titles)} titles.")

    category = request_data.category
    if category != "All" and category not in SYLLABUS_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category: {category}. Must be 'All' or one of {SYLLABUS_CATEGORIES}")
    
    category_focus_prompt = f"Focus specifically on distinct sub-topics within the UPSC subject: '{category}'." if category != "All" else "Ensure a diverse range of subjects from the UPSC syllabus are covered across the generated cards."
    exclude_titles_str = ", ".join(request_data.exclude_titles) if request_data.exclude_titles else "None"
    random_focus_style = random.choice(CARD_FOCUS_STYLES)
    current_timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    batch_prompt = f"""
    Generate {request_data.count} TRULY DISTINCT and conceptually diverse UPSC Prelims study cards.
    {category_focus_prompt}
    Each card MUST cover a unique sub-topic not covered by other cards in this batch and should be distinct from these previously generated titles: {exclude_titles_str}.
    Prioritize significant but potentially less common sub-topics.
    For "Current Events (National & Intl.)", focus on very recent and significant developments (last 6-12 months).
    Incorporate a perspective of: "{random_focus_style}" for this batch.

    Each card MUST strictly follow this JSON structure. `related_links` should be valid external URLs. `news_links` is optional and should also contain valid external URLs.
    {{
    "subject": "string (must be one of: {', '.join(SYLLABUS_CATEGORIES)})",
    "title": "string (concise, unique, and specific sub-topic title, max 15 words, distinct from excluded titles and other titles in this batch)",
    "summary": "string (brief, 1-2 sentence summary, max 40 words, engaging and highlighting the core concept of the distinct sub-topic)",
    "content": "string (detailed explanation, 100-150 words, key facts, critical analysis, and specific details relevant for UPSC Prelims for this distinct sub-topic. Avoid generic statements; be factual and insightful.)",
    "related_links": ["string (Optional: 1-2 valid, external, high-quality reference URLs for this sub-topic. Do NOT include news article links here.)"],
    "news_links": [ {{ "url": "valid_external_url", "title": "News Headline for this sub-topic", "source": "News Source", "publishedAt": "YYYY-MM-DD" }} ]
    }}
    Request timestamp: {current_timestamp_iso}.
    Output: A single JSON array of these {request_data.count} card objects. No extra text.
    """
    try:
        generated_cards_json_string = await _call_gemini_model_json(batch_prompt, {"type": "ARRAY", "items": PRELIMS_CARD_JSON_SCHEMA}, temperature=0.8)
        generated_cards_list_from_ai = json.loads(generated_cards_json_string)
        if not isinstance(generated_cards_list_from_ai, list):
            logger.error(f"Gemini batch card generation did not return a list. Type: {type(generated_cards_list_from_ai)}. Response: {generated_cards_json_string[:200]}")
            raise HTTPException(status_code=500, detail="AI response for batch cards was not a list.")
        logger.info(f"Received {len(generated_cards_list_from_ai)} cards from AI. Processing and filtering...")

        final_cards: List[GeneratedPrelimsCard] = []
        processed_titles_this_batch = {title.lower().strip() for title in request_data.exclude_titles}

        for card_data_ai in generated_cards_list_from_ai:
            if not isinstance(card_data_ai, dict) or not all(key in card_data_ai for key in PRELIMS_CARD_JSON_SCHEMA["required"]):
                logger.warning(f"Skipping malformed card from AI: {card_data_ai}")
                continue
            title = str(card_data_ai.get("title", "")).strip(); content_val = str(card_data_ai.get("content", "")).strip()
            if not title or not content_val: logger.warning(f"Skipping card with empty title/content: '{title}'"); continue
            
            title_lower = title.lower(); content_hash = hashlib.md5(content_val.encode('utf-8')).hexdigest()
            if title_lower in generated_content_tracker["titles"] or title_lower in processed_titles_this_batch or content_hash in generated_content_tracker["content_hashes"]:
                logger.info(f"Skipping duplicate card (title or content hash already tracked): '{title}'")
                continue

            finalized_card_data = await process_card_data_with_news(card_data_ai, None)
            card_to_send = GeneratedPrelimsCard(
                id=f"prelims-gem-batch-{uuid.uuid4()}", type=CARD_TYPE_GEMINI_BATCH,
                subject=finalized_card_data.get("subject", "General Science"), title=title,
                summary=finalized_card_data.get("summary", "N/A"), content=content_val,
                related_links=finalized_card_data.get("related_links", []), news_links=finalized_card_data.get("news_links", []),
                coaching_notes_links=[], generated_at=current_timestamp_iso
            )
            final_cards.append(card_to_send)
            processed_titles_this_batch.add(title_lower); generated_content_tracker["titles"].add(title_lower); generated_content_tracker["content_hashes"].add(content_hash)
            if len(final_cards) >= request_data.count: break # Stop if desired count of unique cards is reached

        logger.info(f"Successfully generated and processed {len(final_cards)} unique study cards for category '{category}'.")
        return final_cards
    except HTTPException as e: logger.error(f"HTTPException in generate_study_cards: {e.detail}"); raise
    except json.JSONDecodeError as e_json:
        logger.error(f"JSON decode error from Gemini's card batch response: {e_json}. Response: {generated_cards_json_string[:200] if 'generated_cards_json_string' in locals() else 'N/A'}")
        raise HTTPException(status_code=500, detail="Failed to decode JSON from Gemini's card batch response.")
    except Exception as e:
        logger.error(f"Error in generate_study_cards_endpoint: {type(e).__name__} - {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate study cards. Error: {str(e)}")

@app.post("/api/refresh_card", response_model=Dict[str, Any], tags=["UPSC Cards - Gemini Interactive"])
async def refresh_card_endpoint(refresh_data: RefreshCardRequest):
    reset_tracker_if_needed(); _ensure_model_initialized()
    logger.info(f"Refreshing card ID '{refresh_data.card_id}', Title: '{refresh_data.current_title}', Type: '{refresh_data.refresh_type}'")

    current_timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    base_prompt_instruction, new_title_instruction = "", f"The card title is \"{refresh_data.current_title}\"."
    
    recent_titles_list = list(generated_content_tracker["titles"])
    titles_to_exclude_for_refresh = recent_titles_list[-MAX_TITLES_TO_SHOW_IN_EXCLUSION_PROMPT:]
    if refresh_data.refresh_type != REFRESH_TYPE_TOPIC and refresh_data.current_title.lower() not in {t.lower() for t in titles_to_exclude_for_refresh}:
        titles_to_exclude_for_refresh.append(refresh_data.current_title)
    
    exclusion_prompt_segment = f"Critically, avoid generating a card with a title or core concept similar to these existing titles: {', '.join(titles_to_exclude_for_refresh) if titles_to_exclude_for_refresh else 'None'}."
    random_focus = random.choice(CARD_FOCUS_STYLES)
    focus_hint = f"For this refreshed card, adopt a perspective focusing on: '{random_focus}'."

    if refresh_data.refresh_type == REFRESH_TYPE_CONTENT:
        base_prompt_instruction = f"Generate significantly UPDATED and conceptually FRESH content for the UPSC Prelims study card titled: \"{refresh_data.current_title}\" (Subject: {refresh_data.current_subject}). {focus_hint} Emphasize new information, recent developments, or a different analytical angle not present in typical explanations. Ensure the core topic remains the same but the content is novel. {exclusion_prompt_segment}"
    elif refresh_data.refresh_type == REFRESH_TYPE_TOPIC:
        base_prompt_instruction = f"Generate a COMPLETELY NEW and DISTINCT UPSC Prelims study card topic and its content within the subject: \"{refresh_data.current_subject}\". {focus_hint} The new topic title MUST be different from \"{refresh_data.current_title}\". {exclusion_prompt_segment}"
        new_title_instruction = "string (completely NEW, unique, and specific title for the new topic, max 15 words, distinct from old/excluded titles)"
    elif refresh_data.refresh_type == REFRESH_TYPE_RELATED:
        base_prompt_instruction = f"For the UPSC Prelims study card: \"{refresh_data.current_title}\" (Subject: {refresh_data.current_subject}), provide substantially ENHANCED content (around 100-150 words focusing on new dimensions or depth) and suggest NEW, highly relevant related links and news links. {focus_hint} {exclusion_prompt_segment}"
    else: raise HTTPException(status_code=400, detail=f"Invalid refresh_type: {refresh_data.refresh_type}")

    refresh_prompt = f"""{base_prompt_instruction}
    The output MUST be a single JSON object adhering to this schema:
    {{
        "subject": "string (must be '{refresh_data.current_subject}' or a valid UPSC subject if new topic)",
        "title": "{new_title_instruction if refresh_data.refresh_type == REFRESH_TYPE_TOPIC else f'string (can be same as old title "{refresh_data.current_title}" or slightly refined if content changes significantly)'}",
        "summary": "string (brief, 1-2 sentence summary, max 40 words)",
        "content": "string (detailed explanation, 100-150 words for content/topic refresh, or enhanced content for related refresh)",
        "related_links": ["string (Optional: 1-2 new, valid, external, high-quality reference URLs)"],
        "news_links": [ {{ "url": "valid_external_url", "title": "News Headline", "source": "News Source", "publishedAt": "YYYY-MM-DD" }} ]
    }}
    Request timestamp: {current_timestamp_iso}. No extra text outside the JSON object.
    """
    try:
        refreshed_card_json_string = await _call_gemini_model_json(refresh_prompt, PRELIMS_CARD_JSON_SCHEMA, temperature=0.85) # Slightly higher temp for refresh
        refreshed_card_ai_data = json.loads(refreshed_card_json_string)
        finalized_refreshed_data = await process_card_data_with_news(refreshed_card_ai_data, None)

        new_title = str(finalized_refreshed_data.get("title","")).strip()
        new_content = str(finalized_refreshed_data.get("content","")).strip()
        if not new_title or not new_content:
            logger.error(f"Refreshed card AI data has empty title/content for card ID {refresh_data.card_id}. AI Response: {finalized_refreshed_data}")
            raise HTTPException(status_code=500, detail="Refreshed card AI data has empty title or content.")

        if refresh_data.refresh_type == REFRESH_TYPE_TOPIC and new_title.lower() == refresh_data.current_title.lower():
            logger.warning(f"AI failed to generate a new distinct topic title for card ID {refresh_data.card_id}. Old: '{refresh_data.current_title}', New: '{new_title}'")
            raise HTTPException(status_code=500, detail="AI failed to generate a new distinct topic title for refresh. Please try again.")
        
        new_content_hash = hashlib.md5(new_content.encode('utf-8')).hexdigest()
        if new_title.lower() not in generated_content_tracker["titles"]: generated_content_tracker["titles"].add(new_title.lower())
        if new_content_hash not in generated_content_tracker["content_hashes"]: generated_content_tracker["content_hashes"].add(new_content_hash)

        final_card_object = GeneratedPrelimsCard(
            id=refresh_data.card_id, type=CARD_TYPE_GEMINI_BATCH, # Assuming refreshed cards are also of this type
            subject=finalized_refreshed_data.get("subject", refresh_data.current_subject), title=new_title,
            summary=finalized_refreshed_data.get("summary", "N/A"), content=new_content,
            related_links=finalized_refreshed_data.get("related_links", []), news_links=finalized_refreshed_data.get("news_links", []),
            coaching_notes_links=[], generated_at=finalized_refreshed_data.get("generated_at", current_timestamp_iso), # Preserve original gen time if possible
            refreshed_at=current_timestamp_iso, refresh_type_used=refresh_data.refresh_type
        )
        logger.info(f"Card ID '{refresh_data.card_id}' successfully refreshed. New title: '{new_title}'")
        return {"card": final_card_object.model_dump(), "message": f"Card '{new_title}' refreshed successfully."}
    except HTTPException as e: logger.error(f"HTTPException during card refresh for ID {refresh_data.card_id}: {e.detail}"); raise
    except json.JSONDecodeError as e_json:
        logger.error(f"JSON decode error in refresh_card_endpoint for ID {refresh_data.card_id}: {e_json}. Response: {refreshed_card_json_string[:200] if 'refreshed_card_json_string' in locals() else 'N/A'}")
        raise HTTPException(status_code=500, detail="Failed to decode AI response during card refresh.")
    except Exception as e:
        logger.error(f"Unexpected error in refresh_card_endpoint for ID {refresh_data.card_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to refresh card. Error: {str(e)}")

@app.get("/api/content_stats", response_model=ContentStatsResponse, tags=["Utility"])
async def get_content_stats():
    reset_tracker_if_needed()
    recent_titles_list = list(generated_content_tracker["titles"])
    logger.info("Content stats requested.")
    return ContentStatsResponse(
        total_unique_titles_generated_session=len(generated_content_tracker["titles"]),
        total_unique_content_hashes_session=len(generated_content_tracker["content_hashes"]),
        tracker_last_reset_utc=generated_content_tracker["last_reset"].isoformat(),
        recent_titles_example=recent_titles_list[-MAX_TITLES_TO_SHOW_IN_EXCLUSION_PROMPT:]
    )

@app.post("/api/generate_related_pyqs", response_model=RelatedPYQsResponse, tags=["UPSC Cards - Gemini Interactive"])
async def generate_related_pyqs_endpoint(request_data: RelatedPYQsRequest):
    reset_tracker_if_needed(); _ensure_model_initialized()
    logger.info(f"Generating {request_data.count} PYQs for topic: '{request_data.topic_title}', subject: '{request_data.subject}'. Offset: {request_data.offset}")

    topic_key_for_tracker = hashlib.md5(request_data.topic_title.lower().encode('utf-8')).hexdigest()
    if topic_key_for_tracker not in generated_content_tracker["pyq_ids_for_topic"]:
        generated_content_tracker["pyq_ids_for_topic"][topic_key_for_tracker] = set()
    
    current_topic_excluded_pyq_ids = generated_content_tracker["pyq_ids_for_topic"][topic_key_for_tracker].copy()
    if request_data.exclude_ids: current_topic_excluded_pyq_ids.update(request_data.exclude_ids)
    
    exclusion_ids_prompt_segment = f"IMPORTANT: Avoid generating questions identical or very similar in concept to those represented by these internal identifiers (if any): {', '.join(list(current_topic_excluded_pyq_ids)[-15:])}." if current_topic_excluded_pyq_ids else "Ensure all generated questions are unique for this topic."

    prompt = f"""
    Given the UPSC study card details:
    Topic Title: "{request_data.topic_title}"
    Subject: "{request_data.subject}"
    Content Snippet: "{request_data.topic_content[:MAX_CONTENT_LENGTH_FOR_PROMPT_SNIPPET]}..."

    Generate {request_data.count} unique, high-quality UPSC Prelims style Multiple Choice Questions (MCQs) directly relevant to this specific topic and content.
    {exclusion_ids_prompt_segment}
    Each MCQ must have:
    1. A clear question stem.
    2. Four distinct options (A, B, C, D).
    3. A single correct answer key (e.g., "A").
    4. A concise relevance explanation (1-2 sentences) connecting the question to the provided topic/content.

    Return a JSON array where each element is an object strictly following this schema:
    {{
        "question": "string (The MCQ question)",
        "options": [ {{ "key": "A", "text": "Option A text" }}, {{ "key": "B", "text": "Option B text" }}, ... ],
        "answer_key": "string (e.g., 'A', 'B', 'C', or 'D')",
        "relevance_explanation": "string (Brief explanation of relevance to the topic)"
    }}
    No extra text outside the JSON array.
    """
    try:
        generated_pyqs_json_string = await _call_gemini_model_json(prompt_text=prompt, response_schema=RELATED_PYQS_RESPONSE_SCHEMA, temperature=0.65)
        raw_pyqs_from_ai = json.loads(generated_pyqs_json_string)
        if not isinstance(raw_pyqs_from_ai, list):
            logger.error(f"Gemini PYQ generation did not return a list. Type: {type(raw_pyqs_from_ai)}. Response: {generated_pyqs_json_string[:200]}")
            raise HTTPException(status_code=500, detail="AI response for PYQs was not a list.")

        processed_pyqs: List[GeneratedPYQItem] = []
        for pyq_data in raw_pyqs_from_ai:
            if not isinstance(pyq_data, dict) or not all(key in pyq_data for key in PYQ_ITEM_SCHEMA["required"]):
                logger.warning(f"Skipping malformed PYQ from AI: {pyq_data}"); continue
            try:
                # Ensure options have 'key' and 'text'
                if not all(isinstance(opt, dict) and "key" in opt and "text" in opt for opt in pyq_data.get("options",[])):
                    logger.warning(f"Skipping PYQ with malformed options: {pyq_data.get('options')}"); continue
                
                pyq_item = GeneratedPYQItem(**pyq_data) # Pydantic validation
                # Check if this PYQ (based on question text hash) was already generated for this topic to avoid simple rephrasing by AI
                question_hash = hashlib.md5(pyq_item.question.lower().encode('utf-8')).hexdigest()
                if question_hash not in generated_content_tracker["pyq_ids_for_topic"][topic_key_for_tracker]:
                    generated_content_tracker["pyq_ids_for_topic"][topic_key_for_tracker].add(question_hash) # Store hash instead of UUID for conceptual uniqueness
                    processed_pyqs.append(pyq_item)
                else:
                    logger.info(f"Skipping conceptually duplicate PYQ (question hash exists): {pyq_item.question[:50]}...")

            except Exception as pydantic_e: logger.warning(f"Pydantic validation error for PYQ: {pyq_data}. Error: {pydantic_e}"); continue
        
        logger.info(f"Successfully generated and processed {len(processed_pyqs)} PYQs for topic '{request_data.topic_title}'.")
        return RelatedPYQsResponse(pyqs=processed_pyqs)
    except HTTPException as e: logger.error(f"HTTPException in generate_related_pyqs: {e.detail}"); raise
    except json.JSONDecodeError as e_json:
        logger.error(f"JSON decode error from Gemini's PYQ response: {e_json}. Response: {generated_pyqs_json_string[:200] if 'generated_pyqs_json_string' in locals() else 'N/A'}")
        raise HTTPException(status_code=500, detail="Failed to decode JSON from Gemini's PYQ response.")
    except Exception as e:
        logger.error(f"Error in generate_related_pyqs_endpoint: {type(e).__name__} - {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate PYQs. Error: {str(e)}")

# --- Main execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for UPSC Prep Hub API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
