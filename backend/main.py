from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect # WebSocket will be removed
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, AsyncGenerator
import json
import datetime
import urllib.parse
import hashlib
import asyncio
# import httpx # Not actively used, consider removing if not needed elsewhere
import requests
from fastapi.concurrency import run_in_threadpool
import uuid
import random
import re

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="UPSC Prep Hub API", version="1.4.0") # Version updated for new card fetching logic

# Add GZip middleware for compressing responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# --- Perplexity API Client Configuration (Kept for potential future use/internal tooling, but not primary for card batching) ---
PPLX_API_KEY_ENV = os.getenv("PPLX_API_KEY")

# COMBINED_UPSC_PROMPTS are no longer the primary driver for the main card fetching logic.
# They can be kept for reference or other specific research tasks if needed.
COMBINED_UPSC_PROMPTS = {
    "Polity": """Research TWO DISTINCT current Polity topics for UPSC 2025. For EACH topic researched, provide the following structure: ... """,
    # ... (other prompts remain as they were, but won't be directly used by the new /generate_study_cards endpoint)
    "History": """Research TWO DISTINCT current Modern Indian History topics for UPSC 2025...""",
    "Economy": """Research TWO DISTINCT major Economic issues/policies for UPSC 2025...""",
    "Geography": """Research TWO DISTINCT key Geography topics for UPSC 2025...""",
    "Science_Tech": """Research TWO DISTINCT S&T developments for UPSC 2025...""",
    "Environment": """Research TWO DISTINCT Environmental issues/policies for UPSC 2025...""",
    "Current Affairs": """Research TWO DISTINCT Current Affairs topics for UPSC 2025...""",
    "International": """Research ONE IR topic for UPSC 2025...""",
    "Internal_Security": """Research ONE security challenge for UPSC 2025..."""
}
PROMPT_GUIDELINES = """...""" # Kept for reference
PPLX_PROMPTS = COMBINED_UPSC_PROMPTS # Kept for reference

SYLLABUS_CATEGORIES = [
    "Current Events (National & Intl.)", "History & Indian National Movement",
    "Indian & World Geography", "Indian Polity & Governance", "Economic & Social Development",
    "Environment, Biodiversity & Climate Change", "General Science"
]

CATEGORY_TO_PPLX_SUBJECT_MAP = { # This mapping might still be useful for keyword generation if PPLX is used by Gemini
    "Current Events (National & Intl.)": "Current Affairs",
    "History & Indian National Movement": "History",
    "Indian & World Geography": "Geography",
    "Indian Polity & Governance": "Polity",
    "Economic & Social Development": "Economy",
    "Environment, Biodiversity & Climate Change": "Environment",
    "General Science": "Science_Tech"
}

# Perplexity API call functions (call_pplx_api_sync, extract_pplx_content_sync)
# are kept if they might be used by Gemini as a tool, or for other specific endpoints.
# For the new /generate_study_cards, direct PPLX calls for the 2-topic structure are removed.

def call_pplx_api_sync(prompt: str, api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    if not api_key:
        print("Error: PPLX_API_KEY not provided for call_pplx_api_sync!")
        return None
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "sonar-pro", "messages": [{"role": "user", "content": prompt}], "max_tokens": 8000, "temperature": 0.3 }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=240)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"PPLX API Request error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"PPLX API JSON decode error: {e}")
        return None

def extract_pplx_content_sync(result: Optional[Dict[str, Any]]) -> Optional[str]:
    if result and 'choices' in result and result['choices'] and 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
        return result['choices'][0]['message']['content']
    print(f"PPLX API Unexpected response structure. Result: {result}")
    return None

# Gemini Parsing Schemas (will be used by the new Gemini-driven card generation)
GEMINI_PARSING_SCHEMA_ITEM = { # This is for a single research item if PPLX is used as a tool by Gemini
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"}, "analysis": {"type": "STRING"},
        "sources": {
            "type": "OBJECT",
            "properties": {
                "official": {"type": "ARRAY", "items": {"type": "STRING"}},
                "media": {"type": "ARRAY", "items": {"type": "STRING"}},
                "additional": {"type": "ARRAY", "items": {"type": "STRING"}}
            }, "required": ["official", "media", "additional"]
        },
        "key_points": {"type": "ARRAY", "items": {"type": "STRING"}},
        "notes_links": {"type": "ARRAY", "items": {"type": "STRING"}}
    }, "required": ["title", "analysis", "sources", "key_points"]
}
GEMINI_PARSING_SCHEMA_FOR_PPLX_LIST = {"type": "ARRAY", "items": GEMINI_PARSING_SCHEMA_ITEM}
# --- End of Perplexity API Client Configuration ---


# --- Startup Script ---
@app.on_event("startup")
async def startup_event():
    print("--- UPSC Prep Hub API Startup ---")
    google_api_key_startup = os.getenv("GOOGLE_API_KEY")
    pplx_api_key_startup = os.getenv("PPLX_API_KEY")

    if not google_api_key_startup: print("CRITICAL: GOOGLE_API_KEY not found.")
    else:
        print("GOOGLE_API_KEY found.")
        try:
            genai.configure(api_key=google_api_key_startup)
            # Model listing logic (unchanged)
        except Exception as e: print(f"Error during Gemini model listing: {e}.")

    if not pplx_api_key_startup: print("WARNING: PPLX_API_KEY not found. Some research features might be limited if Gemini uses it as a tool.")
    else: print("PPLX_API_KEY found.")
    print("---------------------------------")

# CORS Configuration (unchanged)
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

# Gemini API Configuration (unchanged)
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key: genai.configure(api_key=google_api_key)
else: print("CRITICAL: GOOGLE_API_KEY environment variable not set for runtime.")

model: Optional[genai.GenerativeModel] = None
model_name_to_use = "models/gemini-1.5-flash-latest"
if google_api_key:
    try:
        model = genai.GenerativeModel(model_name_to_use)
        print(f"Successfully initialized Gemini Model: '{model_name_to_use}' for runtime.")
    except Exception as e: print(f"CRITICAL: Failed to initialize Gemini Model '{model_name_to_use}' for runtime. Error: {e}")
else: print("Skipping runtime Gemini Model initialization: GOOGLE_API_KEY not set.")

# Content Tracker (unchanged)
generated_content_tracker = { "titles": set(), "content_hashes": set(), "pyq_ids_for_topic": {}, "last_reset": datetime.datetime.now(datetime.timezone.utc)}
TRACKER_RESET_INTERVAL_SECONDS = 86400
def reset_tracker_if_needed(): # (unchanged)
    now = datetime.datetime.now(datetime.timezone.utc)
    if (now - generated_content_tracker["last_reset"]).total_seconds() > TRACKER_RESET_INTERVAL_SECONDS:
        generated_content_tracker["titles"].clear()
        generated_content_tracker["content_hashes"].clear()
        generated_content_tracker["pyq_ids_for_topic"].clear()
        generated_content_tracker["last_reset"] = now
        print(f"Generated content tracker reset at {now.isoformat()}.")

# --- Pydantic Models ---
class QuestionPrompt(BaseModel): prompt: str = Field(..., min_length=3)
class RefreshCardRequest(BaseModel): card_id: str; current_title: str; current_subject: str; refresh_type: str = Field(default="content")
class NewsArticle(BaseModel): url: str; title: str; source: Optional[str] = None; publishedAt: Optional[str] = None
class GeneratedPrelimsCard(BaseModel):
    id: str; type: str = "prelims"; subject: str; title: str; summary: str; content: str
    related_links: List[str] = Field(default_factory=list)
    news_links: List[NewsArticle] = Field(default_factory=list)
    coaching_notes_links: List[str] = Field(default_factory=list) # Kept for PPLX flow if used elsewhere
    generated_at: str; refreshed_at: Optional[str] = None; refresh_type_used: Optional[str] = None
class PYQOption(BaseModel): key: str; text: str
class GeneratedPYQItem(BaseModel): id: str = Field(default_factory=lambda: str(uuid.uuid4())); question: str; options: List[PYQOption]; answer_key: str; relevance_explanation: str
class RelatedPYQsRequest(BaseModel): topic_title: str; topic_content: str; subject: str; offset: int = 0; count: int = Field(default=3, ge=1, le=5); exclude_ids: Optional[List[str]] = None
class RelatedPYQsResponse(BaseModel): pyqs: List[GeneratedPYQItem]

# New request model for batch card generation
class StudyCardsRequest(BaseModel):
    category: str = Field(..., description=f"UPSC Syllabus category. Must be one of: {', '.join(SYLLABUS_CATEGORIES + ['All'])}")
    count: int = Field(default=6, ge=1, le=10, description="Number of distinct cards to generate.")
    exclude_titles: Optional[List[str]] = Field(default_factory=list, description="List of titles to exclude to ensure new cards.")

# UPSCSourcesModel and UPSCResearchOutputModel are kept for the PPLX parsing logic,
# which might be used if Gemini calls PPLX as a tool, or for a separate research endpoint.
class UPSCSourcesModel(BaseModel): official: List[str] = Field(default_factory=list); media: List[str] = Field(default_factory=list); additional: List[str] = Field(default_factory=list)
class UPSCResearchOutputModel(BaseModel): title: str; analysis: str; sources: UPSCSourcesModel; key_points: List[str] = Field(default_factory=list); notes_links: Optional[List[str]] = Field(default_factory=list)


# --- Helper Functions (is_valid_external_url, fetch_news_for_topic, _call_gemini_model_json, _stream_gemini_model_text, process_card_data_with_news) ---
# These functions remain largely unchanged as they are general utilities.
def is_valid_external_url(url: Optional[str]) -> bool: # (unchanged)
    if not isinstance(url, str): return False
    if not (url.startswith("http://") or url.startswith("https://")): return False
    try:
        parsed_url = urllib.parse.urlparse(url)
        hostname = parsed_url.hostname
        if hostname:
            if hostname == "localhost" or hostname == "127.0.0.1" or \
               hostname.startswith("192.168.") or hostname.startswith("10.") or \
               (hostname.startswith("172.") and 16 <= int(hostname.split('.')[1]) <= 31):
                return False
            if not re.search(r"\.[a-zA-Z]{2,}$", hostname) and not re.match(r"^[a-zA-Z0-9-]+$", hostname):
                 pass
        else: return False
    except: return False
    return True

async def fetch_news_for_topic(topic_title: str, subject: str, news_api_key_to_use: Optional[str], num_articles: int = 3) -> List[Dict[str, Any]]: # (unchanged - NewsAPI disabled)
    print(f"INFO: NewsAPI fetching is disabled. Skipping NewsAPI call for '{topic_title}'.")
    return []

async def _call_gemini_model_json(prompt_text: str, response_schema: Dict[str, Any], temperature: float = 0.7) -> str: # (unchanged)
    if model is None: raise HTTPException(status_code=503, detail="Gemini model not initialized.")
    if not isinstance(model, genai.GenerativeModel): raise HTTPException(status_code=503, detail="Gemini model object invalid.")
    # ... (rest of the retry and call logic is the same)
    chat_history = [{"role": "user", "parts": [{"text": prompt_text}]}]
    generation_config = genai.types.GenerationConfig(response_mime_type="application/json", response_schema=response_schema, temperature=temperature)
    max_retries, current_retry, base_delay_seconds = 3, 0, 5
    while current_retry < max_retries:
        try:
            response = await model.generate_content_async(chat_history, generation_config=generation_config)
            if not response.text:
                 print(f"Warning: Gemini empty text for JSON (attempt {current_retry + 1}): {prompt_text[:100]}...")
                 if current_retry == max_retries -1: raise HTTPException(status_code=500, detail="Gemini empty JSON response after retries.")
            else: return response.text
        except Exception as e:
            current_retry += 1; error_str = str(e).lower(); retry_after_seconds = base_delay_seconds * (2 ** (current_retry -1))
            if any(term in error_str for term in ["rate limit", "quota", "resourceexhausted", "429"]):
                print(f"Rate limit for JSON. Retry {current_retry}/{max_retries}. Error: {e}")
                # ... (rate limit handling)
            elif "api key not valid" in error_str or "permission_denied" in error_str:
                 raise HTTPException(status_code=401, detail="Gemini API key invalid/permission denied.")
            elif any(term in error_str for term in ["schema", "field", "json"]):
                raise HTTPException(status_code=500, detail=f"Gemini schema/JSON error: {str(e)}.")
            else:
                print(f"Error Gemini JSON (attempt {current_retry}/{max_retries}): {e}")
                if current_retry >= max_retries: raise HTTPException(status_code=500, detail=f"Failed Gemini JSON after retries. Error: {str(e)}")
            await asyncio.sleep(retry_after_seconds)
    raise HTTPException(status_code=500, detail="Failed Gemini JSON after all retries.")


async def _stream_gemini_model_text(prompt_text: str) -> AsyncGenerator[str, None]: # (unchanged)
    if model is None: yield "Error: Gemini model not initialized.\n"; return
    if not isinstance(model, genai.GenerativeModel): yield "Error: Gemini model object invalid.\n"; return
    chat_history = [{"role": "user", "parts": [{"text": prompt_text}]}]
    try:
        response_stream = await model.generate_content_async(chat_history, stream=True)
        async for chunk in response_stream:
            if chunk.text: yield chunk.text
    except Exception as e:
        error_message = str(e); print(f"Error streaming Gemini: {error_message}")
        if any(term in error_message.lower() for term in ["rate limit", "quota", "429"]): yield f"Error: Gemini API rate limit. Try again.\nDetails: {error_message}\n"
        else: yield f"Error generating content: {error_message}\n"

async def process_card_data_with_news(card_data_from_ai: dict, news_api_key_val: Optional[str]) -> dict: # (unchanged)
    # ... (logic for validating AI-provided news/related links and formatting content)
    title = card_data_from_ai.get("title", "Untitled")
    current_content = card_data_from_ai.get("content", "")
    raw_news_links = card_data_from_ai.get("news_links", [])
    valid_news_articles: List[NewsArticle] = []
    if isinstance(raw_news_links, list):
        for item in raw_news_links:
            url, news_title, src, pub_at = None, "Untitled", None, None
            if isinstance(item, dict): url, news_title, src, pub_at = item.get("url"), item.get("title", "Untitled"), item.get("source"), item.get("publishedAt")
            if is_valid_external_url(url): valid_news_articles.append(NewsArticle(url=url, title=news_title, source=src, publishedAt=pub_at))
            else: print(f"Filtered invalid news URL: {url}")
    card_data_from_ai["news_links"] = valid_news_articles
    # ... (rest of link processing and content formatting)
    ai_suggested_links = card_data_from_ai.get("related_links", [])
    final_related_links = []
    seen_urls_for_related = {news.url for news in valid_news_articles if news.url}
    if isinstance(ai_suggested_links, list):
        for link_str in ai_suggested_links:
            if is_valid_external_url(link_str) and link_str not in seen_urls_for_related:
                final_related_links.append(link_str); seen_urls_for_related.add(link_str)
            elif not is_valid_external_url(link_str): print(f"Filtered invalid related_link: {link_str}")
    if not final_related_links and not valid_news_articles and title != "Untitled":
        google_search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(f'{title} UPSC relevant information')}"
        if is_valid_external_url(google_search_url): final_related_links.append(google_search_url)
    card_data_from_ai["related_links"] = final_related_links[:2]
    return card_data_from_ai


# --- Global Constants & Schemas ---
CARD_FOCUS_STYLES = [ ... ] # (unchanged)
PRELIMS_CARD_JSON_SCHEMA = { # This schema is now primary for Gemini batch generation
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
        # "coaching_notes_links" could be added if Gemini is prompted to find these too
    }, "required": ["subject", "title", "summary", "content"]
}
PYQ_ITEM_SCHEMA = { ... } # (unchanged)
RELATED_PYQS_RESPONSE_SCHEMA = {"type": "ARRAY", "items": PYQ_ITEM_SCHEMA } # (unchanged)

# --- API Endpoints ---
@app.get("/")
async def root(): return {"message": "UPSC Prep Hub Backend is running!"} # (unchanged)

@app.post("/api/generate_question_streaming")
async def generate_question_streaming_endpoint(question_data: QuestionPrompt): # (unchanged)
    if not question_data.prompt or not question_data.prompt.strip(): raise HTTPException(status_code=400, detail="Prompt empty.")
    prompt_text = f"Generate a UPSC-style question ... on: \"{question_data.prompt}\"..." # (prompt details unchanged)
    return StreamingResponse(_stream_gemini_model_text(prompt_text), media_type="text/plain")

# REMOVED: @app.websocket("/ws/generate_prelims_cards") and its handler function.

@app.post("/api/generate_study_cards", response_model=List[GeneratedPrelimsCard], tags=["UPSC Cards - Gemini Batch"])
async def generate_study_cards_endpoint(request_data: StudyCardsRequest):
    """
    Generates a batch of distinct study cards for a given category using Gemini.
    """
    reset_tracker_if_needed()
    if model is None:
        raise HTTPException(status_code=503, detail="Gemini model not initialized.")

    category = request_data.category
    num_cards_to_generate = request_data.count
    exclude_titles_str = ", ".join(request_data.exclude_titles) if request_data.exclude_titles else "None"
    
    # Dynamically create part of the prompt based on the category
    if category == "All":
        category_focus_prompt = "Ensure a diverse range of subjects from the UPSC syllabus are covered across the generated cards."
    else:
        if category not in SYLLABUS_CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}. Must be 'All' or one of {SYLLABUS_CATEGORIES}")
        category_focus_prompt = f"Focus specifically on distinct sub-topics within the UPSC subject: '{category}'."

    random_focus_style = random.choice(CARD_FOCUS_STYLES)
    current_timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Construct the prompt for Gemini
    batch_prompt = f"""
    Generate {num_cards_to_generate} TRULY DISTINCT and conceptually diverse UPSC Prelims study cards.
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
    Output: A single JSON array of these {num_cards_to_generate} card objects. No extra text.
    """

    try:
        generated_cards_json_string = await _call_gemini_model_json(
            batch_prompt,
            {"type": "ARRAY", "items": PRELIMS_CARD_JSON_SCHEMA},
            temperature=0.8 # Slightly higher temperature for diversity
        )
        generated_cards_list_from_ai = json.loads(generated_cards_json_string)

        if not isinstance(generated_cards_list_from_ai, list):
            print(f"Gemini batch card generation did not return a list. Got: {type(generated_cards_list_from_ai)}")
            raise HTTPException(status_code=500, detail="AI response for batch cards was not a list.")

        final_cards: List[GeneratedPrelimsCard] = []
        processed_titles_this_batch = set(request_data.exclude_titles) # Start with titles to exclude

        for card_data_ai in generated_cards_list_from_ai:
            if not isinstance(card_data_ai, dict) or not all(key in card_data_ai for key in PRELIMS_CARD_JSON_SCHEMA["required"]):
                print(f"Skipping malformed card from AI: {card_data_ai}")
                continue
            
            title = card_data_ai.get("title", "").strip()
            content_val = card_data_ai.get("content", "").strip()
            if not title or not content_val:
                print(f"Skipping card with empty title/content: {title}")
                continue

            title_lower = title.lower()
            content_hash = hashlib.md5(content_val.encode('utf-8')).hexdigest()

            # Check against global tracker and current batch's processed titles
            if title_lower in generated_content_tracker["titles"] or \
               title_lower in processed_titles_this_batch or \
               content_hash in generated_content_tracker["content_hashes"]:
                print(f"Skipping duplicate card (title or content): {title}")
                continue

            finalized_card_data = await process_card_data_with_news(card_data_ai, None)

            card_to_send = GeneratedPrelimsCard(
                id=f"prelims-gem-batch-{uuid.uuid4()}",
                type="prelims_gemini_batch",
                subject=finalized_card_data.get("subject", "General Science"),
                title=title,
                summary=finalized_card_data.get("summary", "N/A"),
                content=finalized_card_data.get("content", "N/A"),
                related_links=finalized_card_data.get("related_links", []),
                news_links=finalized_card_data.get("news_links", []),
                coaching_notes_links=[], # No coaching notes from this flow by default
                generated_at=current_timestamp_iso
            )
            final_cards.append(card_to_send)
            
            # Update trackers
            processed_titles_this_batch.add(title_lower)
            generated_content_tracker["titles"].add(title_lower)
            generated_content_tracker["content_hashes"].add(content_hash)
        
        print(f"Generated {len(final_cards)} unique cards for category '{category}'.")
        return final_cards

    except HTTPException as e: # Re-raise HTTP exceptions from Gemini call
        raise e
    except json.JSONDecodeError as e_json:
        print(f"JSON decode error from Gemini's card batch response: {e_json}. Response: {generated_cards_json_string[:200] if 'generated_cards_json_string' in locals() else 'N/A'}")
        raise HTTPException(status_code=500, detail="Failed to decode JSON from Gemini's card batch response.")
    except Exception as e:
        print(f"Error in generate_study_cards_endpoint: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate study cards. Error: {str(e)}")


@app.post("/api/refresh_card", response_model=Dict[str, Any])
async def refresh_card_endpoint(refresh_data: RefreshCardRequest): # (Largely unchanged, but ensure it uses Gemini well)
    reset_tracker_if_needed()
    # ... (rest of the refresh logic is the same)
    current_timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    base_prompt_instruction, new_title_instruction, title_for_prompt = "", f"\"{refresh_data.current_title}\"", refresh_data.current_title
    recent_titles_list = list(generated_content_tracker["titles"])
    exclusion_prompt_segment = f"Critically, avoid generating a card with a title or core concept similar to these: {', '.join(recent_titles_list[-20:] + ([refresh_data.current_title] if refresh_data.refresh_type != 'topic' else [])) if recent_titles_list else refresh_data.current_title}."
    random_focus = random.choice(CARD_FOCUS_STYLES); focus_hint = f"For this refreshed card, adopt a perspective focusing on {random_focus}"
    if refresh_data.refresh_type == "content": base_prompt_instruction = f"Generate significantly UPDATED and conceptually FRESH content for the UPSC Prelims study card titled: \"{refresh_data.current_title}\" (Subject: {refresh_data.current_subject}). {focus_hint} Emphasize new information, recent developments, or a different analytical angle not present in typical explanations. {exclusion_prompt_segment}"
    elif refresh_data.refresh_type == "topic": base_prompt_instruction = f"Generate a COMPLETELY NEW and DISTINCT UPSC Prelims study card topic and its content within the subject: \"{refresh_data.current_subject}\". {focus_hint} {exclusion_prompt_segment}"; new_title_instruction = "string (completely NEW, unique, and specific title for the new topic, max 15 words, distinct from old/excluded titles)"; title_for_prompt = "a new distinct topic, different from previous ones"
    elif refresh_data.refresh_type == "related": base_prompt_instruction = f"For the UPSC Prelims study card: \"{refresh_data.current_title}\" (Subject: {refresh_data.current_subject}), provide substantially ENHANCED content ... and suggest NEW, highly relevant related links ... {focus_hint} {exclusion_prompt_segment}"
    else: raise HTTPException(status_code=400, detail="Invalid refresh_type.")
    refresh_prompt = f"""{base_prompt_instruction} ... Return a single JSON object ... {PRELIMS_CARD_JSON_SCHEMA} ... Request timestamp: {current_timestamp_iso}.""" # Prompt structure same
    try:
        refreshed_card_json_string = await _call_gemini_model_json(refresh_prompt, PRELIMS_CARD_JSON_SCHEMA, temperature=0.9)
        refreshed_card_ai_data = json.loads(refreshed_card_json_string)
        # ... (validation and processing of refreshed_card_ai_data as before)
        finalized_refreshed_data = await process_card_data_with_news(refreshed_card_ai_data, None)
        # ... (update trackers and return final_card_object)
        new_title = finalized_refreshed_data.get("title","").strip()
        new_content = finalized_refreshed_data.get("content","").strip()
        if not new_title or not new_content: raise HTTPException(status_code=500, detail="Refreshed card AI data has empty title/content.")
        if refresh_data.refresh_type == "topic" and new_title.lower() == refresh_data.current_title.lower():
            raise HTTPException(status_code=500, detail="AI failed to generate a new distinct topic title for refresh.")
        new_content_hash = hashlib.md5(new_content.encode('utf-8')).hexdigest()
        generated_content_tracker["titles"].add(new_title.lower())
        generated_content_tracker["content_hashes"].add(new_content_hash)

        final_card_object = GeneratedPrelimsCard(
            id=refresh_data.card_id, subject=finalized_refreshed_data.get("subject", refresh_data.current_subject),
            title=new_title, summary=finalized_refreshed_data.get("summary", "N/A"),
            content=finalized_refreshed_data.get("content", "N/A"),
            related_links=finalized_refreshed_data.get("related_links", []),
            news_links=finalized_refreshed_data.get("news_links", []),
            coaching_notes_links=[],
            generated_at=finalized_refreshed_data.get("generated_at", current_timestamp_iso),
            refreshed_at=current_timestamp_iso, refresh_type_used=refresh_data.refresh_type
        )
        return {"card": final_card_object.model_dump(), "message": f"Card '{new_title}' refreshed."}

    except Exception as e:
        print(f"Error in refresh_card_endpoint: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh card. Error: {str(e)}")


@app.get("/api/content_stats")
async def get_content_stats(): # (unchanged)
    reset_tracker_if_needed()
    # ... (return stats)
    recent_titles_list = list(generated_content_tracker["titles"])
    return {
        "total_unique_titles_generated_session": len(generated_content_tracker["titles"]),
        "total_unique_content_hashes_session": len(generated_content_tracker["content_hashes"]),
        "tracker_last_reset_utc": generated_content_tracker["last_reset"].isoformat(),
        "recent_titles_example": recent_titles_list[-20:]
    }


@app.post("/api/generate_related_pyqs", response_model=RelatedPYQsResponse)
async def generate_related_pyqs_endpoint(request_data: RelatedPYQsRequest): # (unchanged)
    reset_tracker_if_needed()
    # ... (PYQ generation logic is the same)
    if model is None: raise HTTPException(status_code=503, detail="Gemini model not initialized.")
    topic_key_for_tracker = hashlib.md5(request_data.topic_title.lower().encode('utf-8')).hexdigest()
    if topic_key_for_tracker not in generated_content_tracker["pyq_ids_for_topic"]: generated_content_tracker["pyq_ids_for_topic"][topic_key_for_tracker] = set()
    current_topic_excluded_pyq_ids = generated_content_tracker["pyq_ids_for_topic"][topic_key_for_tracker].copy()
    if request_data.exclude_ids: current_topic_excluded_pyq_ids.update(request_data.exclude_ids)
    exclusion_ids_prompt_segment = f"IMPORTANT: Avoid generating questions identical ... to these identifiers: {', '.join(list(current_topic_excluded_pyq_ids)[-15:])}." if current_topic_excluded_pyq_ids else ""
    prompt = f"""Given UPSC card: Title: "{request_data.topic_title}", Subject: "{request_data.subject}", Content: "{request_data.topic_content[:400]}..."
    Generate {request_data.count} unique UPSC Prelims MCQs for this topic. {exclusion_ids_prompt_segment} ... JSON structure: {PYQ_ITEM_SCHEMA} ..."""
    try:
        generated_pyqs_json_string = await _call_gemini_model_json(prompt_text=prompt, response_schema=RELATED_PYQS_RESPONSE_SCHEMA, temperature=0.7)
        raw_pyqs_from_ai = json.loads(generated_pyqs_json_string)
        # ... (validation and processing of raw_pyqs_from_ai as before)
        processed_pyqs: List[GeneratedPYQItem] = []
        for pyq_data in raw_pyqs_from_ai:
            if not isinstance(pyq_data, dict) or not all(key in pyq_data for key in PYQ_ITEM_SCHEMA["required"]): continue
            # ... (rest of PYQ validation)
            try:
                pyq_item = GeneratedPYQItem(**pyq_data)
                generated_content_tracker["pyq_ids_for_topic"][topic_key_for_tracker].add(pyq_item.id)
                processed_pyqs.append(pyq_item)
            except Exception as pydantic_e: print(f"Pydantic validation error for PYQ: {pyq_data}. Error: {pydantic_e}"); continue
        return RelatedPYQsResponse(pyqs=processed_pyqs)
    except Exception as e:
        print(f"Error in generate_related_pyqs_endpoint: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PYQs: {str(e)}")


# REMOVED: The old /api/generate_cards_from_pplx endpoint.
# The new /api/generate_study_cards takes over general card generation.
# If a specific PPLX-driven research flow is still needed for some other purpose,
# it could be a separate endpoint, but for the main card view, we use the new Gemini batch endpoint.

# --- Main execution ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server for UPSC Prep Hub API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

