from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict
import os, sys, pickle

app = FastAPI(title="Hiver AI Support Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Pydantic models ----
class ChatRequest(BaseModel):
    message: str

class TriageRequest(BaseModel):
    id: str
    text: str

class ChatResponse(BaseModel):
    reply: str
    intent: str
    escalated: bool
    escalation_reason: Optional[str] = None
    retrieved_evidence: int = 0

class TriageResponse(BaseModel):
    intent: str
    confidence: float
    intent_scores: Dict[str, float]
    escalated: bool
    reply: str
    evidence: List[dict]
    retrieved_evidence: int

# ---- Models (loaded at startup) ----
sys.path.append('..')
from src.retrieval.retriever import HistoricalRetriever

pipeline = None
retriever = None

@app.on_event("startup")
def load_models():
    global pipeline, retriever
    try:
        with open("../models/intent_baseline.pkl", "rb") as f:
            pipeline = pickle.load(f)
        retriever = HistoricalRetriever(data_path="../apple_conversations.csv", build=False, model_path="../models/retriever.pkl")
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

# ---- Gemini Client ----
from google import genai

API_KEY = os.environ.get("GEMINI_API_KEY", "")
gemini_client = genai.Client(api_key=API_KEY) if API_KEY else None

# ---- Helper: run the full RAG pipeline ----
def run_pipeline(text: str):
    if not pipeline or not retriever:
        return None, 0.5, {}, [], 0
    intent = str(pipeline.predict([text])[0])
    # Get probability scores
    probs = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    intent_scores = {str(c): float(p) for c, p in zip(classes, probs)}
    confidence = float(max(probs))

    results = retriever.retrieve(text, top_k=3)
    evidence = [{"customer_tweet": r.get("customer_tweet", ""), "brand_reply": r.get("brand_reply", ""), "score": float(r.get("score", 0))} for r in results]
    return intent, confidence, intent_scores, evidence, len(results)

def generate_reply(text: str, intent: str, evidence: list) -> tuple:
    best_score = evidence[0]["score"] if evidence else 0
    escalated = best_score < 0.1

    if escalated:
        return "Support Request Escalated: Low confidence match. Routing to human agent.", True

    if not gemini_client:
        return evidence[0]["brand_reply"] if evidence else "Unable to generate reply.", False

    try:
        historical_contexts = "\n".join([f"- {e['brand_reply']}" for e in evidence])
        prompt = (
            f"You are an AppleSupport AI Agent. A customer said: '{text}'. "
            f"The detected intent is '{intent}'. "
            f"Historically, we resolve similar issues by saying:\n{historical_contexts}\n\n"
            f"Draft a polite, helpful response grounded ONLY in this historical precedence. Keep it concise and conversational."
        )
        response = gemini_client.models.generate_content(model='gemini-3.6-flash', contents=prompt)
        return response.text.strip(), False
    except Exception as e:
        return evidence[0]["brand_reply"] if evidence else f"Error: {e}", False

# ---- Routes ----

@app.get("/health")
def health():
    return {"status": "ok", "models": pipeline is not None}

@app.get("/queue")
def get_queue():
    """Return the inbound queue seed items for the Triage Console."""
    return [
        {"id": "t_1184", "handle": "@marlow_j", "ts": "2m", "text": "since ios update my phone dies at 40% and gets hot as hell. 11 pro, less than a year old. this is the third update thats broken something", "thread": []},
        {"id": "t_1185", "handle": "@dana.k", "ts": "6m", "text": "this is the FOURTH time im messaging you. nobody has fixed anything. i want a replacement or my money back, im done being polite about it", "thread": ["@dana.k 2d ago: macbook screen flickering", "@AppleSupport 2d ago: we'd like to look into this, DM us", "@dana.k 1d ago: sent the DM, no reply"]},
        {"id": "t_1186", "handle": "@r_whitcombe", "ts": "11m", "text": "airpods keep disconnecting from my mac but work fine on the phone. tried forgetting the device twice", "thread": []},
        {"id": "t_1187", "handle": "@sofia__tt", "ts": "18m", "text": "charged twice for icloud storage this month. can you check my account?", "thread": []},
        {"id": "t_1188", "handle": "@kev_ngata", "ts": "24m", "text": "update stuck on the apple logo for 2 hours now. cant turn it off, cant do anything. tried holding buttons", "thread": []},
        {"id": "t_1189", "handle": "@tomasz.w", "ts": "31m", "text": "how do i move my photos to a new phone without losing the albums? about 40gb worth", "thread": []}
    ]

@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest):
    try:
        intent, confidence, intent_scores, evidence, num_retrieved = run_pipeline(req.text)
        reply, escalated = generate_reply(req.text, intent or "unknown", evidence)
        return TriageResponse(
            intent=intent or "unknown",
            confidence=confidence,
            intent_scores=intent_scores,
            escalated=escalated,
            reply=reply,
            evidence=evidence,
            retrieved_evidence=num_retrieved
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        return TriageResponse(intent="Error", confidence=0, intent_scores={}, escalated=True, reply=f"Error: {e}", evidence=[], retrieved_evidence=0)

@app.post("/api/chat", response_model=ChatResponse)
def handle_chat(req: ChatRequest):
    try:
        intent, confidence, intent_scores, evidence, num_retrieved = run_pipeline(req.message)
        reply, escalated = generate_reply(req.message, intent or "unknown", evidence)
        return ChatResponse(reply=reply, intent=intent or "unknown", escalated=escalated, retrieved_evidence=num_retrieved)
    except Exception as e:
        import traceback; traceback.print_exc()
        return ChatResponse(reply=f"Error: {e}", intent="Error", escalated=True, retrieved_evidence=0)

# ---- Serve the Triage Console HTML at root ----
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "triage.html")

@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_PATH)
