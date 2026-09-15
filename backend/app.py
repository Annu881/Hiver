from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI(title="Hiver AI Support Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from typing import Optional

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    intent: str
    escalated: bool
    escalation_reason: Optional[str] = None
    retrieved_evidence: int = 0

import sys
sys.path.append('..')
from src.retrieval.retriever import HistoricalRetriever
import pickle

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

import os
from google import genai
from google.genai import types

# Setup Gemini Client using the provided key (Set as environment variable locally)
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if API_KEY:
    try:
        gemini_client = genai.Client(api_key=API_KEY)
    except Exception as e:
        gemini_client = None
else:
    gemini_client = None

@app.post("/api/chat", response_model=ChatResponse)
def handle_chat(req: ChatRequest):
    try:
        if not pipeline or not retriever:
            return ChatResponse(reply="Backend is still loading models or models are missing.", intent="Error", escalated=True, retrieved_evidence=0)
        
        intent = str(pipeline.predict([req.message])[0])
        results = retriever.retrieve(req.message, top_k=3)
        
        # Heuristic escalation: If best retrieve score < 0.1, escalate!
        best_score = results[0]['score'] if results else 0
        escalated = best_score < 0.1
        
        if escalated:
            reply = "Support Request Escalated: Low confidence in historical match."
            reason = f"Top similarity score was only {best_score:.2f} (Threshold 0.1)"
        else:
            reason = None
            if not gemini_client:
                reply = "Generative AI failed to initialize. " + results[0]['brand_reply']
            else:
                historical_contexts = "\n".join([f"- {r['brand_reply']}" for r in results])
                prompt = (f"You are an AppleSupport AI Agent. A customer just said: '{req.message}'. "
                          f"The detected intent is '{intent}'. "
                          f"Historically, we resolve similar issues by saying:\n{historical_contexts}\n\n"
                          f"Draft a polite, helpful response grounded ONLY in this historical precedence. Keep it concise.")
                
                response = gemini_client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=prompt
                )
                reply = response.text
            
        return ChatResponse(
            reply=reply,
            intent=intent,
            escalated=bool(escalated),
            escalation_reason=reason,
            retrieved_evidence=len(results)
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ChatResponse(reply=f"Error: {e}", intent="Error", escalated=True, retrieved_evidence=0)
