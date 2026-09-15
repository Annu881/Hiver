# Hiver AI Support Agent

A full-stack, evaluation-driven AI customer support agent built for the Hiver SDE Intern assignment. 

This repository implements a robust ML Retrieval-Augmented Generation (RAG) pipeline using the Google Gemini 3.6 Flash API to draft support responses. The backend restricts hallucination by requiring generation to be strictly grounded in historical customer interactions (extracted from the AppleSupport Twitter dataset).

## Features
- **Generative AI with Grounding:** Uses Google Gemini combined with a local TF-IDF nearest-neighbor retrieval system to ensure all drafted policies actually exist.
- **Support UI:** A premium, modern React frontend without Tailwind, designed for a Support Agent's inbox.
- **Auto-Escalation:** Rejects generic/low-confidence historical matches with an explicit "Support Request Escalated" flag to loop in humans safely.

## Architecture
- `src/` - Data parsing, ML model training, and historical TF-IDF retrieval pipeline.
- `backend/` - FastAPI wrapper serving `/api/chat` for real-time AI generation.
- `frontend/` - React (Vite) UI.

## Getting Started

1. Set up the ML Environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

2. Configure Google Gemini:
Ensure you have the valid API Key in the backend environment.

3. Run the Backend:
```bash
cd backend
uvicorn app:app --port 8000
```

4. Run the Frontend:
```bash
cd frontend
npm install
npm run dev
```
