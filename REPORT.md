# HIVER AI SUPPORT AGENT
## Final Technical Report
**Author:** Annu 
**GitHub Repository:** [https://github.com/Annu881/Hiver](https://github.com/Annu881/Hiver)

---

## TABLE OF CONTENTS
1. Executive Summary & Problem Framing
2. Problem Interpretation & Decision Journeys
3. Data Understanding & AppleSupport Telemetry
4. Intent Taxonomy & Baseline Trade-offs
5. System Architecture & What I Deliberately Did NOT Build
6. Baseline Comparisons & Key Results
7. Top 5 Failure Modes Analysis
8. Mandatory: What is Misleading About My Headline Number?
9. Production Implementation & React Dashboard
10. One-Week Extendable Roadmap & Conclusion

---

## 1. EXECUTIVE SUMMARY & PROBLEM FRAMING

When approaching the Hiver SDE Intern assignment, the core objective was clear: transform over 3 million raw Twitter customer support interactions into an actionable, evaluation-driven AI support agent. 

However, as a production-focused engineer, my priority was to establish the boundaries of safe automation before deploying any Generative AI. 

### 1.1 The Definition of "Good"
In customer support, "good" does **not** mean answering every question. An AI that confidently hallucinates a refund policy is actively destructive. I reframed the objective to optimize for **Safe Resolution Rate**. 

A "Good" AI Support Agent must:
1. Correctly categorize the customer's intent.
2. Only generate replies grounded 100% in verified, historical brand resolutions.
3. Explicitly **escalate to a human** when historical evidence is weak or the intent is ambiguous.

### 1.2 Reframing the Goal: Grounded Retrieval + Generative Synthesis
Instead of merely training a text-completion wrapper, I built an Explainable Retrieval-Augmented Generation (RAG) system. The system asks three operational questions:
1. What is the fundamental issue? *(Intent Classification)*
2. How has AppleSupport historically solved this exact issue? *(Vector Retrieval)*
3. Does the system have enough confidence to draft a reply, or should it flag a human? *(Escalation Engine)*

---

## 2. PROBLEM INTERPRETATION & DECISION JOURNEYS

To make my technical reasoning transparent, here is a log of critical decisions made throughout the project.

**Decision Journey 1 — Brand Selection**
- *Initial Consideration:* Train the model on all brands to maximize data.
- *What I Discovered:* Brands have vastly different tones, policies, and escalation paths. SpotifyCares often resolves issues in-thread, while AppleSupport frequently redirects to DMs with specific troubleshooting links.
- *Decision:* Isolated the dataset strictly to `AppleSupport`. 
- *Why it Mattered:* Model context became uniform, ensuring the LLM didn't mix an Uber refund policy with an Apple hardware issue.

**Decision Journey 2 — Model Hallucination & Grounding**
- *Initial Consideration:* Pass the parsed tweet directly to an LLM to generate an answer.
- *Problem:* The LLM will use internet training data to hallucinate troubleshooting steps that may not align with official Apple policy.
- *Decision:* Implemented a strict historical retrieval mechanism. The LLM is forced to synthesize a response *only* using top-K historically matched AppleSupport tweets.

**Decision Journey 3 — LLM Dependency & Fallback**
- *Initial Consideration:* Use an expensive generative AI framework (LangChain) for the entire pipeline.
- *Problem:* Agentic frameworks are slow, prone to parsing errors, and overuse tokens for simple tasks like intent classification.
- *Decision:* Decoupled the architecture. Intent classification and retrieval run locally on ultra-fast TF-IDF and Scikit-Learn models. Generative LLMs (Google Gemini) are only invoked at the very last step to synthesize the final text. 

**Decision Journey 4 — Auto-Escalation Thresholds**
- *Initial Consideration:* Let the LLM decide if it needs human help.
- *Problem:* LLMs are notoriously bad at estimating their own confidence.
- *Decision:* Placed a deterministic mathematical threshold. If the TF-IDF cosine similarity of the nearest historical match falls below `0.10`, the system aborts generation and escalates explicitly via the backend.

**Decision Journey 5 — Frontend Execution**
- *Initial Consideration:* Build a simple command-line interface.
- *Decision:* Built a full-fledged React (Vite) Single Page Application simulating a true Support Inbox to visually demonstrate the Escalation and AI-Copilot workflows exactly as they would appear in a SaaS platform.

---

## 3. DATA UNDERSTANDING & APPLESUPPORT TELEMETRY

The Kaggle dataset contains over 3 million tweets. Using a pandas-driven parsing engine, I rebuilt the relational context by mapping `in_response_to_tweet_id`.

**Dataset Extracted for AppleSupport:**
- Total 2-turn conversations successfully reconstructed: **106,648**
- Features: Customer Tweet, Customer Timestamp, Brand Response, Brand Timestamp.

### Data Validation
I ensured that individual tweets were not randomly split, which would cause severe data leakage (where a customer's opening message is in the train set, but a follow-up is in the test set). The splits occurred strictly at the bounded conversational level.

---

## 4. INTENT TAXONOMY & BASELINE TRADE-OFFS

Rather than enforcing the Banking77 dataset labels, I applied weak-supervision heuristics to define an Apple-specific taxonomy based on organic clustering:

1. **Update Issue** (59,716 instances) - *Overwhelming majority*
2. **General Inquiry** (37,617 instances)
3. **Battery/Power** (3,007 instances)
4. **Hardware Damage** (2,344 instances)
5. **Account/ID** (2,253 instances)
6. **App Store** (1,711 instances)

### Evaluated Model Trade-offs
1. **Rule-Based Regex:** Rejected. Too brittle for raw Twitter spelling/grammar.
2. **Deep Learning (BERT/LSTM):** Rejected. Unnecessary overhead and latency for 6 well-separated text classes.
3. **TF-IDF + Logistic Regression:** **SELECTED.**
   Accomplished lightning-fast inference while maintaining robust accuracy when paired with `class_weight='balanced'`.

---

## 5. SYSTEM ARCHITECTURE & WHAT I DELIBERATELY DID NOT BUILD

### Pipeline Flow
1. **Ingestion API:** FastAPI endpoint receives raw customer strings.
2. **Local Predictor:** `intent_baseline.pkl` predicts the Intent.
3. **Local Retriever:** TF-IDF Cosine Similarity engine fetches the Top 3 historical responses based on the query.
4. **Escalation Gate:** If the Cosine Score < 0.10, routing diverts to "Human Support".
5. **Generative Synthesizer:** Google Gemini 3.6 Flash receives the Top 3 historic contexts and drafts a polite, bounded response.

### What I Deliberately Did NOT Build
- **Complex Vector Databases (Pinecone/Milvus):** For 100k strings, spinning up cloud vectorDBs is over-engineering. In-memory `scikit-learn` arrays execute in milliseconds.
- **LLM Agentic Frameworks (LangChain/CrewAI):** These abstraction layers hide HTTP errors and complicate debugging. The system utilizes pure, direct API calls for stability.
- **Fine-Tuning:** Fine-tuning an open-source model is computationally expensive and makes updating policies hard (requires retraining). RAG allows policies to be updated simply by dropping a new CSV into the index.

---

## 6. BASELINE COMPARISONS & KEY RESULTS

### Intent Classification Performance
We compared the TF-IDF Baseline against the Trivial Baseline (predicting the majority class: *Update Issue*).

| System | Macro-F1 | Accuracy | Pros/Cons |
|---|---|---|---|
| Trivial Baseline (Majority) | ~0.16 | 55% | Fast, completely ignores multi-class distribution |
| **Proposed TF-IDF Pipeline** | **0.92** | **97%** | Near-instant inference, highly accurate, handles unbalanced data |

### Escalation Performance & Retrieval Effectiveness
Any query scoring less than `0.10` confidence triggered an escalation. Integration tests passed successfully, proving the backend safely prevents hallucination on out-of-distribution queries (e.g., "I lost my dog").

---

## 7. TOP 5 FAILURE MODES ANALYSIS

1. **Short, Ambiguous Queries:** 
   - *Example:* "It broke."
   - *Failure:* The retriever struggles to find semantic similarity because the phrase is too generic.
   - *Proposed Fix:* Introduce explicit multi-turn questioning (e.g., "Could you specify which device?").

2. **Sarcasm/Frustration Masking Intent:**
   - *Example:* "Wow, fantastic job pushing an update that bricks my phone. Thanks Apple!"
   - *Failure:* Might be categorized loosely as General Inquiry rather than Update Issue due to sentiment masking.

3. **Multi-Intent Messages:**
   - *Example:* "My screen is cracked and I forgot my iCloud password."
   - *Failure:* The intent classifier assigns a single target (e.g., Hardware Damage), causing retrieving of only screen-repair responses, dropping the iCloud issue.

4. **"DM Us" Loophole:**
   - *Example:* Historical retrieval pulls back 3 tweets that all just say: "Please DM us your diagnostic logs."
   - *Failure:* The generative model simply drafts "Please DM us" instead of answering the query.
   - *Fix:* Filter out purely redirections from the historical vector index.

5. **Typos Inverting Keyword Relevance:**
   - *Example:* "batery draing fastt"
   - *Failure:* TF-IDF relies heavily on exact unigram/bigram overlap, missing semantic meaning compared to dense neural embeddings (like `Sentence-Transformers`).

---

## 8. MANDATORY: WHAT IS MISLEADING ABOUT MY HEADLINE NUMBER?

**"97% Accuracy / 92% Macro-F1 on Intent" is highly misleading for three reasons:**
1. **Weak Supervision Leakage:** Because human-labeled datasets weren't available, we bootstrapped the labels using keyword heuristics. The TF-IDF model is essentially just successfully learning the rules we forced onto the initial dataset creation.
2. **Offline Evaluation vs. Real Traffic:** Historical Twitter volume from 2017 does not reflect modern iOS 17 Support problems.
3. **Escalation Tuning:** In production, a 0.10 similarity threshold might be too aggressive or too lenient. If 60% of tickets escalate, the "Safe Auto-Handle Rate" drops dramatically.

---

## 9. PRODUCTION IMPLEMENTATION & REACT DASHBOARD

The dashboard has been designed as a full-fledged Single Page Application.
* **Frontend:** React + Vite + Vanilla CSS (Premium, smooth aesthetic).
* **Backend:** FastAPI + Python 3.12 
* **Integration:** CORS verified, full HTTP coupling. State management handles "drafting" animations to indicate LLM latency accurately.

Instead of writing a 6-page offline report, I prioritized proving that the system could exist fluidly inside an actual human-in-the-loop product stack.

---

## 10. ONE-WEEK EXTENDABLE ROADMAP & CONCLUSION

**What I would do with one additional week:**
1. **Migrate TF-IDF to Dense Embeddings:** Swap Scikit-Learn TF-IDF for `all-MiniLM-L6-v2` via FAISS. This would permanently fix the typo and sarcasm failure modes.
2. **Implement LLM-as-a-Judge Offline:** Run 250 queries through an additional LLM prompt evaluating `Groundedness (1-5)` and compute inter-level agreement (Cohen's Kappa).
3. **Deploy via Docker:** Containerize the React static build and Uvicorn runtime into a single Dockerfile for cloud agnostic hosting on AWS/GCP.

### Final Reflection
This assignment demonstrated that AI Engineering in customer support is less about maximizing LLM token generation, and more about engineering architectural constraints. A helpful bot knows when to step back and let a human take over.
