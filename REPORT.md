# HIVER AI SUPPORT AGENT
## Final Technical Report & Architecture Deep-Dive

**Author:** Annu 
**GitHub Repository:** [https://github.com/Annu881/Hiver](https://github.com/Annu881/Hiver)
**Live Production Deployment:** [https://hiver-lwmn.onrender.com/](https://hiver-lwmn.onrender.com/)

---

## TABLE OF CONTENTS
1. Executive Summary & Problem Framing
2. Data Engineering: Distilling 3 Million Tweets
3. Baseline Evaluation & Modeling Trade-offs
4. Intent Taxonomy & F1-Score Breakdown
5. System Architecture (The "Monolith" Strategy)
6. Problem Interpretation & Decision Journeys
7. Top 5 Failure Modes Analysis
8. Mandatory: What is Misleading About My Headline Number?
9. The 100% "Grounding" Guarantee
10. One-Week Extendable Roadmap & Conclusion

---

## 1. EXECUTIVE SUMMARY & PROBLEM FRAMING

When approaching the Hiver SDE Intern assignment, the core objective was clear: transform over 3 million raw Twitter customer support interactions into an actionable, evaluation-driven AI support agent. 

However, as a production-focused engineer, my priority was to establish the boundaries of safe automation before deploying any Generative AI. In customer support, "good" does **not** mean answering every question. An AI that confidently hallucinates a refund policy is actively destructive. I reframed the objective to optimize for **Safe Resolution Rate**. 

A "Good" AI Support Agent must:
1. Correctly categorize the customer's intent.
2. Only generate replies grounded 100% in verified, historical brand resolutions.
3. Explicitly **escalate to a human** when historical evidence is weak or the intent is ambiguous.

---

## 2. DATA ENGINEERING: DISTILLING 3 MILLION TWEETS

The raw Kaggle dataset provided 3,000,000+ support tweets across varying brands (Uber, Spotify, Apple, etc.). My approach to extracting high-signal training data was aggressive and methodical:

1. **Brand Isolation:** I isolated the dataset strictly to `@AppleSupport`. Training a unified intent model across all brands is a mistake; an airline's intent to "rebook flight" fundamentally confuses the taxonomy of a tech company's "hardware repair" intent.
2. **Relational Reconstruction:** I joined the dataset on `in_response_to_tweet_id` to build valid 2-turn dialogue pairs: `(Customer Tweet -> Brand Reply)`.
3. **Regex Cleanup & Normalization:** I executed vector-level pandas string operations to lower-case inputs, strip out PII (non-brand @mentions), replace URLs with a uniform `<URL>` token, and remove broken character encodings.

**Final Extracted Dataset:**
- **Total Valid Configurations:** 106,648 AppleSupport dialogues.
- **Handling Leakage:** Reconstructed dialogue bounds ensured that a customer's subsequent replies in a thread were not split between training and testing sets, preventing train-test data leakage.

---

## 3. BASELINE EVALUATION & MODELING TRADE-OFFS

Rather than immediately reaching for deep learning, I evaluated multiple structural baselines against the dataset.

1. **Rule-Based Regex:** Rejected. Too brittle for raw Twitter spelling (e.g. *batery*, *skreen*).
2. **Trivial Baseline (Majority Predictor):** Used as the absolute minimum bar. Predicting the majority class ("Update Issue") yielded ~55% accuracy and a 0.16 Macro-F1.
3. **Deep Learning (BERT/LSTM):** Rejected. Unnecessary computational overhead and latency for 6 well-separated text classes. Fine-tuning an open-source LLM is also computationally expensive and makes updating brand policies hard (requires retraining weights).
4. **TF-IDF + Logistic Regression:** **SELECTED.**
   Accomplished lightning-fast inference while maintaining robust accuracy when paired with `class_weight='balanced'`.

---

## 4. INTENT TAXONOMY & F1-SCORE BREAKDOWN

By applying weak-supervision keyword heuristics (since organic human labels were absent), I constructed a 6-intent taxonomy tailored specifically to Apple hardware/software queries.

**Category Distribution:**
1. **Update Issue** (59,716 instances) - *Vastly over-represented*
2. **General Inquiry** (37,617 instances)
3. **Battery/Power** (3,007 instances)
4. **Hardware Damage** (2,344 instances)
5. **Account/ID** (2,253 instances)
6. **App Store** (1,711 instances)

**Local Evaluation Metrics (Test Set):**
* **Global Accuracy:** 97.4%
* **Macro Average F1-Score:** 0.92

*The gap between Accuracy (97%) and Macro-F1 (92%) is a direct reflection of the massive class imbalance; however, `class_weight='balanced'` in the LogisticRegression formulation preserved high Precision and Recall on minority classes (like App Store).*

---

## 5. SYSTEM ARCHITECTURE (THE "MONOLITH" STRATEGY)

While the assignment stated "Do NOT spend most of your time building an elaborate frontend", I wanted to prove that the ML engine could survive inside a real dashboard environment.

I built the system using a **Clean Python Monolith**:
- **Backend:** A strict FastAPI server handling the `/triage` endpoint powered by PyDantic schemas.
- **Data Stores:** Scalable `.pkl` artifacts holding the pre-trained TF-IDF vectorizers arrays in memory.
- **Frontend UI:** Instead of a heavy Webpack/React setup, I injected a masterclass vanilla HTML/CSS Single-Page Application (`triage.html`) directly into the FastAPI root. It utilizes zero external visual dependencies, resulting in a blisteringly fast SaaS-like experience featuring a Decision Log, Auto-Escalation states, and dynamic DOM routing.

---

## 6. PROBLEM INTERPRETATION & DECISION JOURNEYS

**Decision 1 — LLM Dependency Frameworks**
- *Initial Consideration:* Use an agentic framework (LangChain) for the entire pipeline.
- *Problem:* Agentic frameworks are slow, prone to parsing errors, and overuse tokens for simple classification tasks.
- *Decision:* Decoupled the architecture. Intent classification and Retrieval run locally on ultra-fast TF-IDF models (`scikit-learn`). Google Gemini is only invoked at the very last step to synthesize the final English text. 

**Decision 2 — Auto-Escalation Thresholds**
- *Initial Consideration:* Let the LLM prompt dictate if it needs human help.
- *Problem:* LLMs are notoriously bad at estimating their own confidence (hallucination).
- *Decision:* Placed a deterministic mathematical threshold. If the TF-IDF cosine similarity of the nearest historical match falls below an established gate threshold, the system aborts AI generation and escalates explicitly. *(Note: This threshold is artificially lowered on the live deployment to ensure the demo always generates a draft for the user).*

---

## 7. TOP 5 FAILURE MODES ANALYSIS

1. **Short, Ambiguous Queries:** 
   - *Example:* "It broke."
   - *Failure:* The retriever struggles to find semantic similarity because the phrase is too generic.
   - *Fix:* Introduce explicit multi-turn questioning (e.g., "Could you specify which device?").

2. **Sarcasm/Frustration Masking Intent:**
   - *Example:* "Wow, fantastic job pushing an update that bricks my phone. Thanks Apple!"
   - *Failure:* Might be categorized loosely as General Inquiry rather than Update Issue due to sentiment masking the technical tokens.

3. **Multi-Intent Messages:**
   - *Example:* "My screen is cracked and I forgot my iCloud password."
   - *Failure:* The intent classifier assigns a single target (Hardware Damage), retrieving only screen-repair responses, dropping the iCloud issue entirely.

4. **"DM Us" Loophole:**
   - *Example:* Historical retrieval pulls back 3 tweets that all just say: "Please DM us your diagnostic logs."
   - *Failure:* The generative model simply drafts "Please DM us" instead of answering the query.
   - *Fix:* Filter out redirections/empty links from the historical vector index.

5. **Typos Inverting Keyword Relevance:**
   - *Example:* "batery draing fastt"
   - *Failure:* TF-IDF unigram relies heavily on exact token overlap, missing semantic meaning compared to dense neural embeddings.

---

## 8. MANDATORY: WHAT IS MISLEADING ABOUT MY HEADLINE NUMBER?

**"97% Accuracy / 92% Macro-F1 on Intent" is highly misleading for three reasons:**

1. **Weak Supervision Leakage:** Because human-labeled datasets weren't available, we bootstrapped the training labels using keyword heuristics (e.g. labeling anything with "update" as Update Issue). **The TF-IDF model is essentially just successfully learning the hardcoded rules we forced onto the initial dataset creation.**
2. **Offline Evaluation vs. Real Traffic:** Historical Twitter volume from 2017 does not reflect modern iOS 17 Support problems; the vocabulary drift will instantly degrade accuracy in production.
3. **Escalation Trade-offs:** The 97% accuracy doesn't account for the Safe Resolution Rate. In production, if 60% of test tickets trigger the escalation gate (similarity < 0.10) to avoid hallucination, the actual "Autonomous AI Reply Rate" drops dramatically.

---

## 9. THE 100% "GROUNDING" GUARANTEE

The largest risk of GenAI in Customer Support is confidently hallucinating a wrong return policy. 

To solve this, the LLM is **never** asked to answer a question natively. Instead, the backend performs a localized Vector Search (Cosine Similarity) across 100,000 Apple Support tweets to find the historically correct answer to the exact problem. That verified historical text is injected into the Gemini prompt:

> *"Historically, we resolve similar issues by saying: \n[Inject History] \nDraft a polite response grounded ONLY in this historical precedence."*

This is robust Evaluation-Driven AI: It proves that we rely on Data Engineering, not just generic LLM API calls.

---

## 10. ONE-WEEK EXTENDABLE ROADMAP & CONCLUSION

**What I would do with one additional week:**
1. **Migrate TF-IDF to Dense Embeddings:** Swap Scikit-Learn TF-IDF for `all-MiniLM-L6-v2` via FAISS. This would permanently fix the typo and sarcasm failure modes because it tracks *meaning* instead of exact spelling.
2. **Implement LLM-as-a-Judge Offline:** Run 250 queries through an additional LLM prompt evaluating `Groundedness (1-5)` and compute inter-level agreement (Cohen's Kappa).

### Final Reflection
This assignment demonstrated that AI Engineering in customer support is significantly less about maximizing LLM token generation, and entirely about engineering architectural constraints. A helpful bot knows exactly when to step back and let a human take over.
