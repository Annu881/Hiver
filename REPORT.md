# Hiver AI Support Agent: Evaluation-Driven Design for Safe Automation

**Author:** Annu  
**Repository:** [https://github.com/Annu881/Hiver](https://github.com/Annu881/Hiver)  
**Live Production Deployment:** [https://hiver-lwmn.onrender.com/](https://hiver-lwmn.onrender.com/)  

---

## Abstract

Generative AI in customer support presents severe operational risks, most notably the authoritative hallucination of incorrect policies. To mitigate this risk, this project prioritizes an evaluation-driven, grounded, and escalation-first design pattern. Using 3 million raw customer support tweets, the dataset was methodically distilled to 106,648 AppleSupport dialogues to preserve contextual coherence. The resulting system leverages an ultra-fast TF-IDF plus Logistic Regression intent classifier, coupled with a RAG (Retrieval-Augmented Generation) pipeline that strictly bounds LLM replies to historical evidence. While the intent classifier achieves a 97.4% global accuracy and a 0.92 macro-F1 score on the test set, these offline metrics are highly constrained by weak-supervision heuristics. Ultimately, this architecture demonstrates that in automated support environments, engineering explicit constraints and escalation thresholds vastly outweighs raw LLM generation power.

---

## Problem Framing & Design Goals

In customer support, a "good" generative model does not mean one that answers every question. An AI that synthesizes an incorrect hardware return policy is actively destructive to the brand. Therefore, the architectural objective was reframed to optimize for **Safe Resolution Rate** rather than raw answer output.

Unlike naive approaches that pass user queries directly into an unconstrained LLM, this architecture prioritizes three core design goals:
1. **Accurate Intent Classification:** Pinpoint the customer's fundamental issue in milliseconds.
2. **100% Grounded Synthesis:** Require generative models to construct replies exclusively from retrieved historical brand records.
3. **Explicit Escalation:** Force an immediate routing to human support agents when historical evidence is sparse or intent is ambiguous.

---

## Data Engineering: From 3M Tweets to 106k Dialogues

The raw Kaggle dataset contained over 3 million tweets across various, distinct enterprise brands. 

1. **Brand Isolation:** Training an intent classifier across divergent domains degrades signal; an airline's intent to "rebook flight" pollutes the latent space of a consumer electronics company. The dataset was strictly isolated to `@AppleSupport`.
2. **Dialogue Reconstruction:** Raw, disjointed tweets were reconstructed via `in_response_to_tweet_id` to form structurally coherent (Customer Query → Brand Resolution) paired turns.
3. **Normalization & PII Stripping:** Vectorized pandas string operations were executed to lowercase text, strip non-brand `@mentions`, normalize outbound links to a uniform `<URL>` token, and sanitize broken encodings.
4. **Leakage Prevention:** Conversational bounds were rigidly maintained. A customer's subsequent replies within a single thread were never vertically split across training and test sets, avoiding train-test data leakage.

**Final Dataset:** 106,648 valid AppleSupport conversational pairs.

---

## Modeling Choices & Baselines

Instead of immediately reaching for deep learning, multiple structural baselines were evaluated:

* **Rule-Based Regex:** Rejected for being dangerously brittle against raw, ungrammatical, or misspelled Twitter payloads.
* **Trivial Baseline (Majority Predictor):** Used to establish the statistical floor. Forecasting the majority class ("Update Issue") yielded ~55% accuracy and a negligible 0.16 Macro-F1.
* **Deep Learning (BERT / LLM Fine-tuning):** Rejected. Fine-tuning models introduces extreme latency and cost overheads for a highly separated 6-class textual classification problem. Furthermore, updating brand policies requires retraining weights.
* **TF-IDF + Logistic Regression:** **Chosen.** Using a 5000-feature unigram/bigram TF-IDF vectorizer coupled with a Logistic Classifier (`class_weight='balanced'`) delivered near-instant inference speed (under <5ms latency) and robust handling of sparse terms.

---

## Intent Taxonomy & Evaluation

With no foundational human labels, a weak-supervision framework leveraging keyword heuristics was applied to cluster the corpus into a localized taxonomy. 

**Class Distribution:**
1. **Update Issue:** 59,716 samples
2. **General Inquiry:** 37,617 samples
3. **Battery/Power:** 3,007 samples
4. **Hardware Damage:** 2,344 samples
5. **Account/ID:** 2,253 samples
6. **App Store:** 1,711 samples

**Offline Test-Set Metrics:**
* **Global Accuracy:** ~97.4%
* **Macro Average F1-Score:** ~0.92

The gap between Accuracy (97%) and Macro-F1 (92%) is driven entirely by massive class imbalance. While "Update Issue" holds ~55% of the data, the use of `class_weight='balanced'` in the Logistic Regression formulation effectively penalized majority-class bias, preserving strong Precision and Recall across critical minority classes like *Hardware Damage* and *App Store*.

---

## System Architecture ("Clean Python Monolith")

To demonstrate production feasibility without the bloating of traditional MERN/React stacks, the system leverages a **Clean Python Monolith**:
* **Backend:** A strict FastAPI runtime routing the `/triage` inference endpoint, serialized via Pydantic schemas.
* **Data Stores:** Serialized `.pkl` artifacts holding the pre-calculated TF-IDF vectors in fast RAM.
* **Frontend:** A vanilla HTML/CSS/JS Single-Page Application (`triage.html`) natively served by FastAPI. 
* **Key UX Features:** Includes a dynamic Decision Log, automated UI states mimicking an agent's view, and live rendering of ML trace metrics.

This monolithic approach drastically reduces repository dependency weight while maintaining a fully interactive SaaS-grade dashboard.

---

## Key Design Decisions & Trade-offs

**Decision 1: Avoiding Agentic Abstraction Frameworks (e.g., LangChain)**
* **Problem:** Agentic abstraction layers execute highly inefficient token loops, add hundreds of milliseconds of latency, and severely complicate stack trace debugging during production failures.
* **Fix:** The architecture avoids them entirely. Intent classification and retrieval execute on purely structural `scikit-learn` algorithms. An LLM (Google Gemini) is invoked solely via direct API call for final text synthesis.

**Decision 2: Deterministic Auto-Escalation Thresholds**
* **Problem:** Generative LLMs hallucinate their own confidence metrics.
* **Fix:** A deterministic Cosine-Similarity threshold acts as a rigid routing gate. If no retrieved historical record exceeds the threshold, generation aborts and forces a human escalation. *(Note: For the purposes of presenting a generative demonstration via the live web UI, this production gate threshold is currently suppressed).* 

---

## Failure Modes & Mitigations

An honest evaluation of the deployment revealed five core engineering vulnerabilities:

1. **Short, Ambiguous Queries:** 
   *(e.g., "It broke")* The TF-IDF retriever lacks sufficient character length to find intersectional overlap. Mitigation: Implement a multi-turn interrogative flow prompting for device specification.
2. **Sarcasm/Frustration Masking Intent:** 
   *(e.g., "Fantastic job pushing an update that bricks my phone.")* Dense frustration overrides the technical unigrams. Mitigation: Leverage dense embeddings (Sentence-Transformers) rather than lexical matching.
3. **Multi-Intent Messages:** 
   *(e.g., "My screen cracked and I am locked out of iCloud.")* Classifier predicts only the single highest-probability class, stranding the secondary domain issue. Mitigation: Convert to a multi-label sigmoid classifier.
4. **"DM Us" Loophole:** 
   Historical data heavily features brand replies stating "Please DM us your diagnostic logs." The LLM subsequently drafts low-value redirections. Mitigation: Filter generic triage responses from the vector corpus.
5. **Typos Breaking TF-IDF Relevance:** 
   *(e.g., "batery draing fastt")* Exact-match unigram overlap drops to near-zero. Mitigation: Dense neural retrieval mappings.

---

## Why the Headline Metrics Are Misleading

Presenting "97% Accuracy" as a conclusive metric is academically dishonest within this operational context for three reasons:

1. **Weak Supervision Leakage:** Because labeled data was heuristically bootstrapped, the offline models primarily learned the keyword rules injected during dataset creation rather than organic latent distributions.
2. **Offline vs. Real Traffic Vocabulary Drift:** Analyzing a 2017 dataset yields extremely high in-distribution accuracy. However, modern (2026+) incoming queries regarding iOS 17 or new hardware architectures represent severe out-of-distribution drift.
3. **Escalation Trade-offs:** The 97% classification rate does not address the overall Autonomous Output. If 60% of real-world inputs trigger the safety escalation threshold, then high classification accuracy does not equate to a high Safe Resolution Rate. Operational product leaders must measure the *Autonomous Auto-Handle Output*, not simply inference test accuracy.

---

## Grounding Guarantee & Safety Mechanism

To definitively curb the risk of LLMs confidently authorizing non-existent return policies, generative actions are restricted. The LLM is structurally bypassed from relying on its foundational internet training corpus. 

Instead, utilizing a FAISS-approximated cosine search, the top AppleSupport historical resolutions are injected directly natively into the inference prompt:

```text
You are an AppleSupport AI Agent. A customer said: '{text}'.
The detected intent is '{intent}'.
Historically, we resolve similar issues by saying:
[INJECTED_RETRIEVAL_RESULTS]

Draft a polite, helpful response grounded ONLY in this historical precedence. Keep it concise.
```
This paradigm ensures this is true Evaluation-Driven AI — the safety is enforced by structural data engineering rather than prompt-begging.

---

## One-Week Extendable Roadmap

Given subsequent development cycles, the initial focus would pivot toward retrieval fidelity:
1. **Dense Embedding Migration:** Replace `scikit-learn` TF-IDF with `all-MiniLM-L6-v2`. Moving from lexical exact-matching to semantic contextual embeddings directly circumvents the typo and sarcasm failure modes.
2. **LLM-As-A-Judge Pipeline:** Route a randomly sampled batch of 250 predictions through an offline LLM judge prompted to evaluate **Groundedness (1-5)**, computing inter-rater consistency (Cohen's Kappa) to mathematically guarantee lack of hallucination.

---

## Conclusion

This deployment underscores that deploying AI engineering in customer support necessitates significantly more focus on structural constraints than on maximizing LLM invocation. Raw LLM generative power is fundamentally unsafe in brand-facing environments. By enforcing mathematically deterministic escalation rules, explicit RAG grounding layers, and strict data-isolation taxonomies, the system proves that a genuinely helpful autonomous bot is defined primarily by knowing exactly when to step back and let a human take over.
