# Multi-Turn Context-Aware Causal Reasoning Engine

### Data Science Hackathon Submission

---

## 1. Problem Overview

Customer support systems generate large volumes of conversational transcripts. While outcome events such as refunds, complaints, or escalations are recorded, the **causal conversational factors** leading to these outcomes remain implicit.

This project builds a **context-aware analytical reasoning engine** that identifies causal drivers from multi-turn conversations and links them to evidence grounded in transcript data.

---

## 2. Solution Architecture

The system is implemented as a deterministic retrieval and reasoning pipeline:

```
Conversational Transcripts
        ↓
Turn Structuring & Chunking
        ↓
Sentence Embedding Generation
        ↓
FAISS Vector Indexing
        ↓
Semantic Retrieval
        ↓
Session Context Memory
        ↓
Causal Factor Extraction
        ↓
Explanation Formatting Engine
```

---

## 3. Key Capabilities

### 3.1 Multi-Turn Context Awareness

The engine preserves analytical context across queries using explicit session memory.

Supports:

* Follow-up reasoning
* Evidence reuse
* Transcript filtering
* Context refinement

---

### 3.2 Evidence-Grounded Reasoning

All outputs are strictly derived from retrieved transcripts.

Each response includes:

* Transcript IDs
* Turn spans
* Evidence counts
* Similarity scores

This ensures **faithfulness** and prevents hallucination.

---

### 3.3 Causal Factor Taxonomy

Extracted factors are normalized into a structured taxonomy:

| Raw Mentions        | Normalized Factor                      |
| ------------------- | -------------------------------------- |
| Missing delivery    | Incomplete or Missing Delivery         |
| Refund errors       | Payment Processing or Refund Errors    |
| Broken products     | Repeated Defective Product Delivery    |
| Repeated complaints | Repeated Unresolved Customer Complaint |

This improves interpretability and evaluation consistency.

---

### 3.4 Temporal Causality Detection

The system distinguishes:

* **Pre-outcome causal evidence**
* **Post-outcome contextual mentions**

This ensures causal claims are temporally grounded.

---

### 3.5 Comparative Reasoning

The engine supports analytical comparisons such as:

* Product Defects vs Billing Issues
* Evidence strength ranking
* Transcript coverage analysis
* Frequency comparison

---

## 4. Causal Explanation Structure

Each response is formatted deterministically:

**Cause**
Identified driver of the outcome.

**Mechanism**
How the conversational factor leads to the outcome.

**Evidence References**
Transcript IDs and turn spans.

**Factor Ranking**
Frequency, similarity, and coverage metrics.

---

## 5. Evaluation Dataset

A diverse multi-domain query dataset was created covering:

* Initial causal queries
* Evidence requests
* Comparative analysis
* Factor validation
* Clarification queries
* Context-dependent follow-ups

File included:

```
evaluation_queries.csv
```

---

## 6. Sample System Output

Example causal reasoning response:

```
Query: Why do customers request refunds?

Intent: new_causal_query
Follow-up: False
Evidence Reused: 0
New Evidence: 6

[Temporal Analysis]
Pre-outcome causal: 3
Post-outcome contextual: 3

CAUSAL EXPLANATION REPORT
------------------------------------------------------------

FACTOR: Incomplete or Missing Delivery

Cause:
Orders received incomplete or with missing items.

Mechanism:
Customer complaints regarding missing items
precede refund authorization requests, indicating
a causal escalation pattern.

Evidence References:
- Transcript 1495-6317-3915-8298 (Turns 1–2)
- Transcript 3866-9893-6254-7117 (Turns 4–5)

------------------------------------------------------------

FACTOR RANKING

Rank | Factor Name                    | Frequency | Coverage
------------------------------------------------------------
1    | Incomplete or Missing Delivery | 6         | 3
```

This demonstrates:

* Evidence grounding
* Temporal causality
* Factor normalization
* Deterministic reasoning

---

## 7. Implementation Files

Core modules:

* `context_engine.py` — End-to-end reasoning engine
* `retrieval.py` — Semantic search
* `vector_index.py` — FAISS indexing
* `embeddings.py` — Embedding generation
* `evidence_extraction.py` — Evidence parsing
* `causal_explainer.py` — Factor detection
* `causal_explanation_formatter.py` — Structured reporting
* `session_memory.py` — Context tracking

---

## 8. Technology Stack

* Python 3.10
* SentenceTransformers (all-MiniLM-L6-v2)
* FAISS
* NumPy
* Pandas
* Scikit-learn

---

## 9. Reproducibility

### Install dependencies

```
pip install -r requirements.txt
```

### Run system

```
python src/context_engine.py
```

---

## 10. Design Philosophy

Task-2 is implemented as a **stateful analytical reasoning engine rather than a conversational generator**.

Context is preserved through deterministic session memory and evidence reuse to ensure:

* Faithfulness
* Consistency
* Traceability

---

## 11. Deliverables Included

* Source code (Task-1 & Task-2)
* Embeddings & FAISS index
* Evaluation query dataset
* requirements.txt
* README documentation

---

## 12. Conclusion

This system transforms conversational transcripts into explainable causal intelligence through:

* Semantic retrieval
* Context persistence
* Evidence grounding
* Temporal reasoning
* Comparative analytics

It provides a scalable foundation for conversational outcome diagnostics.

---

## ---

## 13. Task Alignment with Problem Statement

### Task-1: Query-Driven Causal Explanation with Evidence

The system satisfies Task-1 through a deterministic causal reasoning pipeline that:

* Accepts natural language analytical queries
* Performs semantic retrieval over transcript chunks
* Identifies recurring conversational factors
* Maps factors to a normalized causal taxonomy
* Extracts supporting dialogue spans
* Links causal drivers to outcome events

Each explanation includes:

* Causal factor identification
* Mechanism description
* Transcript evidence references
* Factor ranking metrics

This ensures explanations are interpretable, traceable, and grounded in conversational data.

---

### Task-2: Multi-Turn Context-Aware Query Handling

Task-2 is implemented as a **stateful analytical reasoning engine**.

Key capabilities include:

* Persistent session memory across queries
* Evidence reuse from prior analyses
* Context-dependent query interpretation
* Transcript filtering using prior evidence pools
* Comparative factor reasoning
* Follow-up clarification handling

Example follow-up handling:

```
Initial Query:
Why do customers request refunds?

Follow-Up:
Are there more cases of delivery problems?

System Action:
Filters prior transcripts → retrieves refined evidence → updates causal analysis.
```

---

### Evaluation Metric Alignment

The system is designed to align with judging metrics:

**ID Recall (Evidence Accuracy)**
→ Transcript IDs and turn spans explicitly referenced.

**Faithfulness (Hallucination Control)**
→ Responses strictly derived from retrieved evidence.

**Relevancy (Conversational Coherence)**
→ Follow-up queries resolved using session context and prior reasoning outputs.

---

This implementation moves beyond event detection toward causal analysis and interactive reasoning over conversational data, fully aligning with the objectives of the problem statement.

---
# Do these to see the result
pip install -r requirements.txt
python context_engine.py
