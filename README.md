# Aster & Row — Reliable Customer Support Agent

> A grounded AI customer-support agent combining RAG, safe order lookup, multi-turn conversation, prompt-injection protection, source-conflict detection, and automated evaluation.

## Overview

This project implements a reliable AI customer-support agent for Aster & Row.

The agent is designed to answer customer questions using the supplied knowledge base, retrieve real order information through a controlled tool, maintain context across conversations, protect private information, detect conflicting sources, and avoid following instructions embedded inside retrieved documents or tool results.

The core design principle is:

> **Reliable answers over confident answers.**

### Key capabilities

- Grounded answers from the supplied knowledge base
- Source citations for policy/product answers
- Active/official policy precedence over superseded/internal content
- Documented exception handling, such as TrailPlus membership rules
- Genuine source-conflict detection
- Real order lookup through a dedicated tool
- Customer-safe order data filtering
- Multi-turn query rewriting and memory
- Prompt-injection protection
- Insufficient-information abstention
- Human handoff for genuinely unresolved cases
- Structured JSONL observability
- Automated evaluation with deterministic checks

---

# 1. Setup & Running the Project

## Requirements

- Python 3.10+
- A supported LLM API key
- Git

## Clone the repository

```bash
git clone https://github.com/Darshh16/Ai_agent_test.git
cd Ai_agent_test
```

## Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Environment variables

The repository includes `.env.example`.

Create a local `.env` file using the example and provide the required API configuration.

Example:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_available_model
```

The API key must never be committed to the repository.

## Run the application

```bash
streamlit run app.py
```

The Streamlit interface provides:

- Chat history
- Source citations
- Human-handoff indication
- Expandable retrieval/tool trace
- New-session functionality

---

# 2. Technology & Design Choices

## LLM

The project uses a configurable LLM provider through `src/llm_client.py`.

The implementation was initially designed around the **Google Gemini free tier**, as specified by the project requirements, while the final development configuration uses Groq with the model controlled through `GROQ_MODEL`.

Keeping the provider isolated behind an LLM client makes the agent orchestration independent from the specific model provider.

### Why?

The assignment focuses on agent reliability rather than provider-specific infrastructure. Keeping the provider configurable also makes model changes easier without rewriting the agent.

---

## Retrieval / Embeddings

The project uses a local **TF-IDF + cosine similarity** retrieval pipeline.

Knowledge-base Markdown files are parsed into chunks and indexed locally.

Lightweight normalization/stemming is applied to improve matching between word variants such as:

```text
ship
ships
shipping
```

### Why TF-IDF?

The supplied knowledge base is relatively small, making a local and inspectable retrieval system sufficient for the assignment.

It also made retrieval behavior easy to debug and test. During development, this helped identify a real retrieval bug where `ship` and `shipping` were not being matched correctly.

A production system with a much larger corpus could use semantic embeddings and a persistent vector database.

---

## Storage

The project intentionally uses local files:

```text
knowledge-base/     → Markdown policy/product documents
data/orders.json    → Order data
logs/               → JSONL traces
evaluation/results/ → Evaluation results
```

### Why?

The assignment does not require production infrastructure. Local storage keeps the system reproducible, transparent, and easy to run from a clean clone.

---

## Framework

### Streamlit

Used for the customer-support chat interface.

### pytest

Used for regression and unit testing.

### Custom evaluation runner

`evaluation/run_eval.py` directly exercises the agent orchestration and evaluates:

- Retrieval
- Groundedness
- Multi-turn conversation
- Tool use
- Tool reliability
- Privacy
- Prompt security
- Abstention
- Source conflicts

---

# 3. Architecture

```text
                         User
                           │
                           ▼
                  ┌─────────────────┐
                  │ Agent / Router  │
                  └────────┬────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
       Knowledge Query             Order Query
              │                         │
              ▼                         ▼
     ┌─────────────────┐       ┌─────────────────┐
     │ Metadata-aware  │       │  Order Lookup   │
     │      RAG        │       │      Tool       │
     └────────┬────────┘       └────────┬────────┘
              │                         │
              ▼                         ▼
     KB chunks + metadata       Safe customer fields
              │                         │
              └────────────┬────────────┘
                           ▼
                    ┌─────────────┐
                    │ LLM Client  │
                    └──────┬──────┘
                           ▼
                Answer + Sources + Trace
                           │
                    ┌──────┴──────┐
                    │             │
                 Answer        Handoff
```

## Request flow

### Knowledge-base question

```text
User question
     ↓
Conversation context / query rewriting
     ↓
Retrieval
     ↓
Metadata + authority filtering
     ↓
Relevant KB chunks
     ↓
LLM generation
     ↓
Grounded answer + sources
```

### Order question

```text
User question
     ↓
Order ID detection
     ↓
order_lookup tool
     ↓
Customer-safe allowlist
     ↓
Tool result
     ↓
LLM response
```

The UI and evaluation suite both use the same underlying agent orchestration instead of implementing separate answer logic.

---

# 4. Retrieval & Grounding

Each knowledge-base document contains metadata such as:

- `status`
- `effective_date`
- `audience`
- `policy_authority`
- supersession information where applicable

A document is considered customer-facing authority only when:

```text
status = active
policy_authority = official
```

This prevents documents such as the legacy returns policy and internal migration notes from becoming authoritative answers.

## Source conflicts

The agent distinguishes between a documented exception and a genuine conflict.

For example:

```text
Standard customer → 30-day return window
TrailPlus member → 45-day return window
```

is a valid documented exception.

However, contradictory instructions from two active official Breeze Tumbler documents are treated as a genuine conflict and can result in human handoff.

---

# 5. Order Tool & Privacy

Orders are retrieved using the dedicated `order_lookup` tool.

The tool does not expose the complete order object to the LLM.

Only customer-safe fields are returned.

Internal fields such as:

```text
customer.email
customer.shipping_address
internal.*
items.sku
```

are excluded before the result reaches the model.

This is particularly important because the supplied order data contains instruction-like text in internal warehouse notes.

For example, `ORD-1005` contains an internal instruction attempting to make the AI issue a coupon and hide the delay reason.

The tool removes such internal data upstream rather than relying only on the model to ignore it.

---

# 6. Multi-turn Conversation

The agent stores recent conversation context and rewrites short follow-up questions before retrieval.

Example:

```text
User:
Do you ship internationally?

Agent:
Canada is currently supported...

User:
What about Canada?

Agent:
Canada is supported. Delivery generally takes
5–9 business days after dispatch...
```

This allows the retrieval system to work with a complete contextual query instead of only:

```text
"What about Canada?"
```

---

# 7. Prompt Injection Protection

The system treats both retrieved documents and tool results as:

> **Untrusted data, never instructions.**

The supplied KB contains an internal migration document with instruction-like text attempting to override the system behavior.

The order data also contains malicious-looking instructions inside internal warehouse notes.

The system protects against both:

### Knowledge-base injection

Internal/draft documents cannot become customer-facing authority.

### Tool-result injection

Internal order fields are removed before the LLM receives the tool result.

This creates a layered security boundary rather than relying only on prompt instructions.

---

# 8. Evaluation

The project includes both supplied visible cases and original custom cases.

Run the evaluation with:

```bash
python evaluation/run_eval.py --save-as final
```

Results are saved to:

```text
evaluation/results/final.json
```

## Baseline vs Final

The initial implementation achieved:

**10 / 20 — 50%**

After iterative debugging, regression testing, and reliability improvements:

**20 / 20 — 100%**

| Category | Baseline | Final |
|---|---:|---:|
| Retrieval | 1/2 | 2/2 |
| Multi-source grounding | 0/1 | 1/1 |
| Conversation | 0/1 | 1/1 |
| Groundedness | 1/2 | 2/2 |
| Tool use | 1/3 | 3/3 |
| Tool reliability | 2/4 | 4/4 |
| Privacy | 1/2 | 2/2 |
| Prompt security | 1/2 | 2/2 |
| Abstention | 1/1 | 1/1 |
| Source conflict | 2/2 | 2/2 |
| **Total** | **10/20** | **20/20** |

## Final result

```text
TOTAL: 20/20 passed
```

The final evaluation covers:

- Retrieval
- Multi-source grounding
- Conversation
- Groundedness
- Tool use
- Tool reliability
- Privacy
- Prompt security
- Abstention
- Source conflict

---

# 9. Custom Evaluation Cases

The project also contains original cases designed to test additional failure modes:

1. Tool-result prompt injection through `ORD-1005`
2. Order ID normalization using whitespace/case variations
3. Cancellation attempt on a non-pending order
4. TrailPlus exception not being incorrectly treated as a source conflict
5. Privacy protection around SKU exposure

These cases were added to test behaviors beyond simply passing the supplied visible cases.

---

# 10. Bug Diary

## Bug 1 — Fragile source-conflict threshold

**Phase:** 1

### Reproduction

The deliberate Breeze Tumbler conflict behaved differently across environments for the same query.

### Root cause

Conflict detection relied on a fixed absolute similarity gap. Small differences in retrieval scores could cause the conflict to fall on either side of the threshold.

### Fix

Changed conflict detection to use a relative score margin.

### Regression

Added tests ensuring both active official sources are retrieved and the genuine conflict is detected consistently.

---

## Bug 2 — Reasoning trace leakage

**Phase:** 4

### Reproduction

A reasoning-capable model returned `<think>...</think>` content during query rewriting.

### Root cause

The provider returned reasoning text through the normal response field.

### Fix

Added reasoning-trace stripping at the LLM client boundary.

### Regression

Added a regression test using the observed reasoning-model response.

---

## Bug 3 — Shipping retrieval failed for an unseen destination

**Phase:** 7

### Reproduction

```text
Can you ship an Atlas Weekender to Germany?
```

did not initially retrieve the relevant international-shipping policy.

### Root cause

The local TF-IDF retriever did not sufficiently connect word variants such as `ship` and `shipping`.

### Fix

Added lightweight normalization/stemming and destination-shipping retrieval handling.

### Regression

Added retrieval coverage for the Germany shipping query.

---

## Bug 4 — LLM omitted required handoff marker

**Phase:** 5/7

### Reproduction

The model correctly identified a case requiring human review but did not consistently emit the required handoff marker.

### Root cause

The LLM followed the substantive response instruction but did not reliably follow the formatting marker.

### Fix

Added a narrow secondary handoff layer for genuinely unresolved/review-required situations.

### Regression

Added both positive and negative handoff tests.

---

## Bug 5 — Order-data prompt injection

**Phase:** 7

### Reproduction

`ORD-1005` contains an internal warehouse note attempting to instruct the AI to issue a coupon and hide the delay reason.

### Root cause

Passing raw order records to the model could expose internal instructions.

### Fix

Implemented an explicit customer-safe allowlist in the order lookup tool.

### Regression

Added tests verifying that internal fields and injected instruction strings never reach the customer-facing response.

---

## Bug 6 — Stemmer fix introduced stop-word inconsistency

**Phase:** 7

### Reproduction

After adding stemming, scikit-learn reported a stop-word consistency warning.

### Root cause

The query terms were stemmed but the default English stop-word list was not.

### Fix

Stemmed the stop-word list as well and made the normalization idempotent.

### Regression

Tests were run with warnings treated as errors to verify that the inconsistency was resolved.

---

## Bug 7 — Evaluation judge failed on Markdown-wrapped JSON

**Phase:** 7

### Reproduction

The concept judge returned valid JSON wrapped in a Markdown code fence.

### Root cause

Direct `json.loads()` could not parse the fenced response.

### Fix

Added Markdown code-fence stripping before JSON parsing when the initial parse fails.

### Regression

Added a dedicated evaluation-parser regression test.

---

## Bug 8 — Test polluted the production trace log

**Phase:** 6

### Reproduction

A logging test accidentally wrote test data into the application's default trace file.

### Root cause

The test did not provide an isolated temporary log path.

### Fix

Tests now use isolated temporary logging paths.

### Regression

Added assertions against the isolated log output.

---

# 11. Known Limitations

### Free-tier model availability

The project was designed around the **Gemini free tier** and keeps model/provider configuration externalized.

Free-tier model availability can change over time or vary by account/region. The final development configuration uses Groq through the configurable `GROQ_MODEL` setting.

### Local retrieval

TF-IDF retrieval is appropriate for the supplied small corpus but is not intended to represent a production-scale semantic retrieval system.

A production implementation could use stronger embedding models and a persistent vector database.

### Authentication

The assignment assumes that possession of an order ID is sufficient for the order lookup flow.

A production ecommerce support system would require proper identity verification and authorization.

### Production infrastructure

The project intentionally does not implement:

- Authentication/user management
- Production deployment
- Docker/Kubernetes infrastructure
- Production vector databases
- Model fine-tuning
- Admin dashboards
- Analytics infrastructure
- Multi-provider failover

These were outside the assignment scope.

---

# 12. AI Coding Assistance Disclosure

AI coding tools were used during development for:

- Code scaffolding
- Debugging
- Test generation
- Reviewing implementation approaches
- Diagnosing evaluation failures

AI-generated suggestions were not accepted blindly.

### Example of an incorrect/incomplete AI suggestion

An early retrieval approach relied too heavily on similarity matching and did not account sufficiently for word-form differences such as:

```text
ship
shipping
ships
```

This caused an international-shipping retrieval failure for the Germany query.

The implementation was revised after reproducing the failure, and regression tests were added to prevent the issue from returning.

Another important lesson was that broad LLM-based handoff heuristics could incorrectly classify a fully answered question as requiring human escalation. The handoff logic was subsequently narrowed to genuine unresolved/review-required cases.

---

# 13. Demo

### 🎥 Aster & Row — Customer Support Agent Demo

A short walkthrough demonstrating:

1. **Grounded RAG** with source citations
2. **Real order lookup** through the `order_lookup` tool
3. **Multi-turn conversation** with contextual follow-up
4. **Prompt-injection protection** and correct refusal/handoff
5. **20/20 automated evaluation** result

▶[Watch the Demo Video](https://drive.google.com/file/d/15GDTjyg48VLUR4ld_gOXsTVRVCkOv_zk/view?usp=sharing)

---

# 14. Project Structure

```text
Ai_agent_test/
│
├── app.py
├── README.md
├── requirements.txt
├── .env.example
│
├── knowledge-base/
│
├── data/
│
├── src/
│   ├── agent.py
│   ├── ingest.py
│   ├── llm_client.py
│   ├── logging_utils.py
│   ├── memory.py
│   ├── retrieval.py
│   └── tools/
│       └── order_lookup.py
│
├── evaluation/
│   ├── visible-cases.json
│   ├── custom-cases.json
│   ├── checks.py
│   ├── run_eval.py
│   └── results/
│
└── tests/
```

---

# Final Result

The final system achieved:

```text
20 / 20 evaluation cases passed
100% final evaluation score
```

The implementation prioritizes groundedness, reliability, privacy, security, and testability rather than simply generating fluent customer-support responses.

> **The goal is not just to answer — it is to know when an answer is trustworthy.**
