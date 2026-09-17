# LogFlow

A cost-aware log classification system that routes each log through the cheapest method capable of handling it — regex, then an embedding-based ML model, then a retrieval-augmented LLM — before escalating to expensive calls. Every prediction ships with a confidence score and a plain-English explanation. Built on top: a tool-calling Q&A agent, an autonomous investigator, a chatbot, a REST API, and Slack automation via n8n.

## How it works

<img width="1779" height="772" alt="image" src="https://github.com/user-attachments/assets/145c2025-d45f-4996-a329-f1bedfac4be2" />


<img width="1661" height="592" alt="image" src="https://github.com/user-attachments/assets/151abfbf-026a-4288-935c-f720cd0b20e6" />


```

**The routing logic, in short:**
- `LegacyCRM` logs always go straight to RAG + LLM — there's no clean pattern or trained model for this source.
- Everything else tries regex first (free, instant), then BERT + Logistic Regression (cheap, fast).
- Only genuinely ambiguous logs — where BERT itself says "Unclassified" — escalate to the LLM, and only if a knowledge base (built-in or user-supplied) is available for retrieval.

This keeps LLM API usage limited to the minority of logs that actually need it.

## Features

**Classification**
- Three-tier pipeline: regex → BERT + Logistic Regression → RAG + LLM
- Every prediction includes a confidence score (0–1) and a human-readable explanation of *why* that label was chosen
- In-memory response caching and retry-with-backoff for LLM calls

**RAG knowledge base**
- Retrieves the 3 most similar past labeled examples via embedding cosine similarity, feeds them to the LLM as few-shot context
- Built-in default knowledge base, or upload your own CSV
- Manual corrections in the UI get embedded and appended back into the knowledge base — the system improves the more you use it, with no retraining required

**Interfaces**
- **Streamlit app** — five tabs: batch CSV classification (with an inline correction editor and label-distribution chart), single-message classification, a tool-calling Q&A agent, an autonomous investigator, and a free-form chatbot
- **FastAPI REST API** — endpoints for batch classification, single-message classification, knowledge base management, corrections, the agent, the chatbot, and the investigator
- **n8n workflow** — watches for new log files, calls the API, and posts a Slack alert if anything comes back critical or unclassified

**Agents**
- **Q&A agent** — ask questions in plain English about a classified batch; it decides which tools to call (count by label, count by source, search, pull unclassified rows)
- **Investigator** — one click, no question needed. Finds every critical/unclassified log, investigates each with RAG-retrieved context, and compiles a Markdown report. Built with LangGraph as a small state machine (find critical → investigate → compile report)

## Project structure

```
.
├── app.py                 # Streamlit frontend
├── server.py               # FastAPI REST API
├── classify.py              # Routing/decision logic
├── classifiers.py            # Regex, BERT+LogReg, RAG+LLM implementations
├── rag_store.py              # In-memory embedding store for custom knowledge bases
├── agent.py                # Tool-calling Q&A agent
├── chatbot.py               # Free-form conversational assistant
├── investigator.py            # LangGraph autonomous investigator
├── test_classifier.py          # Unit tests
├── n8n_workflow.json           # Importable n8n automation workflow
├── requirements.txt            # Python dependencies
├── .env.example              # Environment variable template
├── models/
│   └── log_classifier.joblib     # Trained BERT+LogReg classifier (not included — train your own)
└── resources/
    └── knowledge_base.csv        # Default RAG knowledge base
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your API key

Copy `.env.example` to `.env` and add your Groq API key:

```bash
cp .env.example .env
```

```
GROQ_API_KEY=your_groq_api_key_here
```

You can also enter the key directly in the Streamlit sidebar, or pass it as an `X-Groq-Api-Key` header when calling the API — either takes priority over the `.env` value.

### 3. Provide the required data files

Two files are **not included** in this repo and must be supplied before the app runs end to end:

| File | Purpose | How to get it |
|---|---|---|
| `models/log_classifier.joblib` | Trained BERT+LogReg classifier used for non-regex, non-LegacyCRM logs | Train on your own labeled log data with `sentence-transformers` embeddings + `sklearn.LogisticRegression`, then `joblib.dump()` |
| `resources/knowledge_base.csv` | Default example set for RAG retrieval | Needs `log_message` and `target_label` columns — a starter set is included in this README's companion files |

Without these, regex-only logs classify fine, but BERT-fallback and `LegacyCRM` logs will error until the files are in place.

### 4. Run the app

**Streamlit UI:**
```bash
streamlit run app.py
```

**REST API:**
```bash
uvicorn server:app --reload
```

The API defaults to `http://localhost:8000`. If you're calling it from a browser-based frontend on a different origin, make sure `CORSMiddleware` is enabled in `server.py`.

## Using the RAG feature

1. Upload a CSV with `log_message` and `target_label` columns via the sidebar ("Build knowledge base") — or start from the built-in `resources/knowledge_base.csv`.
2. Classify logs as normal. `LegacyCRM` sources and low-confidence BERT results will retrieve similar examples from this knowledge base and use them as context for the LLM.
3. When a prediction is wrong, correct it in the results table and click "Save correction." That correction is embedded and appended to the knowledge base — future similar logs benefit immediately.
4. Check what's currently loaded via the sidebar caption, or `GET /knowledge-base/status` on the API.

## Running tests

```bash
pytest test_classifier.py -v
```

Covers regex matching, routing logic (with BERT/LLM mocked out), LLM response caching, retry/backoff behavior, and the RAG store's correction-append logic.


## Tech stack

Python · Streamlit · FastAPI · LangChain / LangGraph · Groq (Llama 3.3) · sentence-transformers · scikit-learn · pandas 

