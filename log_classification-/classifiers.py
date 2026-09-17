"""
classifiers.py
--------------
All classification engines used by the hybrid log classifier:

1. Regex          -> fast, rule-based, catches predictable patterns
2. BERT + LogReg  -> sentence-embedding + logistic-regression model for
                     patterns that are too varied for regex but have
                     enough labeled training data
3. RAG + LLM      -> used for sources with too little labeled data
                     (e.g. "LegacyCRM") or as a fallback when BERT is
                     unsure. Retrieves similar labeled examples - either
                     from the built-in knowledge base, or from a user's
                     own uploaded company logs (see rag_store.py) - and
                     gives them to the LLM as few-shot context.

Every classifier returns a (label, confidence, explanation) triple:
  - label:       the predicted category (str), or None (regex only) if
                  no rule matched
  - confidence:  float 0-1
  - explanation: short human-readable string describing *why* this
                  label was chosen (matched rule, embedding confidence,
                  or the RAG examples used as LLM context) - shown
                  directly in the UI so users aren't just trusting a
                  black box

Also includes an in-memory cache + retry/backoff wrapper around the
Groq LLM call, so repeated/near-identical messages don't re-hit the API
and transient failures don't crash the whole batch.
"""

import os
import re
import time
import random
import hashlib

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Regex classifier
# ---------------------------------------------------------------------------

REGEX_PATTERNS = {
    r"User User\d+ logged (in|out).": "User Action",
    r"Backup (started|ended) at .*": "System Notification",
    r"Backup completed successfully.": "System Notification",
    r"System updated to version .*": "System Notification",
    r"File .* uploaded successfully by user .*": "System Notification",
    r"Disk cleanup completed successfully.": "System Notification",
    r"System reboot initiated by user .*": "System Notification",
    r"Account with ID .* created by .*": "User Action",
}


def classify_with_regex(log_message: str):
    """Return (label, confidence, explanation) if the message matches a
    known pattern, else (None, 0.0, explanation). Regex matches are exact
    rule hits, so confidence is always 1.0 when a pattern fires."""
    for pattern, label in REGEX_PATTERNS.items():
        if re.search(pattern, log_message):
            explanation = f"Matched rule pattern `{pattern}` → deterministic, no model involved."
            return label, 1.0, explanation
    return None, 0.0, "No regex rule matched; falling through to the embedding model."


# ---------------------------------------------------------------------------
# 2. BERT (sentence-embedding) + Logistic Regression classifier
# ---------------------------------------------------------------------------

_embedding_model = None
_classification_model = None

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "log_classifier.joblib")
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "resources", "knowledge_base.csv")


def get_embedding_model():
    """Lazy-load the shared sentence-embedding model. Cached as a module
    global, so it's only downloaded/loaded once per running process
    (Streamlit re-runs the script, but keeps the same process/globals)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _embedding_model


def _get_bert_classifier():
    global _classification_model
    if _classification_model is None:
        import joblib

        _classification_model = joblib.load(MODEL_PATH)
    return _classification_model


def classify_with_bert(log_message: str):
    """Return (label, confidence, explanation). confidence is the model's
    max predicted probability across classes."""
    embedding_model = get_embedding_model()
    classification_model = _get_bert_classifier()

    embeddings = embedding_model.encode([log_message])
    probabilities = classification_model.predict_proba(embeddings)[0]
    confidence = float(max(probabilities))

    classes = list(classification_model.classes_)
    ranked = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    runner_up = f", runner-up: {ranked[1][0]} ({ranked[1][1]:.0%})" if len(ranked) > 1 else ""

    if confidence < 0.5:
        explanation = (
            f"Embedding similarity model was unsure (top probability only {confidence:.0%}"
            f"{runner_up}) → labeled Unclassified."
        )
        return "Unclassified", confidence, explanation

    explanation = f"Embedding similarity model, {confidence:.0%} confidence{runner_up}."
    return classification_model.predict(embeddings)[0], confidence, explanation


# ---------------------------------------------------------------------------
# 3. Built-in RAG knowledge base (fallback when the user hasn't uploaded
#    their own company logs)
# ---------------------------------------------------------------------------

_kb_texts = None
_kb_labels = None
_kb_embeddings = None


def _load_default_knowledge_base():
    global _kb_texts, _kb_labels, _kb_embeddings
    if _kb_embeddings is None:
        df = pd.read_csv(KNOWLEDGE_BASE_PATH)
        embedding_model = get_embedding_model()

        _kb_texts = df["log_message"].tolist()
        _kb_labels = df["target_label"].tolist()
        _kb_embeddings = embedding_model.encode(_kb_texts, normalize_embeddings=True)
    return _kb_texts, _kb_labels, _kb_embeddings


def retrieve_similar_examples(log_message: str, k: int = 3, rag_store=None):
    """Return top-k (log_message, label) examples similar to the given
    message. Uses the user's own uploaded RAGStore if provided and ready,
    otherwise falls back to the small built-in knowledge base."""
    if rag_store is not None and rag_store.is_ready():
        return rag_store.retrieve(log_message, k=k)

    texts, labels, kb_embeddings = _load_default_knowledge_base()
    if not texts:
        return []

    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode([log_message], normalize_embeddings=True)[0]
    similarities = kb_embeddings @ query_embedding
    top_k_idx = np.argsort(similarities)[::-1][:k]
    return [(texts[i], labels[i]) for i in top_k_idx]


# ---------------------------------------------------------------------------
# 4. RAG + LLM classifier (Groq) - with caching + retry/backoff + explanation
# ---------------------------------------------------------------------------


def _get_groq_client(api_key: str = None):
    """Build a Groq client using an explicitly-passed key, falling back to
    the GROQ_API_KEY environment variable / .env file if none is given."""
    from groq import Groq

    if api_key:
        return Groq(api_key=api_key)

    from dotenv import load_dotenv

    load_dotenv()
    env_key = os.environ.get("GROQ_API_KEY")
    if not env_key:
        raise ValueError(
            "No Groq API key provided. Pass one in, or set GROQ_API_KEY in your environment/.env file."
        )
    return Groq(api_key=env_key)


# --- simple in-memory cache for LLM classifications ------------------------
# Keyed on a hash of (log_message, examples used), since the result only
# depends on those two things. Caps at MAX_CACHE_SIZE to avoid unbounded
# growth in a long-running Streamlit/FastAPI process.

_llm_cache: dict[str, tuple[str, float, str]] = {}
MAX_CACHE_SIZE = 2000


def _cache_key(log_message: str, examples_block: str) -> str:
    raw = f"{log_message}||{examples_block}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def clear_llm_cache():
    """Exposed mainly for tests / manual cache-busting."""
    _llm_cache.clear()


def _call_groq_with_retry(client, prompt: str, max_retries: int = 3):
    """Call the Groq chat completion endpoint with exponential backoff +
    jitter on transient failures (rate limits, timeouts, 5xx)."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.5,
            )
        except Exception as e:  # groq raises various APIError subclasses
            last_error = e
            if attempt == max_retries - 1:
                break
            sleep_for = (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_for)
    raise RuntimeError(f"Groq API call failed after {max_retries} attempts: {last_error}")


def classify_with_llm(log_message: str, api_key: str = None, rag_store=None):
    """Return (label, confidence, explanation). Confidence is heuristic
    for the LLM path (parsed cleanly -> 0.8, fallback/unparsed -> 0.3),
    since the model doesn't give us a real probability. The explanation
    lists the retrieved examples that were used as few-shot context."""
    examples = retrieve_similar_examples(log_message, k=3, rag_store=rag_store)
    examples_block = "\n".join(
        f'- "{text}" -> {label}' for text, label in examples
    ) or "(no similar examples found)"

    cache_key = _cache_key(log_message, examples_block)
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    client = _get_groq_client(api_key)

    prompt = f"""Classify the log message into a category based on the labeled examples below,
    or use one of: Workflow Error, Deprecation Warning, if none of the examples fit well.
    If you truly can't tell, use "Unclassified".

    Similar previously-labeled log messages:
    {examples_block}

    Put the category inside <category></category> tags.
    Log message: {log_message}"""

    chat_completion = _call_groq_with_retry(client, prompt)

    content = chat_completion.choices[0].message.content
    match = re.search(r"<category>(.*)</category>", content, flags=re.DOTALL)

    example_lines = [f'  • "{t}" → {l}' for t, l in examples] or ["  • (no similar examples found)"]
    explanation_prefix = "RAG + LLM (Groq/Llama 3.3), using these retrieved examples as context:\n" + "\n".join(example_lines)

    if match:
        result = (match.group(1).strip(), 0.8, explanation_prefix)
    else:
        result = ("Unclassified", 0.3, explanation_prefix + "\n  Model output didn't parse cleanly.")

    if len(_llm_cache) >= MAX_CACHE_SIZE:
        _llm_cache.pop(next(iter(_llm_cache)))
    _llm_cache[cache_key] = result

    return result
