"""
classify.py
-----------
Routing/decision logic for the hybrid log classifier.

Routing rules (see resources/arch.png):
    - source == "LegacyCRM"       -> RAG + LLM
    - else: Regex first
            -> if no match: BERT + LogisticRegression
            -> if BERT says "Unclassified" AND the user has uploaded their
               own company logs (rag_store), give RAG + LLM a shot too,
               since it can use real examples the BERT model never saw.

Every classifier now returns (label, confidence, explanation). classify_log
/classify propagate all three so callers (UI, API, agent) can show how
sure the system was AND why, not just the final label.
"""

import pandas as pd

from classifiers import classify_with_regex, classify_with_bert, classify_with_llm


def classify_log(source: str, log_message: str, api_key: str = None, rag_store=None) -> tuple[str, float, str]:
    """Return (label, confidence, explanation) for a single log message."""
    if source == "LegacyCRM":
        return classify_with_llm(log_message, api_key=api_key, rag_store=rag_store)

    label, confidence, explanation = classify_with_regex(log_message)
    if not label:
        label, confidence, explanation = classify_with_bert(log_message)

    if label == "Unclassified" and rag_store is not None and rag_store.is_ready():
        label, confidence, explanation = classify_with_llm(log_message, api_key=api_key, rag_store=rag_store)

    return label, confidence, explanation


def classify(logs: list[tuple[str, str]], api_key: str = None, rag_store=None) -> list[tuple[str, float, str]]:
    """logs: list of (source, log_message) tuples.
    Returns a list of (label, confidence, explanation) triples, same order as input."""
    return [classify_log(source, msg, api_key=api_key, rag_store=rag_store) for source, msg in logs]


def classify_csv(input_file: str, output_file: str = "resources/output.csv", api_key: str = None, rag_store=None) -> str:
    df = pd.read_csv(input_file)

    if "source" not in df.columns or "log_message" not in df.columns:
        raise ValueError("CSV must contain 'source' and 'log_message' columns.")

    results = classify(list(zip(df["source"], df["log_message"])), api_key=api_key, rag_store=rag_store)
    df["target_label"] = [label for label, _, _ in results]
    df["confidence"] = [round(conf, 3) for _, conf, _ in results]
    df["explanation"] = [expl for _, _, expl in results]
    df.to_csv(output_file, index=False)
    return output_file


if __name__ == "__main__":
    out = classify_csv("resources/test.csv")
    print(f"Saved results to {out}")
