"""
rag_store.py
------------
A tiny in-memory RAG store. Lets a user upload their own labeled
company logs (source, log_message, target_label) and retrieve the
most similar examples for a new log message via cosine similarity
over sentence embeddings.

append_examples() lets the UI feed corrected predictions straight
back in as new labeled examples, without requiring a full re-upload.
"""

import numpy as np
import pandas as pd


class RAGStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.texts = []
        self.labels = []
        self.embeddings = None

    def load_dataframe(self, df: pd.DataFrame):
        required = {"log_message", "target_label"}
        if not required.issubset(df.columns):
            raise ValueError("CSV must contain 'log_message' and 'target_label' columns.")

        self.texts = df["log_message"].astype(str).tolist()
        self.labels = df["target_label"].astype(str).tolist()
        self.embeddings = self.embedding_model.encode(self.texts, normalize_embeddings=True)

    def append_examples(self, texts: list[str], labels: list[str]):
        """Add new (text, label) pairs on top of whatever's already loaded -
        used to feed user-corrected predictions back into retrieval."""
        if not texts:
            return
        new_embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        self.texts.extend(texts)
        self.labels.extend(labels)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def is_ready(self) -> bool:
        return self.embeddings is not None and len(self.texts) > 0

    def retrieve(self, query: str, k: int = 3):
        if not self.is_ready():
            return []
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        similarities = self.embeddings @ query_embedding
        top_k_idx = np.argsort(similarities)[::-1][:k]
        return [(self.texts[i], self.labels[i]) for i in top_k_idx]
