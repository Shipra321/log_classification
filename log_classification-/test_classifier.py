"""
test_classifier.py
-------------------
Basic unit tests for the classification pipeline. Run with:

    pytest test_classifier.py -v

These focus on logic that doesn't require network calls or a trained
BERT model on disk:
  - regex pattern matching + confidence + explanation
  - routing rules in classify_log (mocking the BERT/LLM layers)
  - the LLM cache (hits/misses, no duplicate API calls)
  - retry/backoff behavior on transient Groq failures
  - RAGStore.append_examples (used by the manual-correction UI feature)
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

import classifiers
from classifiers import (
    classify_with_regex,
    _call_groq_with_retry,
    _cache_key,
    clear_llm_cache,
)
import classify as classify_module
from rag_store import RAGStore


# ---------------------------------------------------------------------------
# Regex classifier
# ---------------------------------------------------------------------------

class TestRegexClassifier:
    def test_login_matches_user_action(self):
        label, confidence, explanation = classify_with_regex("User User123 logged in.")
        assert label == "User Action"
        assert confidence == 1.0
        assert "Matched rule pattern" in explanation

    def test_backup_matches_system_notification(self):
        label, confidence, explanation = classify_with_regex("Backup completed successfully.")
        assert label == "System Notification"
        assert confidence == 1.0

    def test_unmatched_message_returns_none(self):
        label, confidence, explanation = classify_with_regex("Something totally unexpected happened.")
        assert label is None
        assert confidence == 0.0
        assert "No regex rule matched" in explanation

    def test_account_created_matches_user_action(self):
        label, confidence, explanation = classify_with_regex("Account with ID 991 created by admin.")
        assert label == "User Action"
        assert confidence == 1.0


# ---------------------------------------------------------------------------
# Routing logic in classify_log (mock out BERT + LLM so no model/API needed)
# ---------------------------------------------------------------------------

class TestRoutingLogic:
    def test_legacy_crm_always_routes_to_llm(self):
        with patch.object(classify_module, "classify_with_llm", return_value=("Workflow Error", 0.8, "explained")) as mock_llm:
            label, confidence, explanation = classify_module.classify_log("LegacyCRM", "Sync failed.")
        mock_llm.assert_called_once()
        assert label == "Workflow Error"
        assert confidence == 0.8
        assert explanation == "explained"

    def test_regex_hit_skips_bert_and_llm(self):
        with patch.object(classify_module, "classify_with_bert") as mock_bert, \
             patch.object(classify_module, "classify_with_llm") as mock_llm:
            label, confidence, explanation = classify_module.classify_log("ModernCRM", "User User1 logged in.")
        mock_bert.assert_not_called()
        mock_llm.assert_not_called()
        assert label == "User Action"
        assert confidence == 1.0

    def test_regex_miss_falls_back_to_bert(self):
        with patch.object(classify_module, "classify_with_bert", return_value=("Security Alert", 0.7, "bert says so")) as mock_bert:
            label, confidence, explanation = classify_module.classify_log("ModernCRM", "Totally novel message.")
        mock_bert.assert_called_once()
        assert label == "Security Alert"
        assert confidence == 0.7
        assert explanation == "bert says so"

    def test_bert_unclassified_without_rag_store_stays_unclassified(self):
        with patch.object(classify_module, "classify_with_bert", return_value=("Unclassified", 0.2, "unsure")), \
             patch.object(classify_module, "classify_with_llm") as mock_llm:
            label, confidence, explanation = classify_module.classify_log("ModernCRM", "Weird message.", rag_store=None)
        mock_llm.assert_not_called()
        assert label == "Unclassified"

    def test_bert_unclassified_with_ready_rag_store_escalates_to_llm(self):
        fake_store = MagicMock()
        fake_store.is_ready.return_value = True
        with patch.object(classify_module, "classify_with_bert", return_value=("Unclassified", 0.2, "unsure")), \
             patch.object(classify_module, "classify_with_llm", return_value=("Deprecation Warning", 0.8, "rag says so")) as mock_llm:
            label, confidence, explanation = classify_module.classify_log("ModernCRM", "Weird message.", rag_store=fake_store)
        mock_llm.assert_called_once()
        assert label == "Deprecation Warning"
        assert explanation == "rag says so"


# ---------------------------------------------------------------------------
# LLM cache
# ---------------------------------------------------------------------------

class TestLLMCache:
    def setup_method(self):
        clear_llm_cache()

    def test_cache_key_is_deterministic(self):
        key1 = _cache_key("hello", "examples")
        key2 = _cache_key("hello", "examples")
        assert key1 == key2

    def test_cache_key_differs_for_different_input(self):
        key1 = _cache_key("hello", "examples")
        key2 = _cache_key("goodbye", "examples")
        assert key1 != key2

    def test_repeated_call_uses_cache_not_api(self):
        fake_response = MagicMock()
        fake_response.choices[0].message.content = "<category>Workflow Error</category>"

        with patch.object(classifiers, "_get_groq_client", return_value=MagicMock()), \
             patch.object(classifiers, "_call_groq_with_retry", return_value=fake_response) as mock_call, \
             patch.object(classifiers, "retrieve_similar_examples", return_value=[]):

            label1, conf1, expl1 = classifiers.classify_with_llm("Sync failed.", api_key="fake-key")
            label2, conf2, expl2 = classifiers.classify_with_llm("Sync failed.", api_key="fake-key")

        # Second call should be served from cache -> API hit only once
        mock_call.assert_called_once()
        assert label1 == label2 == "Workflow Error"
        assert expl1 == expl2


# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------

class TestRetryBackoff:
    def test_succeeds_on_first_try_without_retry(self):
        client = MagicMock()
        client.chat.completions.create.return_value = "ok"

        result = _call_groq_with_retry(client, "prompt", max_retries=3)

        assert result == "ok"
        assert client.chat.completions.create.call_count == 1

    def test_retries_then_succeeds(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [Exception("timeout"), Exception("timeout"), "ok"]

        with patch("classifiers.time.sleep"):  # skip real sleeping in tests
            result = _call_groq_with_retry(client, "prompt", max_retries=3)

        assert result == "ok"
        assert client.chat.completions.create.call_count == 3

    def test_raises_after_exhausting_retries(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("still failing")

        with patch("classifiers.time.sleep"):
            with pytest.raises(RuntimeError, match="failed after 3 attempts"):
                _call_groq_with_retry(client, "prompt", max_retries=3)

        assert client.chat.completions.create.call_count == 3


# ---------------------------------------------------------------------------
# RAGStore.append_examples (backs the "save correction" UI feature)
# ---------------------------------------------------------------------------

class TestRAGStoreAppend:
    def _fake_embedding_model(self):
        model = MagicMock()
        # deterministic fake embeddings: one row per input text, dim=4
        model.encode.side_effect = lambda texts, normalize_embeddings=True: np.array(
            [[len(t), 0, 0, 0] for t in texts], dtype=float
        )
        return model

    def test_append_to_empty_store(self):
        store = RAGStore(self._fake_embedding_model())
        store.append_examples(["hello world"], ["Greeting"])
        assert store.is_ready()
        assert store.texts == ["hello world"]
        assert store.labels == ["Greeting"]
        assert store.embeddings.shape == (1, 4)

    def test_append_to_existing_store_grows_it(self):
        store = RAGStore(self._fake_embedding_model())
        store.load_dataframe(__import__("pandas").DataFrame({
            "log_message": ["first msg"], "target_label": ["A"],
        }))
        store.append_examples(["second msg", "third msg"], ["B", "C"])

        assert store.texts == ["first msg", "second msg", "third msg"]
        assert store.labels == ["A", "B", "C"]
        assert store.embeddings.shape == (3, 4)

    def test_append_empty_list_is_noop(self):
        store = RAGStore(self._fake_embedding_model())
        store.append_examples([], [])
        assert not store.is_ready()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
