from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_sentiment_request_default_labels():
    from sentiment_app.app import DEFAULT_LABELS
    from sentiment_app.models import SentimentRequest

    req = SentimentRequest(text="I love this lab")
    assert req.labels is None
    assert DEFAULT_LABELS == ["positive", "negative", "neutral"]


def test_sentiment_request_custom_labels():
    from sentiment_app.models import SentimentRequest

    req = SentimentRequest(text="hello", labels=["spam", "ham"])
    assert req.labels == ["spam", "ham"]


def test_sentiment_response_shape():
    from sentiment_app.models import SentimentResponse

    resp = SentimentResponse(label="positive")
    assert resp.label == "positive"


def test_classifier_label_pick_logic():
    """Mock entailment scores and verify argmax mapping."""
    from sentiment_app.app import ZeroShotSentimentClassifier

    with patch.object(ZeroShotSentimentClassifier, "__init__", lambda self, *a, **kw: None):
        clf = ZeroShotSentimentClassifier(onnx_path="x", tokenizer_path="y")

    fake_session = MagicMock()
    fake_tokenizer = MagicMock()
    fake_tokenizer.encode.return_value = MagicMock(ids=[1, 2, 3], attention_mask=[1, 1, 1])

    # Entailment logit at index 0; we return increasing scores so the third label wins.
    fake_session.run.side_effect = [
        [np.array([[0.1, 0.0, 0.0]], dtype=np.float32)],
        [np.array([[0.5, 0.0, 0.0]], dtype=np.float32)],
        [np.array([[0.9, 0.0, 0.0]], dtype=np.float32)],
    ]

    clf.session = fake_session
    clf.tokenizer = fake_tokenizer

    label = clf.predict("text", labels=["a", "b", "c"])
    assert label == "c"
    assert fake_session.run.call_count == 3
    assert fake_tokenizer.encode.call_count == 3
