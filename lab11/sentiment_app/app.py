import os
import json
import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
import onnxruntime as ort
from fastapi import FastAPI, Request
from mangum import Mangum
from tokenizers import Tokenizer

from sentiment_app.models import SentimentRequest, SentimentResponse



BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ONNX_PATH = BASE_DIR / "model/onnx/zeroshot.onnx"
DEFAULT_TOKENIZER_PATH = BASE_DIR / "model/onnx/tokenizer.json"

DEFAULT_LABELS = ["positive", "negative", "neutral"]
ENTAILMENT_IDX = 0


class ZeroShotSentimentClassifier:
    def __init__(self, onnx_path: str, tokenizer_path: str):
        self.session = ort.InferenceSession(onnx_path)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def predict(self, text: str, labels: list[str]) -> str:
        scores = []
        for label in labels:
            hypothesis = f"This text expresses {label} sentiment."
            encoded = self.tokenizer.encode(text, hypothesis, add_special_tokens=True)
            input_ids = np.array([encoded.ids], dtype=np.int64)
            attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
            ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}   
            logits = self.session.run(None, ort_inputs)[0]
            scores.append(float(logits[0, ENTAILMENT_IDX]))
        return labels[int(np.argmax(scores))]


@asynccontextmanager
async def lifespan(app: FastAPI):
    onnx_path = os.environ.get("ONNX_PATH", str(DEFAULT_ONNX_PATH))
    tokenizer_path = os.environ.get("TOKENIZER_PATH", str(DEFAULT_TOKENIZER_PATH))
    app.state.classifier = ZeroShotSentimentClassifier(onnx_path=onnx_path, tokenizer_path=tokenizer_path)
    yield
    app.state.classifier = None


app = FastAPI(lifespan=lifespan)


@app.post("/predict", response_model=SentimentResponse)
def predict(request: Request, payload: SentimentRequest) -> SentimentResponse:
    labels = payload.labels or DEFAULT_LABELS
    label = request.app.state.classifier.predict(payload.text, labels)
    return SentimentResponse(label=label)


handler = Mangum(app, lifespan="on") if os.environ.get("AWS_LAMBDA_RUNTIME_API") else None
