import shutil
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.scripts.settings import Settings


def export_zeroshot_to_onnx(settings: Settings) -> Path:
    src_dir = Path(settings.local_model_dir)
    onnx_dir = Path(settings.onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(src_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(src_dir))
    model.eval()

    dummy_text = "This lab is great."
    dummy_hypothesis = "This text expresses positive sentiment."
    inputs = tokenizer(dummy_text, dummy_hypothesis, return_tensors="pt")

    onnx_path = Path(settings.onnx_model_path)
    print(f"Exporting model to {onnx_path}...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size"},
            },
            opset_version=18,
            dynamo=False,
        )

    # Copy the fast tokenizer file next to the ONNX model so the runtime can load it
    # via tokenizers.Tokenizer.from_file (no transformers dependency at inference time).
    src_tokenizer = src_dir / "tokenizer.json"
    if not src_tokenizer.exists():
        # Some tokenizers save under different names; force a fast save into the onnx_dir.
        tokenizer.save_pretrained(str(onnx_dir))
    else:
        shutil.copy2(src_tokenizer, Path(settings.tokenizer_path))

    print(f"ONNX model and tokenizer exported to {onnx_dir}")
    return onnx_path


if __name__ == "__main__":
    export_zeroshot_to_onnx(Settings())
