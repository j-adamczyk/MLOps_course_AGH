from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.scripts.settings import Settings


def prepare_model_from_hf(settings: Settings) -> None:
    target_dir = Path(settings.local_model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {settings.hf_model_id} into {target_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(settings.hf_model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(settings.hf_model_id)

    tokenizer.save_pretrained(str(target_dir))
    model.save_pretrained(str(target_dir))
    print(f"Saved tokenizer and model to {target_dir}")


if __name__ == "__main__":
    prepare_model_from_hf(Settings())
