from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    hf_model_id: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

    local_model_dir: str = "model/zeroshot"
    onnx_dir: str = "model/onnx"
    onnx_model_path: str = "model/onnx/zeroshot.onnx"
    tokenizer_path: str = "model/onnx/tokenizer.json"

    default_labels: list[str] = ["positive", "negative", "neutral"]
