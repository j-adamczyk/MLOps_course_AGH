from pydantic import BaseModel


class SentimentRequest(BaseModel):
    text: str
    labels: list[str] | None = None


class SentimentResponse(BaseModel):
    label: str
