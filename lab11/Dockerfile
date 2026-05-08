# Dockerfile
# Production Lambda image: runs sentiment_app.app.handler under awslambdaric.
# Build stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Install dependencies (inference group only)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --group inference --no-install-project

# Runtime stage
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy virtual environment
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY sentiment_app ./sentiment_app

# Copy ONNX model + tokenizer files
COPY model ./model

# Lambda runtime entrypoint
ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["sentiment_app.app.handler"]
