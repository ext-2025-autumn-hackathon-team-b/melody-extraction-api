FROM ghcr.io/astral-sh/uv:debian-slim
WORKDIR /app

RUN apt update && apt install -y ca-certificates build-essential fluidsynth

COPY pyproject.toml uv.lock /app/
RUN uv sync --locked

COPY . .

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
