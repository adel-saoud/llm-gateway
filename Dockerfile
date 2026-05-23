# ── Builder ─────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock* README.md ./
COPY src ./src

# Install only runtime deps (no dev)
RUN uv sync --no-dev --no-editable

# ── Runtime ─────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Non-root user
RUN groupadd --system --gid 1001 gateway && \
    useradd --system --uid 1001 --gid gateway --no-create-home --shell /usr/sbin/nologin gateway

WORKDIR /app
COPY --from=builder --chown=gateway:gateway /app/.venv /app/.venv
COPY --chown=gateway:gateway src ./src

# Data directory for SQLite — mounted as a volume in docker-compose
RUN mkdir -p /app/data && chown gateway:gateway /app/data

USER gateway

EXPOSE 8000

CMD ["uvicorn", "llm_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
