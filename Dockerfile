# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

COPY pyproject.toml ./
COPY src ./src

RUN pip install --upgrade pip \
 && pip install --prefix=/install .

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

# Non-root user
RUN groupadd --gid 1001 appgroup \
 && useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy only installed packages from builder
COPY --from=builder /install /usr/local

# Copy source (no build tools, no pip cache)
COPY src ./src

# Workspace dir owned by appuser (agent read_file tool mounts here)
RUN mkdir /workspace && chown appuser:appgroup /workspace

USER appuser

CMD ["python", "-m", "agentforge_worker"]
