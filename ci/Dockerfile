ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

ENV DEBIAN_FRONTEND=noninteractive

# Keep Python output unbuffered and avoid writing .pyc
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (include doc tools so docs/jobs don't apt-get every time)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv once in the image
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir "uv==0.10.3"

# Helpful default
WORKDIR /work

FROM base AS tests

ARG CONSTRAINT_FILE

# Copy only the files needed to resolve and install test dependencies.
COPY pyproject.toml README.md ./
COPY pyfuga ./pyfuga
COPY constraints ./constraints

RUN test -n "$CONSTRAINT_FILE" \
    && test -f "$CONSTRAINT_FILE" \
    && uv pip install --system -c "$CONSTRAINT_FILE" ".[tests]"

WORKDIR /work
