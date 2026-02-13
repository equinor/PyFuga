ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# Keep Python output unbuffered and avoid writing .pyc
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (include doc tools so docs/jobs don't apt-get every time)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      make \
      pandoc \
    && rm -rf /var/lib/apt/lists/*

# Install uv once in the image
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir uv

# Helpful default
WORKDIR /work
