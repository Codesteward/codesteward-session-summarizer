# syntax=docker/dockerfile:1

FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files first for layer caching
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# Copy source code
COPY src/ src/

# Install the project itself
RUN uv sync --no-dev

# Build metadata — embedded as labels and available at runtime via env vars
ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown

ENV SUMMARIZER_BUILD_VERSION=${VERSION}
ENV SUMMARIZER_BUILD_COMMIT=${COMMIT}
ENV SUMMARIZER_BUILD_DATE=${BUILD_DATE}

LABEL org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.title="codesteward-session-summarizer" \
      org.opencontainers.image.description="Background service that summarizes dev sessions from ClickHouse audit events using an LLM"

CMD ["uv", "run", "python", "-m", "summarizer.main"]
