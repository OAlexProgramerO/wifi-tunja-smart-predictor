FROM python:3.12.14-slim-bookworm

ARG APP_VERSION=0.5.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

LABEL org.opencontainers.image.title="WiFi Tunja Smart Predictor" \
    org.opencontainers.image.version="${APP_VERSION}" \
    org.opencontainers.image.description="Synthetic-data WiFi demand prediction demonstration"

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/data/raw /app/data/processed /app/data/sample /app/data/interim \
        /app/models /app/reports/metrics /app/reports/figures \
    && chown -R app:app /app/data /app/models /app/reports

COPY requirements.lock ./
RUN python -m pip install --no-cache-dir --require-hashes -r requirements.lock

ENV PYTHONPATH=/app/src:/app \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/matplotlib

COPY src/ ./src/
COPY api/ ./api/
COPY assistant_api/ ./assistant_api/
COPY app/ ./app/
COPY scripts/ ./scripts/

RUN python -c "import api.main, assistant_api.main, app.dashboard"

USER 10001:10001

EXPOSE 8000 8001 8501
