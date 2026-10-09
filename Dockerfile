FROM python:3.12.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY api/ ./api/
COPY assistant_api/ ./assistant_api/
COPY app/ ./app/

RUN python -m pip install --no-cache-dir --no-compile -e .

USER 10001:10001

EXPOSE 8000 8001 8501
