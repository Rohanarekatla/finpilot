FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FINPILOT_SCHEDULER=1 \
    FINPILOT_DB=/data/finpilot.db

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
RUN mkdir -p /data && useradd --create-home finpilot && chown finpilot /data
USER finpilot

EXPOSE 8000
# One worker: the weekly digest scheduler runs inside the app process.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
