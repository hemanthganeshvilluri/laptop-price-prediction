# STAGE 1: Dependency Builder
FROM python:3.10-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# STAGE 2: Final Production Runtime
FROM python:3.10-slim
WORKDIR /app
RUN groupadd -g 999 appuser && useradd -r -u 999 -g appuser appuser
COPY --from=builder /app/wheels /images/wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/images/wheels -r requirements.txt && rm -rf /images/wheels

# Copy code and your notebook's exact saved joblib models
COPY app.py .
COPY model_low.joblib .
COPY model_med.joblib .
COPY model_high.joblib .

RUN chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]