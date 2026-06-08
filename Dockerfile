# STAGE 1: Dependency Pre-builder
FROM python:3.10-slim AS builder
WORKDIR /app

# Install compilation tools needed for C-extensions (like greenlet for SQLAlchemy or parts of XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip and build wheels for everything including nested sub-dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# STAGE 2: Secure Production Container 
FROM python:3.10-slim
WORKDIR /app

# Install native system graphics and utilities required for computer vision (OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root security group user
RUN groupadd -g 999 appuser && useradd -r -u 999 -g appuser appuser

# Copy built wheels from stage 1
COPY --from=builder /app/wheels /images/wheels
COPY --from=builder /app/requirements.txt .

# Install using the local compiled wheels pool cleanly
RUN pip install --no-cache-dir --no-index --find-links=/images/wheels -r requirements.txt \
    && rm -rf /images/wheels

# Copy code and your saved model files (.joblib weights)
COPY . .

# Set dynamic local folder write access permissions for SQLAlchemy/SQLite stability
RUN mkdir -p static/uploads && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
