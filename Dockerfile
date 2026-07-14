# Stage 1: Build dependency wheel and environment
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final runtime environment
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source code and configs
COPY src/ ./src/
COPY configs/ ./configs/
COPY notebooks/ ./notebooks/

# Expose default port for any downstream services (e.g. FastAPI / Streamlit)
EXPOSE 8000

# Set python path environment variable
ENV PYTHONPATH=/app

CMD ["python", "src/models/inference.py"]
