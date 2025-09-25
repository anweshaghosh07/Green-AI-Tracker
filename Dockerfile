# ---------- Stage 1: builder ----------
FROM python:3.11.8-slim-bullseye AS builder

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build tools (will not remain in final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY requirements.txt .

# Install Python dependencies globally (not --user)
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.11.8-slim-bullseye

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local /usr/local

# Copy project files
COPY data/ /app/data/
COPY models/ /app/models/
COPY . /app

# Create non-root user for safety
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Default command: launch Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]