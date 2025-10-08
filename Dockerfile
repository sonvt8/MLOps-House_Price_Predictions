FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional but useful for scientific stacks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest and install first (leverages Docker cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application source
COPY src/ /app/src/

# Ensure Python can find the src package (api, models, etc.)
ENV PYTHONPATH=/app/src

# Expose API port
EXPOSE 8000

# Default command: run FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]