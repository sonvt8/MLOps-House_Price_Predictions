FROM python:3.11-slim

WORKDIR /app

# Install system build tools (optional but helpful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy root requirements and install
COPY ../../requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application source
COPY ../.. /app

# Ensure Python can import from src
ENV PYTHONPATH=/app/src

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "src/streamlit_app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
