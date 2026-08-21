FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend Python code and benchmark assets (large data parquet/faiss auto-downloaded from Hugging Face Hub at startup)
COPY *.py ./
COPY *.json ./

EXPOSE 8000

CMD ["python", "app.py"]
