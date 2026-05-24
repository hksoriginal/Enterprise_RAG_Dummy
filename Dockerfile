# Use a slim Python image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies often needed by ML / native wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       curl \
       ca-certificates \
       libsndfile1 \
       libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Expose port
EXPOSE 8000

# Default command: run the FastAPI app
# Use the module path from the repository root
CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "8000"]
