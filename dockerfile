FROM python:3.12.1-bookworm

ENV PYTHONUNBUFFERED=True

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create model and cache directories
RUN mkdir -p ./models
RUN mkdir -p ./cache

# Pre-download and save the model during build time
RUN python -c "from app import download_model_during_build; download_model_during_build()"

# Start gunicorn with increased timeout
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "8", "--timeout", "300", "app:app"]