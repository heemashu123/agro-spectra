FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them, including fastapi for the inference server
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Copy project files
COPY . .

# Expose port for OpenEnv validation API
EXPOSE 8000

# Start the inference server
CMD ["python", "inference.py"]
