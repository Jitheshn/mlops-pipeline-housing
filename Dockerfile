# Use official Python image
FROM python:3.10-slim

# Install sqlite3 (and clean up to keep image small)
RUN apt-get update && apt-get install -y sqlite3 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into container
COPY . .

# Expose port (default FastAPI port when using uvicorn)
EXPOSE 8000

# Run the app using uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
