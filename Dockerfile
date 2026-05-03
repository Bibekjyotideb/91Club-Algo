FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directory
RUN mkdir -p data model/checkpoints

# Expose port
EXPOSE 8000

# Run the server (includes built-in API poller)
CMD ["python", "run.py"]
