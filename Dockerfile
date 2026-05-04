FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Install PyTorch CPU-only first (saves ~1.5GB vs full torch)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directory
RUN mkdir -p data model/checkpoints

# Expose port
EXPOSE 8000

# Run the server (includes built-in API poller)
CMD ["python", "run.py"]
