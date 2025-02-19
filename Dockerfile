FROM python:3.12-slim-bookworm

# Install essential packages (curl and ca-certificates), then install Node.js (includes npm/npx) and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates ffmpeg && \
    curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs

# Set working directory to /app
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Activate the virtual environment
ENV PATH="/app/venv/bin:$PATH"

# Copy application code
COPY app.py .

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
