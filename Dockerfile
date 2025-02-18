FROM python:3.12-slim-bookworm

# Install essential packages (curl and ca-certificates), then install Node.js (includes npm/npx)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
apt-get install -y nodejs

# Download and install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Install FastAPI and Uvicorn
RUN pip install fastapi uvicorn

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin:$PATH"

# Set up the application directory
WORKDIR /app

# Copy application files
COPY . /app

# Explicitly set the correct binary path and use `sh -c`
CMD ["/root/.local/bin/uv", "run", "app.py"]