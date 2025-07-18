# Multi-stage build for M.I.A v0.1.0
FROM python:3.9-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
    pip install --user -r /tmp/requirements.txt

# Production stage
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/mia/.local/bin:$PATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    alsa-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash mia

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/mia/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=mia:mia . .

# Install the package
USER mia
RUN pip install --user -e .

# Create necessary directories
RUN mkdir -p logs memory config

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from mia.__version__ import __version__; print(__version__)" || exit 1

# Expose port (if web interface is added later)
EXPOSE 8000

# Default command
CMD ["mia", "--info"]
CMD ["python", "-m", "main_modules.main"]
