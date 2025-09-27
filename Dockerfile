# ---- Base ----
FROM python:3.12-slim-bookworm AS base

# Set working directory
WORKDIR /app

# Create a non-root user and switch to it
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# ---- Builder ----
FROM base AS builder

# Install build dependencies
COPY --chown=appuser:appuser pyproject.toml setup.py ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user build

# Install application dependencies
RUN pip install --no-cache-dir --user .

# ---- Final ----
FROM base AS final

# Copy installed packages from builder
COPY --from=builder /home/appuser/.local /home/appuser/.local

# Copy application source
COPY --chown=appuser:appuser src/ ./src/

# Set the PATH to include the installed packages
ENV PATH=/home/appuser/.local/bin:$PATH

# Set a default command
CMD ["python", "-c", "print('Welcome to py_document_chunker')"]