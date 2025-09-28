# Stage 1: Builder
# This stage installs poetry, copies the project files, and exports the dependencies to a requirements.txt file.
FROM python:3.12-slim as builder

# Install Poetry
RUN pip install poetry

# Set working directory
WORKDIR /app

# Copy only the files needed for dependency resolution
COPY pyproject.toml poetry.lock ./

# Export dependencies to a requirements.txt file
# --without-hashes is used for broader compatibility, and dev dependencies are excluded
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Stage 2: Runtime
# This stage uses a slim Python image, creates a non-root user, and installs dependencies.
FROM python:3.12-slim as runtime

# Set working directory
WORKDIR /app

# Create a non-root user and group
RUN addgroup --system nonroot && adduser --system --ingroup nonroot nonroot
USER nonroot

# Copy the requirements.txt from the builder stage
COPY --from=builder /app/requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the application source code
COPY ./src ./src

# Set the default command for the container
# This is a placeholder; users can override it
CMD ["python"]
