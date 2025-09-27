# CI/CD Strategy and Implementation

## 1. Overview

This document outlines the architecture and rationale behind the CI/CD pipeline implemented for the `py_document_chunker` repository. The primary goal is to establish a robust, secure, and efficient automated workflow that ensures code quality, dependency consistency, and container security.

The pipeline automates linting, testing, and containerization, providing fast feedback to developers and preparing a secure, production-ready Docker image.

## 2. Technology Stack and Rationale

After a thorough analysis of the repository, the following technology stack was confirmed and adopted:

- **Dependency Management:** **pip with `pyproject.toml` (PEP 621)**
  - **Rationale:** The repository was already configured to use the modern PEP 621 standard for declaring dependencies in `pyproject.toml`, with `setuptools` as the build backend. This is a clean, standard, and widely supported approach within the Python ecosystem. It avoids adding unnecessary complexity from third-party managers like Poetry or Hatch, which were not previously in use.

- **Testing Framework:** **pytest**
  - **Rationale:** `pytest` was declared as a development dependency and is the de-facto standard for testing in the Python community, offering a powerful and extensible framework.

- **Linting & Formatting:** **pre-commit** with **Ruff**, **Black**, and **Mypy**
  - **Rationale:** The repository had a basic `.pre-commit-config.yaml`. This was retained and significantly enhanced to create a comprehensive static analysis suite. `pre-commit` automates the execution of hooks, ensuring consistent code quality checks before code is committed.

## 3. Proactive Improvements Made

Several foundational improvements were made to address gaps in the existing repository structure:

1.  **Consolidated CI/CD Workflows:**
    - **Action:** Legacy, unused workflow files (`publish.yml`, `python-tests.yml`) were removed from `.github/workflows/`.
    - **Result:** The repository now has a single, unambiguous set of CI/CD workflows (`ci.yml` and `docker.yml`), eliminating confusion and centralizing pipeline logic.

2.  **Comprehensive `pre-commit` Configuration:**
    - **Action:** The existing `.pre-commit-config.yaml` was enhanced. The `mypy` hook was added to enforce strict static type checking, and `check-toml` was added to validate the `pyproject.toml` file.
    - **Result:** Code quality is now enforced more rigorously, catching potential type-related errors and configuration issues early.

3.  **Added `Dockerfile` for Containerization:**
    - **Action:** A new, production-grade `Dockerfile` was created from scratch.
    - **Result:** The application can now be containerized. The `Dockerfile` implements best practices, including:
        - **Multi-stage builds:** A `builder` stage installs dependencies, and a `final` stage copies only the necessary application code and installed packages, resulting in a smaller, more secure image.
        - **Non-root user:** A dedicated `appuser` is created and used, adhering to the principle of least privilege and enhancing security.
        - **Lean base image:** It uses the `python:3.12-slim-bookworm` base image to minimize the image size and attack surface.

4.  **Added `.dockerignore`:**
    - **Action:** A comprehensive `.dockerignore` file was created.
    - **Result:** The Docker build context is now minimal, excluding the `.git` directory, virtual environments, caches, and other unnecessary files. This improves build speed and enhances security by preventing secrets or sensitive history from being included in the image.

## 4. Workflow Architecture

The CI/CD pipeline is split into two distinct, parallel workflows:

### `ci.yml` - Continuous Integration

This workflow focuses on code quality and correctness. It is designed for fast feedback.

- **Triggers:** Runs on `push` and `pull_request` events to the `main` branch.
- **Jobs:**
    1.  **`lint`:**
        - **Purpose:** Performs fast static analysis checks.
        - **Implementation:** Runs on `ubuntu-latest` with Python 3.12. It uses the `pre-commit/action` to execute all configured hooks (`ruff`, `black`, `mypy`, etc.). A failure in this job prevents the `test` job from running.
    2.  **`test`:**
        - **Purpose:** Runs the full test suite across a variety of environments.
        - **Implementation:** This job only runs after the `lint` job succeeds (`needs: [lint]`). It uses a matrix strategy to test across multiple operating systems and Python versions.

### `docker.yml` - Docker Build & Scan

This workflow focuses on containerization and security scanning.

- **Triggers:** Runs on `push` and `pull_request` events to the `main` branch.
- **Jobs:**
    1.  **`build-and-scan`:**
        - **Purpose:** Builds a Docker image and scans it for vulnerabilities.
        - **Implementation:** It builds the image using the `Dockerfile` but **does not push it**. The image is loaded locally into the runner so it can be scanned by Trivy. This provides security feedback on pull requests without polluting a container registry.

## 5. Testing Strategy

- **Matrix Testing:** The `test` job in `ci.yml` runs a matrix across:
  - **Operating Systems:** `ubuntu-latest`, `macos-latest`, `windows-latest`
  - **Python Versions:** `3.8`, `3.11`, `3.12`
- **Test Execution:** Tests are executed using `pytest --cov=src`, which runs all tests and generates a coverage report.
- **Code Coverage:**
  - Coverage reports (`coverage.xml`) are generated for each test run in the matrix.
  - The `codecov/codecov-action` uploads these reports to Codecov.io.
  - Each report is tagged with a unique flag (`${{ matrix.os }}-py${{ matrix.python-version }}`) to allow for distinct coverage analysis per environment in the Codecov UI.

## 6. Dependency Management and Caching

- **Installation:** Dependencies are installed via `pip install .[dev]`. This command reads the `pyproject.toml` file and installs all main dependencies along with the optional development dependencies.
- **Caching:** The `actions/setup-python` action is configured with `cache: 'pip'`. This leverages GitHub's native caching mechanism to store and restore downloaded packages, significantly speeding up dependency installation on subsequent runs.

## 7. Security Hardening

Security was a core consideration in the design of these workflows:

- **Principle of Least Privilege (PoLP):** All workflows are configured with `permissions: contents: read` at the top level. This ensures that jobs only have the minimum required access to the repository and cannot perform write operations unless explicitly granted at the job or step level.
- **Action Pinning:** All third-party GitHub Actions (e.g., `actions/checkout`, `docker/login-action`) are pinned to their full 40-character commit SHA. This prevents the workflow from being compromised by a malicious update to a mutable tag (like `v3` or `main`).
- **Docker Hub Authentication:** The `docker.yml` workflow includes a step to log in to Docker Hub using secrets. This is a best practice to avoid anonymous pull-rate limits, which can cause CI failures. The login step is conditionally run only if the required secrets are present.
- **Non-Root Docker Container:** The `Dockerfile` creates and runs the application as a non-root user (`appuser`), reducing the potential impact of a container breakout vulnerability.
- **Vulnerability Scanning:** The `docker.yml` workflow uses the `aquasecurity/trivy-action` to scan the built Docker image for known vulnerabilities. The workflow is configured to fail the build (`exit-code: '1'`) if any `CRITICAL` or `HIGH` severity vulnerabilities are found.

## 8. Docker Strategy

- **Multi-Stage Build:** The `Dockerfile` uses a multi-stage process to create a lean and secure final image, separating build-time dependencies from the runtime environment.
- **Build Caching:** The `docker/build-push-action` is configured to use the GitHub Actions cache (`type=gha`) as a remote cache. This significantly speeds up image builds by reusing layers from previous runs.
- **Verification, Not Pushing:** On pull requests and pushes to `main`, the image is built and scanned but **not** pushed to a registry. This verifies that the `Dockerfile` is valid and the resulting image is secure, without creating unnecessary image artifacts. A separate release workflow would be required to publish images.

## 9. How to Run Locally

Developers can replicate the CI checks locally to ensure their changes will pass in the pipeline.

### Linting and Static Analysis

To run the same checks as the `lint` job, use `pre-commit`:

```bash
# Install pre-commit if you haven't already
pip install pre-commit

# Install the git hooks
pre-commit install

# Run all hooks against all files
pre-commit run --all-files
```

### Testing

To run the same tests as the `test` job:

```bash
# Install development dependencies
pip install .[dev]

# Run pytest with coverage
pytest --cov=src
```