# CI/CD Implementation Strategy

This document outlines the architecture and strategy for the CI/CD pipelines implemented in this repository. The goal is to establish a robust, secure, and efficient automated system for code integration, testing, and containerization.

## 1. Technology Stack

A standardized and modern technology stack was chosen to ensure clarity, maintainability, and performance.

*   **Dependency Management:** **Poetry** was selected as the sole dependency manager. It provides a deterministic dependency resolution process via the `poetry.lock` file, an integrated build system, and a clear separation of dependencies. The project was migrated from `setuptools`.
*   **Linting & Formatting:** **Ruff** is used for both high-speed linting and code formatting, enforced via `pre-commit`.
*   **Static Type Checking:** **Mypy** is used for static type analysis to catch type-related errors before runtime, enforced via `pre-commit`.
*   **Code Quality Enforcement:** **pre-commit** is used to run all code quality tools automatically before each commit, ensuring that code adheres to standards without manual intervention.

## 2. Foundational Improvements

Before implementing the workflows, several foundational improvements were made to the repository to support a modern CI/CD strategy.

*   **Comprehensive `pre-commit` Configuration:** A `.pre-commit-config.yaml` file was added with hooks for formatting (Ruff Format), linting (Ruff), static type checking (Mypy), and general repository hygiene (YAML checks, end-of-file fixes).
*   **Optimized Multi-Stage `Dockerfile`:** A new `Dockerfile` was created following best practices:
    *   **Builder Stage:** A dedicated stage installs Poetry and exports dependencies to a standard `requirements.txt` file. This keeps the final image lean.
    *   **Runtime Stage:** A separate, minimal `python:3.12-slim` image is used for the final container. It installs dependencies from the exported `requirements.txt` without needing Poetry itself.
    *   **Non-Root User:** The runtime stage creates and switches to a `nonroot` user to enhance security by avoiding running the container as root.
*   **Optimized `.dockerignore`:** A `.dockerignore` file was added to minimize the Docker build context, excluding version control files, virtual environments, caches, and documentation. This speeds up the build process and reduces the chance of leaking sensitive information.

## 3. Workflow Architecture

The CI/CD system is composed of three distinct, focused workflows: `ci.yml`, `docker.yml`, and `publish.yml`.

### `ci.yml` (Linting & Testing)

This workflow validates code quality and correctness on every push and pull request to the `main` and `develop` branches.

*   **Structure:** It follows a sequential `Lint -> Test` structure. The `test` job only runs if the `lint` job succeeds, providing fast feedback on quality issues.
*   **Linting:** The `lint` job uses `pre-commit/action` to efficiently run all configured pre-commit hooks, leveraging caching for speed.
*   **Testing Matrix:** The `test` job runs on a matrix to ensure cross-platform and multi-version compatibility:
    *   **Operating Systems:** `ubuntu-latest`, `macos-latest`, `windows-latest`
    *   **Python Versions:** `3.11`, `3.12`
*   **Dependency Caching:** `actions/setup-python` is configured with `cache: 'poetry'` to cache installed dependencies, significantly speeding up subsequent runs.

### `docker.yml` (Build, Cache & Scan)

This workflow builds and scans the Docker image to ensure it is free of known vulnerabilities.

*   **Docker Hub Authentication:** The workflow safely logs into Docker Hub to prevent rate-limiting on image pulls, a common issue in CI environments.
*   **Build Caching:** It uses `docker/build-push-action` with the GHA (GitHub Actions) cache backend (`type=gha`). This caches Docker layers between runs, resulting in much faster builds for minor changes.
*   **Vulnerability Scanning:** After building, the image is scanned with `aquasecurity/trivy-action`. The workflow is configured to fail if any `CRITICAL` or `HIGH` severity vulnerabilities are detected.

### `publish.yml` (Package Publishing)

This workflow automates the process of publishing the package to PyPI.

*   **Trigger:** It runs only when a new release is published on GitHub, ensuring that only tagged, official versions are sent to PyPI.
*   **Authentication:** It uses PyPI's trusted publishing (OIDC) for secure, tokenless authentication, which is the current best practice.

## 4. Testing and Coverage Strategy

*   **Test Execution:** Tests are run using `pytest` via `poetry run pytest`.
*   **Coverage Reporting:** A coverage report (`coverage.xml`) is generated during the test run.
*   **Codecov Integration:** The `codecov/codecov-action` uploads this report to Codecov. To prevent data overwrites from the test matrix, unique flags (`flags: ${{ matrix.os }}-py${{ matrix.python-version }}`) are assigned to each coverage report. This allows Codecov to correctly merge reports from all jobs in the matrix.

## 5. Security Measures

Security was a non-negotiable principle throughout the implementation.

*   **Action Pinning (Full SHA):** All third-party GitHub Actions used in all workflows are pinned to their full-length commit SHA. This prevents supply chain attacks where a malicious actor could take over a tag (e.g., `v4`) and inject malicious code.
*   **Principle of Least Privilege (PoLP):** All workflows are configured with default `permissions: contents: read`. More permissive tokens (like `id-token: write` for publishing) are only granted to the specific jobs that require them, minimizing the potential impact of a compromised workflow.
*   **Non-Root Container:** The production `Dockerfile` runs the application as a `nonroot` user, reducing the attack surface if a vulnerability were to be exploited inside the container.
*   **Vulnerability Scanning:** The `docker.yml` workflow automatically scans every new image build for `CRITICAL` and `HIGH` severity vulnerabilities with Trivy, preventing insecure images from being used.
*   **Secret-Scanning:** While not explicitly part of the CI workflows, the use of `pre-commit` allows for the easy addition of secret-scanning hooks (e.g., `detect-secrets`) in the future.