# Dockerfile for running Playwright QA tests using a Debian-based Python image

# Use an official Python 3.11 Slim Debian (Bookworm) base image
FROM python:3.11-slim-bookworm

# Install base system dependencies using apt-get
# - build-essential includes gcc, g++, make etc. (like build-base on Alpine)
# - libyaml-dev for PyYAML C extensions (like yaml-dev on Alpine)
# - unzip, gnupg potentially needed by Playwright install scripts
# - curl is useful for debugging network issues
# - ca-certificates helps ensure HTTPS connections work reliably
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        unzip \
        gnupg \
        build-essential \
        libyaml-dev \
        curl \
        ca-certificates \
    # Clean apt cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first for better Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies using python3's pip module
# Ensures we use the correct pip associated with python3
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers and their OS dependencies using the --with-deps flag
# This command works on Debian/Ubuntu too; it will use apt-get internally.
# Installing only chromium, add 'firefox' or 'webkit' if needed.
RUN python3 -m playwright install --with-deps chromium

# Copy the main application script and configuration file
# Assumes your Playwright script is named main.py
COPY main.py /app/main.py
COPY test_config.yml /app/test_config.yml

# --- Optional: Copy Report Generation Files ---
# Uncomment only if you intend to run report generation *inside* this container.
# COPY scripts/generate_html_report.py /app/scripts/generate_html_report.py
# COPY scripts/templates/report_template.html /app/scripts/templates/report_template.html
# ---------------------------------------------

# Set environment variable for Playwright browser path (standard location for root user)
ENV PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright

# Define the default command to run the test script using python3
CMD ["python3", "main.py", "--config", "test_config.yml"]

# --- Optional: Alternative CMD for In-Container Report Generation ---
# CMD ["sh", "-c", "python3 main.py --config test_config.yml && python3 scripts/generate_html_report.py"]
# ---------------------------------------------------------------------