# Dockerfile for running Playwright QA tests

# Use the specified base image (ensure it has Python 3.x)
FROM flybase/harvdev-docker:latest

# Install base system dependencies potentially needed by Playwright install or Python build tools
# Playwright browser dependencies will be handled by 'playwright install --with-deps' later
RUN apk update && \
    apk add --no-cache \
        unzip \
        gnupg \
        build-base \
        # Keep yaml-dev in case PyYAML needs C extensions, safer than removing
        yaml-dev \
    # Clean apk cache to reduce image size
    && rm -rf /var/cache/apk/*

# Set working directory
WORKDIR /app

# Copy requirements file first for better Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies from requirements file (including Playwright)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers and their OS dependencies using the --with-deps flag
# This is crucial as it installs necessary system libraries for the browser on Alpine Linux.
# Currently installing only Chromium, as implied by the original Dockerfile.
# Add 'firefox' or 'webkit' if needed: playwright install --with-deps chromium firefox webkit
RUN playwright install --with-deps chromium

# Copy the main application script and configuration file
# Assumes your Playwright script is named main.py
COPY main.py /app/main.py
COPY test_config.yml /app/test_config.yml

# --- Optional: Copy Report Generation Files ---
# These are typically NOT needed inside *this* container if your GitHub Action
# generates the report *after* this container runs and artifacts are copied out.
# Uncomment only if you intend to run report generation *inside* this container.
# COPY scripts/generate_html_report.py /app/scripts/generate_html_report.py
# COPY scripts/templates/report_template.html /app/scripts/templates/report_template.html
# ---------------------------------------------

# Set environment variable for Playwright browser path (recommended for consistency)
# The path /root/.cache/ms-playwright is common when running as root user in containers.
ENV PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright

# Define the default command to run the test script.
# This aligns with the GitHub Action running tests and then handling reports externally.
CMD ["python", "main.py", "--config", "test_config.yml"]

# --- Optional: Alternative CMD for In-Container Report Generation ---
# If you uncommented the COPY lines for report scripts above and want to
# run everything inside the container, you could use this CMD instead:
# CMD ["sh", "-c", "python main.py --config test_config.yml && python scripts/generate_html_report.py"]
# Note: This would require mounting /app/docs volume in your 'docker run' command
# in the GitHub Action if you want to retrieve the generated report.
# ---------------------------------------------------------------------