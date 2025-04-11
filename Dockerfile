# Dockerfile for a QA Testing Suite

FROM flybase/harvdev-docker:latest

# Install Python3, py3-pip, build tools, and needed packages including Chromium and its driver
RUN apk update && \
    apk add --no-cache \
        unzip \
        gnupg \
        chromium \
        chromium-chromedriver \
        build-base \
        yaml-dev
        
# Copy requirements (optional if we have a requirements.txt)
# Or we can just do a pip install of each
COPY requirements.txt /app/requirements.txt

# Upgrade pip and build tools FIRST
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy your scripts and config into the image
COPY main.py /app/main.py
COPY test_config.yml /app/test_config.yml

WORKDIR /app

# By default, run the QA tests
CMD ["python", "main.py", "--config", "test_config.yml"]