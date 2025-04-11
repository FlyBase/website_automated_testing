FROM flybase/harvdev-docker:latest

# Install Python3, pip, and all necessary dependencies
RUN apk update && \
    apk add --no-cache \
        unzip \
        gnupg \
        chromium \
        chromium-chromedriver \
        build-base \
        yaml-dev

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY main.py /app/main.py
COPY test_config.yml /app/test_config.yml

WORKDIR /app

CMD ["python", "main.py", "--config", "test_config.yml"]
