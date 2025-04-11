# FlyBase Automated Website Testing 

This repository contains a QA testing suite designed to automate the testing of the FlyBase preview website. The suite uses Selenium for browser automation and OpenAI's GPT API for advanced text and image analysis. The configuration for tests is managed through a YAML file used to define and customize test cases.

## Features
- Automated browser navigation and interaction using Selenium.
- Text and image comparison between staging and production environments.
- YAML-based configuration for defining test cases.
- Integration with OpenAI's GPT API for advanced analysis and reporting.
- Dockerized environment for consistent and reproducible testing.

## Repository Structure
- `Dockerfile`: Defines the Docker image for the testing suite.
- `LICENSE`: License information for the repository.
- `main.py`: Main script for running the tests.
- `README.md`: Documentation for the repository.
- `requirements.txt`: Python dependencies for the project.
- `test_config.yml`: YAML configuration file for defining test cases.

## Prerequisites
- Docker installed on your system.
- An OpenAI API key. Set it as an environment variable `OPENAI_API_KEY`.

## Getting Started

### 1. Clone the Repository
```bash
git clone <repository-url>
cd website_automated_testing
```

### 2. Build the Docker Image
```bash
docker build -t website-testing-suite .
```

### 3. Run the Tests
```bash
docker run --rm -e OPENAI_API_KEY="your-api-key" website-testing-suite
```

### 4. Customize Test Cases
Edit the `test_config.yml` file to define or modify test cases. Each test case includes the following fields:
- `name`: Name of the test.
- `enabled`: Whether the test is enabled.
- `url`: URL to test.
- `check_types`: Types of checks to perform (e.g., `text`, `picture`).
- `compare_to_production`: Whether to compare the staging environment to production.
- `prompt`: Description of what to verify.

## Artifacts
Test results and artifacts (e.g., screenshots, text files) are saved in the `/app/artifacts` directory inside the Docker container.

## Dependencies
The following Python libraries are used:
- `selenium`: For browser automation.
- `openai`: For interacting with OpenAI's GPT API.
- `PyYAML`: For parsing the YAML configuration file.

## License
This project is licensed under the terms of the LICENSE file.
