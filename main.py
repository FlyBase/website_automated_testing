import os
import sys
import time
import json
import base64
import yaml
import difflib
import argparse
# Make sure urlparse is imported
from urllib.parse import urlparse, urldefrag

# Updated OpenAI imports
from openai import OpenAI, OpenAIError

# Simplified Selenium Imports (only for navigation and capture)
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# Import exceptions for safer JS execution
from selenium.common.exceptions import JavascriptException, NoSuchElementException

###############################################################################
# CONFIGURATION AND HELPERS
###############################################################################

# Define the path for artifacts inside the container
ARTIFACTS_DIR = "/app/artifacts"
# Define the base Production URL for comparison purposes
PROD_BASE_URL = "https://flybase.org"

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    sys.exit("Please set the OPENAI_API_KEY environment variable inside the container (e.g. -e OPENAI_API_KEY='sk-...').")

# Initialize OpenAI client (v1.x)
client = OpenAI()
# Use the new model name if intended
MODEL_NAME = "gpt-4o-mini" # Changed from gpt-4o

FUNCTION_SCHEMA = [
    {
        "name": "record_test_result",
        "description": (
            "Records whether the test passed or failed, and which component failed "
            "(text, image, or both). Also provides an explanation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": "Overall test outcome",
                    "enum": ["pass", "fail"]
                },
                "failed_component": {
                    "type": "string",
                    "description": "Which component failed, or 'none' if passed.",
                    "enum": ["text", "image", "both", "none", "page load"]
                },
                "explanation": {
                    "type": "string",
                    "description": "Explanation for the result"
                }
            },
            "required": ["result", "explanation"]
        }
    }
]

def setup_selenium():
    """Setup Selenium with headless Chrome/Chromium, and inject a custom WAF header if present."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(1280, 1024)

    # If your WAF expects some hidden header, read it from env and set it
    waf_secret = os.getenv("WAF_SECRET_HEADER")
    if waf_secret:
        try:
            # Enable network, then set extra header
            driver.execute_cdp_cmd("Network.enable", {})
            driver.execute_cdp_cmd("Network.setExtraHTTPHeaders", {
                "headers": {
                    "x-waf-secret": waf_secret
                }
            })
            print("WAF secret header set successfully.")
        except Exception as e:
            print(f"Warning: Could not set WAF secret header: {e}")

    return driver

def get_page_text(driver, url):
    """Fetch the entire visible text from the page body of the given URL."""
    try:
        print(f"    Getting text from: {url}")
        driver.get(url)
        # Increased wait slightly for potentially complex pages
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # --- Add check for fragment and scroll if necessary ---
        _base, fragment = urldefrag(url)
        if fragment:
            try:
                print(f"    Fragment '#{fragment}' detected. Attempting to scroll into view.")
                # Find element by ID matching the fragment
                target_element = driver.find_element(By.ID, fragment)
                # Use JavaScript to scroll the element into view
                driver.execute_script("arguments[0].scrollIntoView(true);", target_element)
                time.sleep(0.75) # Short pause for scroll animation/rendering
                print(f"    Scrolled to element with ID '{fragment}'.")
            except NoSuchElementException:
                print(f"    Warning: Could not find element with ID '{fragment}' to scroll to.")
            except JavascriptException as js_err:
                print(f"    Warning: JavaScript error while scrolling to '#{fragment}': {js_err}")
            except Exception as scroll_err: # Catch other potential errors
                 print(f"    Warning: Error during scrolling to '#{fragment}': {scroll_err}")
        # --- End fragment check ---

        # Re-find body element after potential scroll
        body_element = driver.find_element(By.XPATH, "//body")
        return body_element.text
    except Exception as e:
        print(f"    Error extracting text from {url}: {e}")
        return f"Error extracting text: {e}"


def get_page_screenshot(driver, url, screenshot_filename_with_path):
    """
    Capture a screenshot of the given URL. If the URL has a fragment (#anchor),
    attempts to scroll to that anchor before taking the screenshot.
    Returns the image bytes.
    """
    try:
        print(f"    Getting screenshot from: {url}")
        if url:
            driver.get(url)
            # Wait for basic page load state
            WebDriverWait(driver, 20).until(lambda d: d.execute_script('return document.readyState') == 'complete') # Increased wait
            print(f"    Page '{url}' loaded (readyState complete).")

            # --- Add check for fragment and scroll if necessary ---
            _base, fragment = urldefrag(url) # Use urldefrag to safely get fragment
            if fragment:
                try:
                    print(f"    Fragment '#{fragment}' detected. Attempting to scroll into view.")
                    # Find element by ID matching the fragment
                    # Wait briefly for the element to potentially exist after load
                    target_element = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.ID, fragment))
                    )
                    # Use JavaScript to scroll the element into view
                    driver.execute_script("arguments[0].scrollIntoView(true);", target_element)
                    print(f"    Executed scrollIntoView for ID '{fragment}'.")
                    # Wait a bit longer after scroll for rendering/animations
                    time.sleep(1.5) # Increased sleep after scroll command
                except NoSuchElementException:
                     # It's possible the fragment doesn't correspond to an ID, or the element isn't found
                     print(f"    Warning: Could not find element with ID '{fragment}' to scroll to.")
                except JavascriptException as js_err:
                     print(f"    Warning: JavaScript error while scrolling to '#{fragment}': {js_err}")
                except Exception as scroll_err: # Catch other potential errors like timeout waiting for element
                     print(f"    Warning: Error during scrolling attempt for '#{fragment}': {scroll_err}")
            else:
                 # If no fragment, still wait a standard amount
                 time.sleep(1)
            # --- End fragment check ---

        # Proceed to take screenshot after load and potential scroll attempt
        os.makedirs(os.path.dirname(screenshot_filename_with_path), exist_ok=True)
        driver.save_screenshot(screenshot_filename_with_path)
        print(f"    Saved screenshot artifact: {screenshot_filename_with_path}")
        with open(screenshot_filename_with_path, "rb") as img_file:
            return img_file.read()
    except Exception as e:
        print(f"    Error capturing or reading screenshot {screenshot_filename_with_path} for URL {url}: {e}")
        return None


def encode_image_to_data_uri(image_bytes, image_format="png"):
    """Encode image bytes in base64 with data URI format."""
    if image_bytes is None:
        return ""
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{image_format};base64,{image_b64}"

def load_yaml_config(config_path):
    """Load tests from a YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config.get("tests", [])
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file {config_path}: {e}")
        sys.exit(1)

def call_openai_api(messages):
    """Call OpenAI's ChatCompletion API using function calling (v1.x API)."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            functions=FUNCTION_SCHEMA,
            function_call={"name": "record_test_result"},
            temperature=0
        )
        return response
    except OpenAIError as e:
        print(f"OpenAI API call error: {e}")
        return None

def parse_api_response(response):
    """Parse the response from OpenAI - returns a dict with result, failed_component, explanation."""
    result_status = "fail"
    failed_component = "unknown"
    explanation = "Could not parse API response or API call failed."

    if not response:
        explanation = "No API response received from OpenAI (API call likely failed)."
    else:
        try:
            message = response.choices[0].message
            if message.function_call:
                arguments_str = message.function_call.arguments
                try:
                    result_json = json.loads(arguments_str)
                    if "result" in result_json and "explanation" in result_json:
                        result_status = result_json.get("result", "fail")
                        failed_component = result_json.get("failed_component", "none" if result_status == "pass" else "unknown")
                        explanation = result_json.get("explanation", "No explanation provided.")
                        # Successfully parsed API response
                        return {
                            "result": result_status,
                            "failed_component": failed_component,
                            "explanation": explanation
                        }
                    else:
                        explanation = (
                            f"Parsed JSON from function call missing required fields. Parsed: {result_json}"
                        )
                        print(f"Warning: {explanation}")
                except json.JSONDecodeError as e:
                    explanation = f"Error parsing function call arguments: {e}. Arguments: {arguments_str}"
                    print(explanation)
            elif message.content:
                explanation = f"API returned content instead of expected function call: {message.content}"
                print(explanation)
            else:
                explanation = "API response did not contain a function call or content."
                print(explanation)
        except (AttributeError, IndexError, KeyError) as e:
            explanation = f"Error accessing data in API response structure: {e}. Response: {response}"
            print(explanation)

    # Return failure details if parsing failed at any point above
    return {
        "result": result_status, # fail
        "failed_component": failed_component, # unknown or determined before error
        "explanation": explanation # Specific error explanation
    }

def run_test(driver, test_def):
    """
    Execute a basic test (navigate, capture) based on YAML config.
    Returns a dict with test name, prompt, result, failed_component, and explanation.
    """
    test_name = test_def.get("name", "Unnamed Test")
    safe_test_name = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in test_name).rstrip('_')
    # <<< Get prompt early to include in all return paths >>>
    prompt = test_def.get("prompt", "")

    enabled = test_def.get("enabled", True)
    if not enabled:
        print(f"Skipping disabled test: {test_name}")
        return {
            "test_name": test_name,
            "result": "skipped",
            "failed_component": "none",
            "explanation": "Test is disabled.",
            "prompt": prompt # <<< Include prompt >>>
        }

    url_from_yaml = test_def.get("url")
    if not url_from_yaml:
        print(f"Skipping test '{test_name}': Missing required 'url' field in YAML.")
        return {
            "test_name": test_name,
            "result": "skipped",
            "failed_component": "none",
            "explanation": "Missing 'url'.",
            "prompt": prompt # <<< Include prompt >>>
        }

    compare_to_production = test_def.get("compare_to_production", False)
    check_types = test_def.get("check_types", [])
    # Prompt already retrieved above
    ticket = test_def.get("ticket", "")

    preview_url = url_from_yaml
    prod_url = None

    if compare_to_production:
        try:
            # Construct prod URL preserving path, query, and fragment from target
            parsed_target = urlparse(preview_url)
            prod_path_query_fragment = urlunparse(('', '', parsed_target.path, parsed_target.params, parsed_target.query, parsed_target.fragment))
            prod_url = urljoin(PROD_BASE_URL, prod_path_query_fragment) # Handles relative paths correctly
        except Exception as e:
            print(f"Warning: Could not construct production URL for comparison from {preview_url}: {e}")
            compare_to_production = False # Disable comparison if prod URL fails
            prod_url = None

    print(f"\n=== Running Test: {test_name} ===")
    print(f"Preview URL: {preview_url}")
    if compare_to_production and prod_url:
        print(f"Comparing against Production URL: {prod_url}")

    system_content = (
        "You are a meticulous QA automation assistant specializing in website visual and textual consistency..."
        # Shorten system prompt if needed for context window or cost
    )
    messages = [{"role": "system", "content": system_content}]
    user_content_parts = []
    screenshots_data = []

    # This is the text part sent to the LLM, combining test name, prompt, and ticket
    prompt_display = f"Test: {test_name}\nPrompt: {prompt}"
    if ticket:
        prompt_display += f"\nReference Ticket: {ticket}"
    user_content_parts.append({"type": "text", "text": prompt_display})

    try:
        # --- Page interaction and data gathering ---
        if compare_to_production and prod_url:
            user_content_parts.append({"type": "text", "text": f"\n--- Comparing Production vs Preview ---"})
            user_content_parts.append({"type": "text", "text": f"Production URL: {prod_url}"})
            user_content_parts.append({"type": "text", "text": f"Preview URL: {preview_url}"})

            # Get text *after* potential scrolling (handled in get_page_text)
            prod_text = get_page_text(driver, prod_url) if "text" in check_types else "Text check not requested."
            preview_text = get_page_text(driver, preview_url) if "text" in check_types else "Text check not requested."

            if "text" in check_types:
                prod_text_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_prod.txt")
                preview_text_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_preview.txt")
                try:
                    with open(prod_text_path, "w", encoding="utf-8") as f: f.write(prod_text)
                    with open(preview_text_path, "w", encoding="utf-8") as f: f.write(preview_text)
                    print(f"Saved text artifacts to {ARTIFACTS_DIR}/")
                except Exception as e: print(f"Warning: Could not save text files for {test_name}: {e}")

                user_content_parts.append({"type": "text", "text": f"\nProduction Text:\n```\n{prod_text}\n```"})
                user_content_parts.append({"type": "text", "text": f"\nPreview State Text:\n```\n{preview_text}\n```"})

            if "picture" in check_types:
                prod_screenshot_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_prod.png")
                preview_screenshot_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_preview.png")
                # Get screenshots *after* potential scrolling (handled in get_page_screenshot)
                prod_screenshot_bytes = get_page_screenshot(driver, prod_url, prod_screenshot_path)
                preview_screenshot_bytes = get_page_screenshot(driver, preview_url, preview_screenshot_path)
                prod_img_uri = encode_image_to_data_uri(prod_screenshot_bytes)
                preview_img_uri = encode_image_to_data_uri(preview_screenshot_bytes)
                if prod_img_uri:
                    screenshots_data.append({"type": "image_url", "image_url": {"url": prod_img_uri}})
                    user_content_parts.append({"type": "text", "text": "\nProduction Screenshot:"})
                if preview_img_uri:
                    screenshots_data.append({"type": "image_url", "image_url": {"url": preview_img_uri}})
                    user_content_parts.append({"type": "text", "text": "\nPreview State Screenshot:"})
        else: # Single page test
            user_content_parts.append({"type": "text", "text": f"\n--- Single Page Test ---"})
            user_content_parts.append({"type": "text", "text": f"URL Tested: {preview_url}"})

            if "text" in check_types:
                 # Get text *after* potential scrolling (handled in get_page_text)
                page_text = get_page_text(driver, target_url)
                text_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_single.txt")
                try:
                    with open(text_path, "w", encoding="utf-8") as f: f.write(page_text)
                    print(f"Saved text artifact to {ARTIFACTS_DIR}/")
                except Exception as e: print(f"Warning: Could not save text file for {test_name}: {e}")
                user_content_parts.append({"type": "text", "text": f"\nPage Text:\n```\n{page_text}\n```"})

            if "picture" in check_types:
                screenshot_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_single.png")
                 # Get screenshot *after* potential scrolling (handled in get_page_screenshot)
                screenshot_bytes = get_page_screenshot(driver, target_url, screenshot_path)
                img_uri = encode_image_to_data_uri(screenshot_bytes)
                if img_uri:
                    screenshots_data.append({"type": "image_url", "image_url": {"url": img_uri}})
                    user_content_parts.append({"type": "text", "text": "\nPage Screenshot:"})

    except Exception as page_err:
        # --- Handle errors during page loading/capture ---
        error_msg = f"Failed during page load or data capture for test '{test_name}'. URL(s): {target_url}"
        if prod_url: error_msg += f", {prod_url}"
        error_msg += f". Error: {page_err}"
        print(f"    ERROR: {error_msg}")
        try: # Attempt failure screenshot
            fail_screenshot_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_FAILURE_capture.png")
            # Try taking screenshot without navigating again
            get_page_screenshot(driver, None, fail_screenshot_path)
        except Exception as screen_err: print(f"    Could not take failure screenshot: {screen_err}")
        return {
            "test_name": test_name,
            "result": "fail",
            "failed_component": "page load",
            "explanation": error_msg,
            "prompt": prompt # <<< Include prompt >>>
        }

    # --- Prepare and call OpenAI API ---
    user_message_content_list = user_content_parts + screenshots_data
    messages.append({"role": "user", "content": user_message_content_list})

    print("Prompt snippet for LLM call:", prompt_display[:200], "...\n") # Log snippet sent to LLM

    response = call_openai_api(messages)
    result_data = parse_api_response(response) # Contains result, failed_component, explanation

    # --- Combine test info with API result ---
    final_result = {
        "test_name": test_name,
        "prompt": prompt, # <<< Include prompt >>>
        **result_data # Unpack result, failed_component, explanation
    }
    return final_result


def main(config_path):
    # Create the artifacts directory
    try:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        print(f"Artifacts will be saved to container path: {ARTIFACTS_DIR}")
    except OSError as e:
        print(f"Error creating artifacts directory {ARTIFACTS_DIR}: {e}")
        # Decide if you want to exit or continue if dir creation fails

    tests = load_yaml_config(config_path)
    if not tests:
        print("No tests found in YAML configuration.")
        return

    driver = setup_selenium()
    all_results = []
    passed_count = 0
    failed_count = 0
    skipped_count = 0

    for test_def in tests:
        result = None
        try:
            result = run_test(driver, test_def) # run_test now returns the prompt too
            status = result.get("result")
            if status == "pass":
                passed_count += 1
            elif status == "fail":
                failed_count += 1
            elif status == "skipped":
                skipped_count += 1
        except Exception as err:
            test_name_fallback = test_def.get("name", "Unnamed Test")
            prompt_fallback = test_def.get("prompt", "") # Get prompt for error case too
            print(f"Test '{test_name_fallback}' encountered an unhandled exception: {err}")
            result = {
                "test_name": test_name_fallback,
                "result": "fail",
                "failed_component": "test execution",
                "explanation": f"Unhandled Exception occurred: {err}",
                "prompt": prompt_fallback # <<< Include prompt >>>
            }
            failed_count += 1

        if result:
            all_results.append(result) # result already contains the prompt

    driver.quit()

    results_filename = os.path.join(ARTIFACTS_DIR, "test_results.json")
    try:
        with open(results_filename, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results saved to container path: {results_filename}")
    except Exception as e:
        print(f"\nError saving results to {results_filename}: {e}")

    # --- Print Summary ---
    print("\n--- TEST SUMMARY ---")
    print(f"Total tests defined: {len(tests)}")
    print(f"Tests run: {passed_count + failed_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")

    if failed_count > 0:
        print("\n--- FAILED TEST DETAILS ---")
        for res in all_results:
            if res.get("result") == "fail":
                print(f"  Test FAILED: {res.get('test_name')}")
                print(f"    Prompt: {res.get('prompt', 'N/A')}") # <<< Also print prompt in console summary >>>
                failed_comp = res.get('failed_component', 'N/A')
                print(f"    Failed Component: {failed_comp}")
                print(f"    Explanation: {res.get('explanation', 'No explanation.')}\n")
        print("-------------------------")
    # ... [rest of the summary print statements] ...
    elif passed_count > 0 and failed_count == 0:
        print("\nALL EXECUTED TESTS PASSED.")
    elif skipped_count == len(tests):
        print("\nALL TESTS WERE SKIPPED.")
    elif passed_count == 0 and failed_count == 0 and skipped_count < len(tests):
        print("\nNO TESTS FAILED, but some tests may not have run as expected.")
    else: # Only skipped tests or no tests defined/run
        print("\nNO TESTS FAILED (excluding skipped).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QA tests using OpenAI's multimodal API and YAML config.")
    parser.add_argument("--config", default="test_config.yml", help="Path to the YAML test configuration file.")
    args = parser.parse_args()
    # Import urljoin and urlunparse for constructing prod URL
    from urllib.parse import urljoin, urlunparse
    main(args.config)