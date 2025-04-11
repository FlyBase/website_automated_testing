import os
import sys
import time
import json
import base64
import yaml
import difflib
import argparse
# Make sure urlparse is imported
from urllib.parse import urlparse, urldefrag, urljoin, urlunparse

# Updated OpenAI imports
from openai import OpenAI, OpenAIError

# Simplified Selenium Imports (only for navigation and capture)
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# Import exceptions - TimeoutException might not be needed anymore but keep others
from selenium.common.exceptions import JavascriptException, NoSuchElementException, TimeoutException

###############################################################################
# CONFIGURATION AND HELPERS
###############################################################################

# Define the path for artifacts inside the container
ARTIFACTS_DIR = "/app/artifacts"
# Define the base Production URL for comparison purposes
PROD_BASE_URL = "https://flybase.org"
# <<< Define default fixed wait time in seconds after page load/scroll >>>
DEFAULT_POST_LOAD_WAIT_S = 1.5

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    sys.exit("Please set the OPENAI_API_KEY environment variable inside the container (e.g. -e OPENAI_API_KEY='sk-...').")

# Initialize OpenAI client (v1.x)
client = OpenAI()
MODEL_NAME = "gpt-4o-mini"

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

    waf_secret = os.getenv("WAF_SECRET_HEADER")
    if waf_secret:
        try:
            driver.execute_cdp_cmd("Network.enable", {})
            driver.execute_cdp_cmd("Network.setExtraHTTPHeaders", {"headers": {"x-waf-secret": waf_secret}})
            print("WAF secret header set successfully.")
        except Exception as e:
            print(f"Warning: Could not set WAF secret header: {e}")

    return driver

# <<< Modified Helper Function for Waits >>>
def wait_for_page_conditions(driver, url, extra_wait_s=None):
    """Handles waiting for page load, fragment scroll, and applies a fixed wait."""
    # 1. Wait for basic page load state (increased timeout slightly)
    WebDriverWait(driver, 25).until(lambda d: d.execute_script('return document.readyState') == 'complete')
    print(f"    Page '{url}' loaded (readyState complete).")

    # 2. Handle fragment scrolling if present
    scroll_wait = 1.0 # Time to wait after attempting scroll
    _base, fragment = urldefrag(url)
    if fragment:
        try:
            print(f"    Fragment '#{fragment}' detected. Attempting to scroll into view.")
            # Wait slightly longer for element presence if it might be dynamic
            target_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, fragment))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", target_element)
            print(f"    Executed scrollIntoView for ID '{fragment}'.")
            time.sleep(scroll_wait) # Pause after scroll command
        except (NoSuchElementException, TimeoutException):
             print(f"    Warning: Could not find/scroll to element with ID '{fragment}'.")
             # Still proceed to fixed wait even if scroll fails
        except JavascriptException as js_err:
             print(f"    Warning: JavaScript error while scrolling to '#{fragment}': {js_err}")
        except Exception as scroll_err:
             print(f"    Warning: Error during scrolling attempt for '#{fragment}': {scroll_err}")

    # 3. Apply fixed wait (either default or override)
    wait_duration = DEFAULT_POST_LOAD_WAIT_S # Start with default
    if extra_wait_s is not None:
        try:
            # Ensure the value is a valid positive number
            custom_wait = float(extra_wait_s)
            if custom_wait >= 0:
                wait_duration = custom_wait
                print(f"    Applying custom extra wait: {wait_duration}s")
            else:
                print(f"    Warning: Invalid negative extra_wait_s value ({extra_wait_s}), using default: {wait_duration}s")
        except (ValueError, TypeError):
            print(f"    Warning: Invalid non-numeric extra_wait_s value ('{extra_wait_s}'), using default: {wait_duration}s")
    else:
        # Only print default if no custom value was attempted
        print(f"    Applying default extra wait: {wait_duration}s")

    # Execute the final wait
    if wait_duration > 0:
        time.sleep(wait_duration)


# <<< Modified Function Signature >>>
def get_page_text(driver, url, extra_wait_s=None):
    """Fetch the entire visible text from the page body, handling waits."""
    try:
        print(f"    Getting text from: {url}")
        driver.get(url)
        # <<< Call unified wait function with extra_wait_s >>>
        wait_for_page_conditions(driver, url, extra_wait_s)

        # Proceed with text extraction
        body_element = driver.find_element(By.XPATH, "//body")
        return body_element.text
    except Exception as e:
        print(f"    Error extracting text from {url}: {e}")
        return f"Error extracting text: {e}"


# <<< Modified Function Signature >>>
def get_page_screenshot(driver, url, screenshot_filename_with_path, extra_wait_s=None):
    """Capture screenshot, handling waits. Pass url=None to screenshot current state."""
    try:
        action = f"screenshot for: {url}" if url else "screenshot of current state"
        print(f"    Getting {action}")
        if url:
            driver.get(url)
             # <<< Call unified wait function with extra_wait_s >>>
            wait_for_page_conditions(driver, url, extra_wait_s)
        else:
             print("    Screenshotting current state (no navigation/wait).")
             time.sleep(0.5) # Keep small pause for current state screenshots

        # Proceed to take screenshot
        os.makedirs(os.path.dirname(screenshot_filename_with_path), exist_ok=True)
        driver.save_screenshot(screenshot_filename_with_path)
        print(f"    Saved screenshot artifact: {screenshot_filename_with_path}")
        with open(screenshot_filename_with_path, "rb") as img_file:
            return img_file.read()
    except Exception as e:
        url_info = f"for URL {url}" if url else "(current state)"
        print(f"    Error capturing or reading screenshot {screenshot_filename_with_path} {url_info}: {e}")
        return None


def encode_image_to_data_uri(image_bytes, image_format="png"):
    """Encode image bytes in base64 with data URI format."""
    if image_bytes is None: return ""
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{image_format};base64,{image_b64}"

def load_yaml_config(config_path):
    """Load tests from a YAML configuration file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        tests = config.get("tests")
        if tests is None:
             print(f"Warning: 'tests' key not found in {config_path}. Returning empty list.")
             return []
        if not isinstance(tests, list):
            print(f"Warning: 'tests' key in {config_path} is not a list. Returning empty list.")
            return []
        return tests
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
         print(f"Error reading configuration file {config_path}: {e}")
         sys.exit(1)

def call_openai_api(messages):
    """Call OpenAI's ChatCompletion API using function calling (v1.x API)."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, functions=FUNCTION_SCHEMA,
            function_call={"name": "record_test_result"}, temperature=0
        )
        return response
    except OpenAIError as e:
        print(f"OpenAI API call error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error calling OpenAI API: {e}")
        return None

def parse_api_response(response):
    """Parse the response from OpenAI - returns a dict with result, failed_component, explanation."""
    result_status = "fail"; failed_component = "unknown"; explanation = "Could not parse API response or API call failed."
    if not response: explanation = "No API response received (API call likely failed)."
    else:
        try:
            if not response.choices: explanation = "API response 'choices' field missing/empty."
            else:
                message = response.choices[0].message
                if not message: explanation = "Message field missing in first choice."
                elif message.function_call:
                    arguments_str = message.function_call.arguments
                    if not arguments_str: explanation = "Function call arguments missing."
                    else:
                        try:
                            result_json = json.loads(arguments_str)
                            if "result" in result_json and "explanation" in result_json:
                                result_status = result_json.get("result", "fail")
                                failed_component = result_json.get("failed_component", "none" if result_status == "pass" else "unknown")
                                explanation = result_json.get("explanation", "No explanation.")
                                return {"result": result_status, "failed_component": failed_component, "explanation": explanation}
                            else: explanation = f"Parsed JSON missing required fields ('result','explanation'). Parsed: {result_json}"
                        except json.JSONDecodeError as e: explanation = f"Error parsing args JSON: {e}. Args: ```{arguments_str}```"
                elif message.content: explanation = f"API returned content instead of function call: {message.content}"; failed_component = "api_response_format"
                else: explanation = "API response lacked function call or content."; failed_component = "api_response_format"
        except (AttributeError, IndexError, KeyError, TypeError) as e: explanation = f"Error accessing API response structure: {e}. Response: {response}"; failed_component = "api_response_parsing"
        if result_status == 'fail': print(f"Warning: API Response issue: {explanation}") # Log issues if parsing fails
    return {"result": result_status, "failed_component": failed_component, "explanation": explanation}


def run_test(driver, test_def):
    """
    Execute a basic test (navigate, capture) based on YAML config.
    Returns a dict with test name, prompt, result, failed_component, and explanation.
    """
    # --- Test Definition Extraction ---
    test_name = test_def.get("name", "Unnamed Test")
    safe_test_name = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in test_name).rstrip('_')
    prompt = test_def.get("prompt", "")
    # <<< Get the specific extra wait time for this test >>>
    extra_wait_s = test_def.get("extra_wait_s") # Will be None if not specified
    enabled = test_def.get("enabled", True)
    url_from_yaml = test_def.get("url")
    compare_to_production = test_def.get("compare_to_production", False)
    check_types = test_def.get("check_types", [])
    ticket = test_def.get("ticket", "")

    # --- Pre-checks ---
    if not enabled:
        print(f"Skipping disabled test: {test_name}")
        return {"test_name": test_name, "result": "skipped", "failed_component": "none", "explanation": "Test is disabled.", "prompt": prompt}
    if not url_from_yaml:
        print(f"Skipping test '{test_name}': Missing required 'url' field in YAML.")
        return {"test_name": test_name, "result": "skipped", "failed_component": "none", "explanation": "Missing 'url'.", "prompt": prompt}

    preview_url = url_from_yaml
    prod_url = None

    # --- URL Setup ---
    if compare_to_production:
        try:
            parsed_target = urlparse(preview_url)
            path = parsed_target.path if parsed_target.path else "/"
            prod_path_query_fragment = urlunparse(('', '', path, parsed_target.params, parsed_target.query, parsed_target.fragment))
            prod_url = urljoin(PROD_BASE_URL.rstrip('/') + '/', prod_path_query_fragment.lstrip('/'))
        except Exception as e:
            print(f"Warning: Could not construct production URL for comparison from {preview_url}: {e}")
            compare_to_production = False
            prod_url = None

    print(f"\n=== Running Test: {test_name} ===")
    print(f"Preview URL: {preview_url}")
    # Log the wait time being used (custom or default inferred)
    effective_wait = extra_wait_s if extra_wait_s is not None else DEFAULT_POST_LOAD_WAIT_S
    print(f"Effective Post-Load Wait: {effective_wait}s")
    if compare_to_production and prod_url:
        print(f"Comparing against Production URL: {prod_url}")

    # --- Prepare for OpenAI API Call ---
    system_content = "You are a meticulous QA automation assistant..."
    messages = [{"role": "system", "content": system_content}]
    user_content_parts = []
    screenshots_data = []

    prompt_display = f"Test: {test_name}\nPrompt: {prompt}"
    if ticket: prompt_display += f"\nReference Ticket: {ticket}"
    user_content_parts.append({"type": "text", "text": prompt_display})

    # --- Main Test Logic (Capture Data) ---
    try:
        if compare_to_production and prod_url:
            user_content_parts.append({"type": "text", "text": f"\n--- Comparing Production vs Preview ---"})
            user_content_parts.append({"type": "text", "text": f"Production URL: {prod_url}"})
            user_content_parts.append({"type": "text", "text": f"Preview URL: {preview_url}"})

            # Get text data (passing extra_wait_s)
            if "text" in check_types:
                prod_text = get_page_text(driver, prod_url, extra_wait_s)
                preview_text = get_page_text(driver, preview_url, extra_wait_s)
                prod_text_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_prod.txt")
                preview_text_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_preview.txt")
                try:
                    with open(prod_text_path, "w", encoding="utf-8") as f: f.write(prod_text)
                    with open(preview_text_path, "w", encoding="utf-8") as f: f.write(preview_text)
                    print(f"Saved text artifacts: {prod_text_path}, {preview_text_path}")
                except Exception as e: print(f"Warning: Could not save text files for {test_name}: {e}")
                user_content_parts.append({"type": "text", "text": f"\nProduction Text:\n```\n{prod_text}\n```"})
                user_content_parts.append({"type": "text", "text": f"\nPreview State Text:\n```\n{preview_text}\n```"})

            # Get image data (passing extra_wait_s)
            if "picture" in check_types:
                prod_screenshot_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_prod.png")
                preview_screenshot_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_preview.png")
                prod_screenshot_bytes = get_page_screenshot(driver, prod_url, prod_screenshot_path, extra_wait_s)
                preview_screenshot_bytes = get_page_screenshot(driver, preview_url, preview_screenshot_path, extra_wait_s)
                prod_img_uri = encode_image_to_data_uri(prod_screenshot_bytes)
                preview_img_uri = encode_image_to_data_uri(preview_screenshot_bytes)
                if prod_img_uri:
                    screenshots_data.append({"type": "image_url", "image_url": {"url": prod_img_uri}})
                    user_content_parts.append({"type": "text", "text": "\nProduction Screenshot:"})
                else: print(f"Warning: Failed to get Production screenshot for {test_name}")
                if preview_img_uri:
                    screenshots_data.append({"type": "image_url", "image_url": {"url": preview_img_uri}})
                    user_content_parts.append({"type": "text", "text": "\nPreview State Screenshot:"})
                else: print(f"Warning: Failed to get Preview screenshot for {test_name}")

        else: # Single page test (using preview_url)
            user_content_parts.append({"type": "text", "text": f"\n--- Single Page Test ---"})
            user_content_parts.append({"type": "text", "text": f"URL Tested: {preview_url}"})

            if "text" in check_types:
                page_text = get_page_text(driver, preview_url, extra_wait_s)
                text_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_single.txt")
                try:
                    with open(text_path, "w", encoding="utf-8") as f: f.write(page_text)
                    print(f"Saved text artifact: {text_path}")
                except Exception as e: print(f"Warning: Could not save text file for {test_name}: {e}")
                user_content_parts.append({"type": "text", "text": f"\nPage Text:\n```\n{page_text}\n```"})

            if "picture" in check_types:
                screenshot_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_single.png")
                screenshot_bytes = get_page_screenshot(driver, preview_url, screenshot_path, extra_wait_s)
                img_uri = encode_image_to_data_uri(screenshot_bytes)
                if img_uri:
                    screenshots_data.append({"type": "image_url", "image_url": {"url": img_uri}})
                    user_content_parts.append({"type": "text", "text": "\nPage Screenshot:"})
                else: print(f"Warning: Failed to get single screenshot for {test_name}")

    except Exception as page_err:
        error_msg = f"Failed during page load or data capture for test '{test_name}'. URL(s): {preview_url}";
        if prod_url: error_msg += f", {prod_url}"
        error_msg += f". Error: {page_err}"; print(f"    ERROR: {error_msg}")
        try:
            fail_screenshot_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_FAILURE_capture.png")
            get_page_screenshot(driver, None, fail_screenshot_path, None) # Pass None for URL/wait
        except Exception as screen_err: print(f"    Could not take failure screenshot: {screen_err}")
        return {"test_name": test_name, "result": "fail", "failed_component": "page load/capture", "explanation": error_msg, "prompt": prompt}

    # --- Finalize and Call OpenAI API ---
    if "picture" in check_types and not screenshots_data:
        print(f"Note for test '{test_name}': Proceeding without images as capture failed/returned None.")
        # Optionally add note to explanation or force fail?
        # explanation_note = "\n[Note: Screenshot capture failed]"

    user_message_content_list = user_content_parts + screenshots_data
    messages.append({"role": "user", "content": user_message_content_list})
    print("Prompt snippet for LLM call:", prompt_display[:200], "...\n")

    response = call_openai_api(messages)
    result_data = parse_api_response(response)
    # if explanation_note: result_data["explanation"] += explanation_note # Append note if needed

    # --- Combine and Return Final Result ---
    return {"test_name": test_name, "prompt": prompt, **result_data}


def main(config_path):
    """Main function to load config, set up driver, run tests, and save results."""
    try:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        print(f"Artifacts will be saved to container path: {ARTIFACTS_DIR}")
    except OSError as e: print(f"Error creating artifacts directory {ARTIFACTS_DIR}: {e}")

    tests = load_yaml_config(config_path)
    if not tests: print("No tests found/loaded."); return

    driver = None
    all_results = []
    try:
        driver = setup_selenium()
        for test_def in tests:
            if not isinstance(test_def, dict) or not test_def.get("name"):
                print(f"Warning: Skipping invalid test definition: {test_def}"); continue
            result = None
            try:
                result = run_test(driver, test_def)
            except Exception as err:
                test_name_fallback = test_def.get("name", "Unknown Test")
                prompt_fallback = test_def.get("prompt", "")
                print(f"Test '{test_name_fallback}' CRASHED during execution: {err}")
                import traceback; traceback.print_exc()
                result = {"test_name": test_name_fallback, "result": "fail", "failed_component": "test execution crash",
                          "explanation": f"Unhandled Exception CRASH: {err}", "prompt": prompt_fallback}
            if result: all_results.append(result)
    finally:
        if driver:
            try: driver.quit(); print("\nSelenium driver quit.")
            except Exception as q_err: print(f"Error quitting selenium driver: {q_err}")

    # --- Process Results ---
    results_filename = os.path.join(ARTIFACTS_DIR, "test_results.json")
    try:
        if all_results:
             with open(results_filename, "w", encoding="utf-8") as f: json.dump(all_results, f, indent=2)
             print(f"\nFull results saved to container path: {results_filename}")
        else: print("\nNo test results generated to save.")
    except Exception as e: print(f"\nError saving results to {results_filename}: {e}")

    # Calculate final counts from recorded results
    final_passed = sum(1 for r in all_results if r.get("result") == "pass")
    final_failed = sum(1 for r in all_results if r.get("result") == "fail")
    final_skipped = sum(1 for r in all_results if r.get("result") == "skipped")

    print("\n--- TEST SUMMARY ---")
    print(f"Total tests defined in config: {len(tests)}")
    print(f"Total results recorded: {len(all_results)}")
    print(f"Passed: {final_passed}"); print(f"Failed: {final_failed}"); print(f"Skipped: {final_skipped}")

    if final_failed > 0:
        print("\n--- FAILED TEST DETAILS ---")
        for res in all_results:
            if res.get("result") == "fail":
                print(f"  Test FAILED: {res.get('test_name')}")
                print(f"    Prompt: {res.get('prompt', 'N/A')}")
                print(f"    Failed Component: {res.get('failed_component', 'N/A')}")
                print(f"    Explanation: {res.get('explanation', 'No explanation.')}\n")
        print("-------------------------")

    if final_failed == 0 and final_passed > 0: print("\nALL EXECUTED TESTS PASSED.")
    elif final_failed == 0 and final_passed == 0 and final_skipped > 0: print("\nNO TESTS FAILED (only skipped tests).")
    elif final_failed == 0 and final_passed == 0 and final_skipped == 0: print("\nNO TESTS WERE EXECUTED OR RECORDED.")
    else: print(f"\n{final_failed} TEST(S) FAILED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QA tests using OpenAI's multimodal API and YAML config.")
    parser.add_argument("--config", default="test_config.yml", help="Path to the YAML test configuration file.")
    args = parser.parse_args()
    main(args.config)