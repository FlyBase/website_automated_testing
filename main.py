import os
import sys
import time
import json
import base64
import yaml
import json
import difflib
import argparse
from urllib.parse import urlparse, urldefrag, urljoin, urlunparse

# Updated OpenAI imports
from openai import OpenAI, OpenAIError

# Selenium Imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    JavascriptException, NoSuchElementException, TimeoutException,
    ElementClickInterceptedException, StaleElementReferenceException
)

###############################################################################
# CONFIGURATION AND HELPERS
###############################################################################

SYSTEM_PROMPT_CONTENT = """
You are a meticulous QA automation assistant evaluating web page tests for FlyBase.org, a Drosophila genomics database.
Your task is to determine if a test passes or fails based on a user prompt and the provided evidence (text content and/or screenshots).

You will receive:
1.  **User Prompt:** Describes the specific condition to verify (e.g., check for errors, verify content presence/absence, check visual elements, compare states). **This is the primary requirement.**
2.  **Text Content:** Provided in ``` code blocks, potentially labeled "Production Text", "Preview State Text" (for comparisons), or "Preview Page Text" (for single page tests). Use this to verify textual requirements from the prompt.
3.  **Screenshots:** Provided as images, potentially labeled "Production Screenshot" and "Preview State Screenshot". Use these to verify visual requirements from the prompt (layout, specific elements, colors, absence of errors). Note: Screenshot capture might occasionally fail; proceed with text evaluation if possible.

Evaluation Guidelines:
- **Focus on the User Prompt:** Base your pass/fail decision strictly on whether the evidence meets the criteria defined in the prompt.
- **Comparison Tests (Production vs. Preview):** Carefully compare the Production and Preview text/screenshots. Look for differences relevant to the prompt's goal. Minor rendering variations (e.g., slight spacing, anti-aliasing) are usually acceptable unless the prompt specifically targets them. A failure occurs if Preview shows an incorrect state (e.g., an error, missing data, wrong visual) compared to Production OR compared to the explicit expectation in the prompt.
- **Single Page Tests:** Evaluate the provided "Preview Page Text" and/or "Preview State Screenshot" directly against the criteria in the user prompt.
- **Text Checks:** Look for specific error messages, expected text strings, or the absence of specific text as required by the prompt.
- **Image Checks:** Visually inspect screenshots for layout correctness, presence/absence of UI elements, correct data display (e.g., colored boxes as described), error messages, etc., according to the prompt.
- **Action Failures:** If the user provides a note like "*** NOTE: An action failed before capture... ***", consider this when evaluating the state. The state might be incorrect due to the failed action. Evaluate if the *resulting* state (even if incomplete) meets the prompt's criteria or if the action failure itself constitutes a test failure based on the prompt.

Output Requirements (Function Call):
- Call the `record_test_result` function.
- `result`: Must be "pass" or "fail".
- `failed_component`: If failed, specify "text", "image", "both", "page load", or "action" based on the primary reason for failure identified from the evidence and prompt. If passed, use "none".
- `explanation`: Provide a clear, concise explanation for your decision, directly referencing the prompt and the specific evidence (text or visual element) that led to the pass or fail status. E.g., "Test failed because the 'Something is broken' text was found in the Preview State Text." or "Test passed as the Preview State Screenshot shows blue boxes in the GO ribbon as required by the prompt." Avoid generic explanations.
"""

ARTIFACTS_DIR = "/app/artifacts"
PROD_BASE_URL = "https://flybase.org"
DEFAULT_POST_LOAD_WAIT_S = 1.5
CLICK_WAIT_TIMEOUT_S = 10 # Timeout for finding/waiting for elements
DEFAULT_WAIT_AFTER_CLICK_S = 1.0
DEFAULT_WAIT_AFTER_INPUT_S = 0.5
DEFAULT_WAIT_AFTER_KEY_S = 1.0


API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    sys.exit("Please set the OPENAI_API_KEY environment variable...")

client = OpenAI()
MODEL_NAME = "gpt-4.1-mini"

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
                    "enum": ["text", "image", "both", "none", "page load", "action"] # Added 'action'
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
    """Setup Selenium with headless Chrome/Chromium, and inject WAF header."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(1280, 1024) # Fixed window size

    waf_secret = os.getenv("WAF_SECRET_HEADER")
    if waf_secret:
        try:
            driver.execute_cdp_cmd("Network.enable", {})
            driver.execute_cdp_cmd("Network.setExtraHTTPHeaders", {"headers": {"x-waf-secret": waf_secret}})
            print("WAF secret header set successfully.")
        except Exception as e:
            print(f"Warning: Could not set WAF secret header: {e}")

    return driver

def wait_for_page_conditions(driver, url_loaded, extra_wait_s=None):
    """Handles waiting for readyState, fragment scroll, and applies a fixed wait."""
    WebDriverWait(driver, 25).until(lambda d: d.execute_script('return document.readyState') == 'complete')
    print(f"    Page '{url_loaded}' loaded (readyState complete).")

    scroll_wait = 1.0
    _base, fragment = urldefrag(url_loaded)
    if fragment:
        try:
            print(f"    Fragment '#{fragment}' detected. Attempting to scroll into view.")
            target_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, fragment))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", target_element)
            print(f"    Executed scrollIntoView for ID '{fragment}'.")
            time.sleep(scroll_wait)
        except (NoSuchElementException, TimeoutException):
             print(f"    Warning: Could not find/scroll to element with ID '{fragment}'.")
        except JavascriptException as js_err:
             print(f"    Warning: JavaScript error while scrolling to '#{fragment}': {js_err}")
        except Exception as scroll_err:
             print(f"    Warning: Error during scrolling attempt for '#{fragment}': {scroll_err}")

    wait_duration = DEFAULT_POST_LOAD_WAIT_S
    if extra_wait_s is not None:
        try:
            custom_wait = float(extra_wait_s)
            if custom_wait >= 0:
                wait_duration = custom_wait
                print(f"    Applying custom extra wait: {wait_duration}s")
            else:
                print(f"    Warning: Invalid negative extra_wait_s value ({extra_wait_s}), using default: {wait_duration}s")
        except (ValueError, TypeError):
            print(f"    Warning: Invalid non-numeric extra_wait_s value ('{extra_wait_s}'), using default: {wait_duration}s")
    else:
        print(f"    Applying default extra wait: {wait_duration}s")

    if wait_duration > 0:
        time.sleep(wait_duration)


def perform_actions(driver, actions):
    """ Performs a list of actions defined in the YAML, scrolling element into view first. """
    if not actions:
        return True

    print("    Performing actions before capture...")
    for i, action_def in enumerate(actions):
        action_type = action_def.get("action")
        locate_by = action_def.get("locate_by")
        value = action_def.get("value")
        # Get specific default wait based on action type
        default_wait = DEFAULT_WAIT_AFTER_CLICK_S
        if action_type == "input_text": default_wait = DEFAULT_WAIT_AFTER_INPUT_S
        if action_type == "press_key": default_wait = DEFAULT_WAIT_AFTER_KEY_S

        wait_after_s = action_def.get("wait_after_s", default_wait)
        action_desc = f"Action #{i+1} ({action_type}"
        if locate_by: action_desc += f": {locate_by}='{value}'"
        action_desc += ")"

        # --- Map locator strategy ---
        locator_strategy = None
        if locate_by:
            # ... (locator mapping remains the same) ...
            if locate_by == "css_selector": locator_strategy = By.CSS_SELECTOR
            elif locate_by == "xpath": locator_strategy = By.XPATH
            elif locate_by == "id": locator_strategy = By.ID
            elif locate_by == "link_text": locator_strategy = By.LINK_TEXT
            elif locate_by == "partial_link_text": locator_strategy = By.PARTIAL_LINK_TEXT
            elif locate_by == "class_name": locator_strategy = By.CLASS_NAME
            elif locate_by == "tag_name": locator_strategy = By.TAG_NAME
            else:
                print(f"    ERROR: Invalid 'locate_by' value ('{locate_by}') for {action_desc}.")
                return False
        elif action_type in ["click", "input_text", "press_key"]:
             print(f"    ERROR: Missing 'locate_by' or 'value' for required action '{action_type}' ({action_desc}).")
             return False

        # --- Preliminary Find and Scroll ---
        element_to_interact = None
        if locator_strategy:
            try:
                print(f"      Ensuring element is present for scrolling: {locate_by}='{value}'")
                preliminary_element = WebDriverWait(driver, CLICK_WAIT_TIMEOUT_S / 2).until(
                     EC.presence_of_element_located((locator_strategy, value))
                )
                print(f"      Scrolling element into view (center): {locate_by}='{value}'")
                driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", preliminary_element)
                time.sleep(0.3)
                element_to_interact = preliminary_element
            except TimeoutException:
                print(f"    ERROR: Timed out finding element *before* interaction/scroll for {action_desc}.")
                return False
            except Exception as scroll_err:
                print(f"    ERROR: Could not find or scroll to element for {action_desc}: {scroll_err}")
                return False

        # --- Action Execution (with waits for interactability) ---
        try:
            element = None
            if action_type == "click":
                print(f"      Waiting for element to be clickable: {locate_by}='{value}' (Timeout: {CLICK_WAIT_TIMEOUT_S}s)")
                element = WebDriverWait(driver, CLICK_WAIT_TIMEOUT_S).until(
                    EC.element_to_be_clickable((locator_strategy, value))
                )
                print(f"      Clicking element: {locate_by}='{value}'")
                element.click()

            elif action_type == "input_text":
                text_to_input = action_def.get("text")
                if text_to_input is None:
                    print(f"    ERROR: Missing 'text' field for input_text action ({action_desc}).")
                    return False
                # Wait for visibility *after* scrolling
                print(f"      Waiting for element visibility: {locate_by}='{value}' (Timeout: {CLICK_WAIT_TIMEOUT_S}s)")
                element = WebDriverWait(driver, CLICK_WAIT_TIMEOUT_S).until(
                    EC.visibility_of_element_located((locator_strategy, value))
                )

                # --- Set value using JavaScript --- <<< MODIFICATION HERE >>>
                print(f"      Setting input value via JavaScript: {locate_by}='{value}'")
                # Use json.dumps to safely escape the text for JavaScript
                js_escaped_text = json.dumps(text_to_input)
                driver.execute_script(f"arguments[0].value = {js_escaped_text};", element)

                # Optional: Trigger change/input events if the page/typeahead needs them
                # If setting value works but pressing Enter later fails, uncomment these.
                # print("      Triggering 'input' and 'change' events after JS set.")
                # driver.execute_script("arguments[0].dispatchEvent(new Event('input', { bubbles: true }));", element)
                # driver.execute_script("arguments[0].dispatchEvent(new Event('change', { bubbles: true }));", element)
                # ----------------------------------

            elif action_type == "press_key":
                key_name = action_def.get("key")
                if not key_name:
                     print(f"    ERROR: Missing 'key' field for press_key action ({action_desc}).")
                     return False

                key_to_press = None
                if key_name.upper() == "ENTER": key_to_press = Keys.ENTER
                elif key_name.upper() == "TAB": key_to_press = Keys.TAB
                else:
                    print(f"    ERROR: Unsupported key name '{key_name}' for {action_desc}.")
                    return False

                print(f"      Waiting for element visibility: {locate_by}='{value}' (Timeout: {CLICK_WAIT_TIMEOUT_S}s)")
                element = WebDriverWait(driver, CLICK_WAIT_TIMEOUT_S).until(
                    EC.visibility_of_element_located((locator_strategy, value))
                )
                print(f"      Pressing key '{key_name}' in: {locate_by}='{value}'")
                element.send_keys(key_to_press)

            else:
                print(f"    WARNING: Unsupported action type '{action_type}' in {action_desc}. Skipping.")
                continue

            # Wait after successful action if specified
            if wait_after_s > 0:
                print(f"      Waiting {wait_after_s}s after {action_type}...")
                time.sleep(wait_after_s)

        # --- Error Handling ---
        # ... (Error handling remains the same as previous version) ...
        except TimeoutException:
            print(f"    ERROR: Timed out waiting for element state (clickable/visible) for {action_desc}.")
            return False
        except (NoSuchElementException, StaleElementReferenceException) as e:
            print(f"    ERROR: Element not found or stale during interaction for {action_desc}: {e}")
            return False
        except ElementClickInterceptedException:
             print(f"    ERROR: Element click intercepted for {action_desc}. Trying JavaScript click.")
             try:
                 element = driver.find_element(locator_strategy, value)
                 driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", element)
                 time.sleep(0.2)
                 driver.execute_script("arguments[0].click();", element)
                 print("      JavaScript click executed.")
                 if wait_after_s > 0: time.sleep(wait_after_s)
             except Exception as js_e:
                 print(f"    ERROR: JavaScript click also failed for {action_desc}: {js_e}")
                 return False
        except Exception as e:
            print(f"    ERROR: Unexpected error during {action_desc}: {e}")
            return False

    print("    Finished performing actions.")
    return True

# <<< Other functions (capture_page_text, capture_page_screenshot, etc.) remain the same >>>
def capture_page_text(driver):
    try:
        body_element = driver.find_element(By.XPATH, "//body")
        return body_element.text
    except Exception as e:
        print(f"    Error capturing text from current page state: {e}")
        return f"Error capturing text: {e}"

def capture_page_screenshot(driver, screenshot_filename_with_path):
    try:
        os.makedirs(os.path.dirname(screenshot_filename_with_path), exist_ok=True)
        driver.save_screenshot(screenshot_filename_with_path)
        print(f"    Saved screenshot artifact: {screenshot_filename_with_path}")
        with open(screenshot_filename_with_path, "rb") as img_file:
            return img_file.read()
    except Exception as e:
        print(f"    Error capturing or reading screenshot {screenshot_filename_with_path} (current state): {e}")
        return None

def encode_image_to_data_uri(image_bytes, image_format="png"):
    if image_bytes is None: return ""
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{image_format};base64,{image_b64}"

def load_yaml_config(config_path):
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
                                failed_component = result_json.get("failed_component", "action" if result_status == "fail" else "none")
                                explanation = result_json.get("explanation", "No explanation.")
                                return {"result": result_status, "failed_component": failed_component, "explanation": explanation}
                            else: explanation = f"Parsed JSON missing required fields ('result','explanation'). Parsed: {result_json}"
                        except json.JSONDecodeError as e: explanation = f"Error parsing args JSON: {e}. Args: ```{arguments_str}```"
                elif message.content: explanation = f"API returned content instead of function call: {message.content}"; failed_component = "api_response_format"
                else: explanation = "API response lacked function call or content."; failed_component = "api_response_format"
        except (AttributeError, IndexError, KeyError, TypeError) as e: explanation = f"Error accessing API response structure: {e}. Response: {response}"; failed_component = "api_response_parsing"
        if result_status == 'fail': print(f"Warning: API Response issue: {explanation}")
    return {"result": result_status, "failed_component": failed_component, "explanation": explanation}


def run_test(driver, test_def):
    """
    Execute a test: navigate, wait, perform actions, capture, analyze.
    Returns a dict with test name, prompt, result, failed_component, and explanation.
    """
    # --- Test Definition Extraction ---
    test_name = test_def.get("name", "Unnamed Test")
    safe_test_name = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in test_name).rstrip('_')
    prompt = test_def.get("prompt", "")
    extra_wait_s = test_def.get("extra_wait_s")
    enabled = test_def.get("enabled", True)
    url_from_yaml = test_def.get("url")
    compare_to_production = test_def.get("compare_to_production", False)
    check_types = test_def.get("check_types", [])
    ticket = test_def.get("ticket", "")
    actions_to_perform = test_def.get("actions_before_capture", [])

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
    effective_wait = extra_wait_s if extra_wait_s is not None else DEFAULT_POST_LOAD_WAIT_S
    print(f"Effective Initial Post-Load Wait: {effective_wait}s")
    if compare_to_production and prod_url:
        print(f"Comparing against Production URL: {prod_url}")
    else:
        print("Running as single page test (no production comparison).")
        compare_to_production = False
    if actions_to_perform:
        print(f"Actions to perform before capture: {len(actions_to_perform)}")

    # --- Prepare for OpenAI API Call ---
    messages = [{"role": "system", "content": SYSTEM_PROMPT_CONTENT}] # Use the constant defined at the top
    user_content_parts = []
    screenshots_data = []
    prompt_display = f"Test: {test_name}\nPrompt: {prompt}"
    if ticket: prompt_display += f"\nReference Ticket: {ticket}"
    user_content_parts.append({"type": "text", "text": prompt_display})

    # --- Main Test Logic ---
    action_failure = False
    try:
        if compare_to_production: # This implies prod_url is valid
            user_content_parts.append({"type": "text", "text": f"\n--- Comparing Production vs Preview ---"})
            user_content_parts.append({"type": "text", "text": f"Production URL: {prod_url}"})
            user_content_parts.append({"type": "text", "text": f"Preview URL: {preview_url}"})

            # --- Production Data Capture ---
            print("  -- Processing Production URL --")
            driver.get(prod_url)
            wait_for_page_conditions(driver, prod_url, extra_wait_s)
            if actions_to_perform:
                 if not perform_actions(driver, actions_to_perform):
                      action_failure = True
                      print("    WARNING: Action failed on Production, capturing state anyway.")

            if "text" in check_types and not action_failure:
                prod_text = capture_page_text(driver)
                prod_text_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_prod.txt")
                try:
                    with open(prod_text_path, "w", encoding="utf-8") as f: f.write(prod_text)
                    print(f"    Saved text artifact: {prod_text_path}")
                except Exception as e: print(f"    Warning: Could not save prod text file: {e}")
                user_content_parts.append({"type": "text", "text": f"\nProduction Text:\n```\n{prod_text}\n```"})
            elif "text" in check_types and action_failure:
                 user_content_parts.append({"type": "text", "text": "\nProduction Text: (Skipped due to action failure)"})

            if "picture" in check_types:
                prod_screenshot_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_prod.png")
                prod_screenshot_bytes = capture_page_screenshot(driver, prod_screenshot_path)
                prod_img_uri = encode_image_to_data_uri(prod_screenshot_bytes)
                if prod_img_uri:
                    screenshots_data.append({"type": "image_url", "image_url": {"url": prod_img_uri}})
                    user_content_parts.append({"type": "text", "text": "\nProduction Screenshot:"})
                else: print(f"    Warning: Failed to get Production screenshot")

            # --- Preview Data Capture ---
            print("  -- Processing Preview URL --")
            driver.get(preview_url)
            wait_for_page_conditions(driver, preview_url, extra_wait_s)
            if actions_to_perform:
                 if not perform_actions(driver, actions_to_perform):
                      action_failure = True # Mark failure but continue capture
                      print("    WARNING: Action failed on Preview, capturing state anyway.")

            if "text" in check_types and not action_failure:
                preview_text = capture_page_text(driver)
                preview_text_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_preview.txt")
                try:
                     with open(preview_text_path, "w", encoding="utf-8") as f: f.write(preview_text)
                     print(f"    Saved text artifact: {preview_text_path}")
                except Exception as e: print(f"    Warning: Could not save preview text file: {e}")
                user_content_parts.append({"type": "text", "text": f"\nPreview State Text:\n```\n{preview_text}\n```"})
            elif "text" in check_types and action_failure:
                 user_content_parts.append({"type": "text", "text": "\nPreview Text: (Skipped due to action failure)"})

            if "picture" in check_types:
                preview_screenshot_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_preview.png")
                preview_screenshot_bytes = capture_page_screenshot(driver, preview_screenshot_path)
                preview_img_uri = encode_image_to_data_uri(preview_screenshot_bytes)
                if preview_img_uri:
                    screenshots_data.append({"type": "image_url", "image_url": {"url": preview_img_uri}})
                    user_content_parts.append({"type": "text", "text": "\nPreview State Screenshot:"})
                else: print(f"    Warning: Failed to get Preview screenshot")

        # --- Single Page Test Logic ---
        else: # Only runs if compare_to_production is False
            user_content_parts.append({"type": "text", "text": f"\n--- Single Page Test ---"})
            user_content_parts.append({"type": "text", "text": f"URL Tested: {preview_url}"})

            driver.get(preview_url)
            wait_for_page_conditions(driver, preview_url, extra_wait_s)
            if actions_to_perform:
                if not perform_actions(driver, actions_to_perform):
                     action_failure = True
                     print("    WARNING: Action failed on Single Page, proceeding with capture.")

            # --- Capture Text ---
            if "text" in check_types and not action_failure:
                page_text = capture_page_text(driver)
                text_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_preview.txt") # Use preview suffix
                try:
                    with open(text_path, "w", encoding="utf-8") as f: f.write(page_text)
                    print(f"    Saved text artifact: {text_path}")
                except Exception as e: print(f"    Warning: Could not save text file: {e}")
                user_content_parts.append({"type": "text", "text": f"\nPreview Page Text:\n```\n{page_text}\n```"})
            elif "text" in check_types and action_failure:
                 user_content_parts.append({"type": "text", "text": "\nPreview Page Text: (Skipped due to action failure)"})

            # --- Capture Screenshot (as _preview.png) ---
            if "picture" in check_types:
                preview_screenshot_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_preview.png") # Use preview suffix
                preview_screenshot_bytes = capture_page_screenshot(driver, preview_screenshot_path)
                preview_img_uri = encode_image_to_data_uri(preview_screenshot_bytes)
                if preview_img_uri:
                    screenshots_data.append({"type": "image_url", "image_url": {"url": preview_img_uri}})
                    user_content_parts.append({"type": "text", "text": "\nPreview State Screenshot:"}) # Consistent label
                else:
                    print(f"    Warning: Failed to get screenshot for single page test {test_name}")

        # --- Handle Action Failure Reporting ---
        if action_failure:
             user_content_parts.append({"type": "text", "text": "\n*** NOTE: An action failed before capture. The text/screenshot may reflect the state BEFORE the failed action completed. ***"})

    except Exception as page_err:
        error_msg = f"Failed during page load, action, or data capture for test '{test_name}'. URL(s): {preview_url}";
        if prod_url: error_msg += f", {prod_url}"
        error_msg += f". Error: {page_err}"; print(f"    ERROR: {error_msg}")
        try:
            fail_screenshot_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_FAILURE_capture.png")
            capture_page_screenshot(driver, fail_screenshot_path)
        except Exception as screen_err: print(f"    Could not take failure screenshot: {screen_err}")
        return {"test_name": test_name, "result": "fail", "failed_component": "page load/action/capture", "explanation": error_msg, "prompt": prompt}

    # --- Finalize and Call OpenAI API ---
    if "picture" in check_types and not screenshots_data:
        print(f"Note for test '{test_name}': Proceeding without images as capture failed/returned None.")

    user_message_content_list = user_content_parts + screenshots_data
    messages.append({"role": "user", "content": user_message_content_list})
    print("Prompt snippet for LLM call:", prompt_display[:200], "...\n")

    response = call_openai_api(messages)
    result_data = parse_api_response(response)

    return {"test_name": test_name, "prompt": prompt, **result_data}


# <<< main function remains unchanged >>>
def main(config_path):
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
