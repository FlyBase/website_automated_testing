import os
import sys
import time
import json
import base64
import yaml
# import json # Duplicate
import difflib
import argparse
from urllib.parse import urlparse, urldefrag, urljoin, urlunparse

# Updated OpenAI imports
from openai import OpenAI, OpenAIError

# --- Playwright Imports ---
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError

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
DEFAULT_POST_LOAD_WAIT_S = 1.5
CLICK_WAIT_TIMEOUT_S = 15
DEFAULT_WAIT_AFTER_CLICK_S = 1.0
DEFAULT_WAIT_AFTER_INPUT_S = 0.5
DEFAULT_WAIT_AFTER_KEY_S = 1.0

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    sys.exit("Please set the OPENAI_API_KEY environment variable...")

client = OpenAI()
# <<< UPDATED MODEL NAME >>>
MODEL_NAME = "gpt-4.1-mini"

# Function schema remains the same definition
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
                    "enum": ["text", "image", "both", "none", "page load", "action"]
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

# --- (wait_for_page_conditions_pw, perform_actions_pw, capture_page_text_pw, capture_page_screenshot_pw, encode_image_to_data_uri, load_yaml_config remain the same as the last version) ---
def wait_for_page_conditions_pw(page, url_loaded, extra_wait_s=None):
    """Handles waiting for load state, fragment scroll, and applies a fixed wait using Playwright."""
    print(f"    Page '{url_loaded}' navigation initiated (Playwright handles load state).")
    scroll_wait = 1.0 # Seconds
    _base, fragment = urldefrag(url_loaded)
    if fragment:
        try:
            print(f"    Fragment '#{fragment}' detected. Attempting to scroll into view.")
            target_locator = page.locator(f"#{fragment}")
            target_locator.wait_for(state="attached", timeout=10000)
            target_locator.scroll_into_view_if_needed(timeout=5000)
            print(f"    Executed scrollIntoView for ID '{fragment}'.")
            page.wait_for_timeout(scroll_wait * 1000)
        except PlaywrightTimeoutError:
             print(f"    Warning: Timed out finding/scrolling to element with ID '{fragment}'.")
        except Exception as scroll_err:
             print(f"    Warning: Error during scrolling attempt for '#{fragment}': {scroll_err}")
    wait_duration_s = DEFAULT_POST_LOAD_WAIT_S
    if extra_wait_s is not None:
        try:
            custom_wait = float(extra_wait_s)
            if custom_wait >= 0:
                wait_duration_s = custom_wait
                print(f"    Applying custom extra wait: {wait_duration_s}s")
            else:
                print(f"    Warning: Invalid negative extra_wait_s value ({extra_wait_s}), using default: {wait_duration_s}s")
        except (ValueError, TypeError):
            print(f"    Warning: Invalid non-numeric extra_wait_s value ('{extra_wait_s}'), using default: {wait_duration_s}s")
    else:
        print(f"    Applying default extra wait: {wait_duration_s}s")
    if wait_duration_s > 0:
        page.wait_for_timeout(wait_duration_s * 1000)
def perform_actions_pw(page, actions):
    """ Performs a list of actions defined in the YAML using Playwright, expecting direct selectors. """
    if not actions: return True
    print("    Performing actions before capture...")
    action_timeout_ms = CLICK_WAIT_TIMEOUT_S * 1000
    for i, action_def in enumerate(actions):
        action_type = action_def.get("action")
        selector = action_def.get("selector")
        default_wait_s = DEFAULT_WAIT_AFTER_CLICK_S
        if action_type == "input_text": default_wait_s = DEFAULT_WAIT_AFTER_INPUT_S
        if action_type == "press_key": default_wait_s = DEFAULT_WAIT_AFTER_KEY_S
        wait_after_s = action_def.get("wait_after_s", default_wait_s)
        action_desc = f"Action #{i+1} ({action_type}"
        if selector: action_desc += f": selector='{selector}'"
        action_desc += ")"
        if not action_type:
            print(f"    ERROR: Missing 'action' type in action definition #{i+1}.")
            return False
        if action_type in ["click", "input_text", "press_key"] and not selector:
             print(f"    ERROR: Missing 'selector' for required action '{action_type}' ({action_desc}).")
             return False
        locator = None
        original_locator = None
        if selector:
            try:
                locator = page.locator(selector)
                original_locator = locator
            except Exception as loc_err:
                print(f"    ERROR: Invalid Playwright selector '{selector}' for {action_desc}: {loc_err}")
                return False
        try:
            if locator:
                 try:
                     locator.wait_for(state="attached", timeout=action_timeout_ms / 2)
                     count = locator.count()
                     if count > 1:
                         print(f"    WARNING: Selector '{selector}' resolved to {count} elements. Using the first element for action.")
                         locator = locator.first
                     elif count == 0:
                          print(f"    ERROR: Selector '{selector}' resolved to 0 elements.")
                          locator.wait_for(state="attached", timeout=action_timeout_ms / 2)
                          print(f"    INFO: Element '{selector}' appeared after waiting.")
                          count = locator.count()
                          if count > 1:
                               print(f"    WARNING: Selector '{selector}' now resolved to {count} elements after waiting. Using the first.")
                               locator = locator.first
                          elif count == 0:
                               print(f"    ERROR: Selector '{selector}' still resolved to 0 elements after waiting.")
                               return False
                 except PlaywrightTimeoutError:
                      print(f"    ERROR: Timed out waiting for element '{selector}' to be attached.")
                      return False
                 except Exception as count_err:
                      print(f"    ERROR: Unexpected error checking element count for '{selector}': {count_err}")
                      return False
            if locator:
                try:
                    print(f"      Scrolling element into view (if needed): {selector}")
                    locator.scroll_into_view_if_needed(timeout=action_timeout_ms / 2)
                    page.wait_for_timeout(300)
                except PlaywrightTimeoutError as scroll_timeout_err:
                    if original_locator and original_locator.count() > 1:
                        print(f"    WARNING: Scrolling timed out for '{selector}', possibly due to multiple elements found. Attempting action on first element directly.")
                    else:
                        print(f"    ERROR: Timed out scrolling element into view for {action_desc}.")
                        return False
                except Exception as scroll_err:
                    print(f"    WARNING: Could not scroll element for {action_desc}, attempting action anyway: {scroll_err}")
            if action_type == "click":
                if not locator: return False
                print(f"      Clicking element: {selector}")
                locator.click(timeout=action_timeout_ms)
            elif action_type == "input_text":
                if not locator: return False
                text_to_input = action_def.get("text")
                if text_to_input is None:
                    print(f"    ERROR: Missing 'text' field for input_text action ({action_desc}).")
                    return False
                print(f"      Filling input: {selector} with provided text")
                locator.fill(str(text_to_input), timeout=action_timeout_ms)
            elif action_type == "press_key":
                if not locator: return False
                key_name = action_def.get("key")
                if not key_name:
                     print(f"    ERROR: Missing 'key' field for press_key action ({action_desc}).")
                     return False
                print(f"      Pressing key '{key_name}' in: {selector}")
                locator.press(key_name, timeout=action_timeout_ms)
            else:
                print(f"    WARNING: Unsupported action type '{action_type}' in {action_desc}. Skipping.")
                continue
            if wait_after_s > 0:
                print(f"      Waiting {wait_after_s}s after {action_type}...")
                page.wait_for_timeout(wait_after_s * 1000)
        except PlaywrightTimeoutError as action_timeout_err:
            err_suffix = ""
            if original_locator and original_locator.count() > 1:
                err_suffix = f" (Note: Original selector resolved to {original_locator.count()} elements; action attempted on the first)"
            print(f"    ERROR: Timed out performing {action_type} on element '{selector}'.{err_suffix}")
            print(f"    Timeout Error Details: {action_timeout_err}")
            return False
        except PlaywrightError as e:
            error_message = str(e)
            if "strict mode violation" in error_message:
                 print(f"    ERROR: Strict mode violation during {action_desc} despite handling multiple elements. Element might have changed or detached.")
                 print(f"    Error Details: {e}")
                 return False
            elif "interceptor" in error_message or "element is not visible" in error_message or "ailed visibility" in error_message:
                 print(f"    ERROR: Element interaction failed for {action_desc} (check visibility/state): {error_message}")
                 if action_type == 'click':
                     print("    Trying force click as fallback...")
                     try:
                         locator.click(force=True, timeout=action_timeout_ms)
                         print("      Force click succeeded.")
                         if wait_after_s > 0: page.wait_for_timeout(wait_after_s * 1000)
                         continue
                     except Exception as force_e:
                         print(f"    ERROR: Force click also failed for {action_desc}: {force_e}")
                 return False
            else:
                 print(f"    ERROR: Playwright error during {action_desc}: {e}")
                 return False
        except Exception as e:
            print(f"    ERROR: Unexpected error during {action_desc}: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            return False
    print("    Finished performing actions.")
    return True
def capture_page_text_pw(page):
    """Captures the text content of the body element using Playwright."""
    try:
        page.locator('body').wait_for(state="attached", timeout=5000)
        return page.locator('body').inner_text()
    except PlaywrightTimeoutError:
        print(f"    Error capturing text: Body element not found within timeout.")
        return "Error capturing text: Body element not found."
    except Exception as e:
        print(f"    Error capturing text from current page state: {e}")
        return f"Error capturing text: {e}"
def capture_page_screenshot_pw(page, screenshot_filename_with_path):
    """Captures a screenshot of the current viewport using Playwright."""
    try:
        os.makedirs(os.path.dirname(screenshot_filename_with_path), exist_ok=True)
        page.screenshot(path=screenshot_filename_with_path) # Viewport only
        print(f"    Saved screenshot artifact (viewport): {screenshot_filename_with_path}")
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
        comparison_url = config.get("COMPARISON_URL", "[https://flybase.org](https://flybase.org)")
        tests = config.get("tests")
        if tests is None:
             print(f"Warning: 'tests' key not found in {config_path}. Returning empty list.")
             return [], comparison_url
        if not isinstance(tests, list):
            print(f"Warning: 'tests' key in {config_path} is not a list. Returning empty list.")
            return [], comparison_url
        return tests, comparison_url
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
         print(f"Error reading configuration file {config_path}: {e}")
         sys.exit(1)


# --- UPDATED call_openai_api function ---
def call_openai_api(system_prompt, user_content_parts):
    """
    Calls the OpenAI API using the client.responses.create endpoint structure.
    Leverages 'instructions' for system prompt and 'tools'/'tool_choice' for function calling.
    """
    # Transform user content parts to use "input_text" and "input_image" types
    transformed_content = []
    for part in user_content_parts:
        if part.get("type") == "text":
            transformed_content.append({"type": "input_text", "text": part.get("text", "")})
        elif part.get("type") == "image_url":
            # Pass image URL dict directly, assuming it contains url and potentially detail
            image_data = {"type": "input_image", "image_url": part.get("image_url", {})}
            transformed_content.append(image_data)
        else:
             print(f"Warning: Skipping unknown content part type during transformation: {part.get('type')}")

    # Construct the 'input' payload - a list containing ONE user message dict
    input_payload = [{"role": "user", "content": transformed_content}]

    # Add context as initial text parts if desired (or could be in instructions)
    current_time_str = f"Current time context: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    # location_str = "Location context: Halifax, Nova Scotia, Canada." # Example
    # input_payload[0]["content"].insert(0, {"type": "input_text", "text": location_str})
    input_payload[0]["content"].insert(0, {"type": "input_text", "text": current_time_str})

    print(f"Attempting API call with client.responses.create (model: {MODEL_NAME})...")

    try:
        # Using client.responses.create with parameters based on new documentation
        response = client.responses.create(
            model=MODEL_NAME,
            instructions=system_prompt, # Pass system prompt here
            input=input_payload,        # Pass the user message list here
            tools=FUNCTION_SCHEMA,      # Pass function definitions via 'tools'
            tool_choice={"type": "function", "function": {"name": "record_test_result"}}, # Force specific function
            temperature=0               # Keep deterministic behavior
            # Add other parameters like max_output_tokens if needed
        )
        # print(f"DEBUG: Raw API Response: {response}") # Optional: for deep debugging
        return response
    except OpenAIError as e:
        print(f"OpenAI API call error using responses.create: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error calling client.responses.create: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- UPDATED parse_api_response function ---
def parse_api_response(response):
    """
    Parses the API response from client.responses.create,
    specifically looking for the desired function call output.
    """
    result_status = "fail"; failed_component = "api_error"; explanation = "Could not parse API response or API call failed."
    if not response:
        explanation = "No API response received (API call likely failed)."
        return {"result": result_status, "failed_component": failed_component, "explanation": explanation}

    try:
        # The 'output' is a list. Usually the assistant message with tool calls is the first item.
        if hasattr(response, 'output') and response.output:
            assistant_output = response.output[0] # Assuming the first output item is the assistant's message/action

            # Check for tool_calls within the first output item
            # The exact structure might vary, common patterns are checked.
            tool_calls = None
            if hasattr(assistant_output, 'tool_calls') and assistant_output.tool_calls:
                tool_calls = assistant_output.tool_calls
            elif hasattr(assistant_output, 'content') and isinstance(assistant_output.content, list):
                 # Sometimes tool calls might be nested within content
                 for content_item in assistant_output.content:
                     if hasattr(content_item, 'type') and content_item.type == 'tool_calls' and hasattr(content_item, 'tool_calls'):
                          tool_calls = content_item.tool_calls
                          break
                     # Or maybe directly as function call object? (Less common now)
                     elif hasattr(content_item, 'type') and content_item.type == 'function_call' and content_item.function_call:
                          # Adapt structure if needed, treat as single tool call
                           tool_calls = [{"type": "function", "function": content_item.function_call}]
                           break


            if tool_calls:
                # We expect only one tool call due to tool_choice forcing it
                if len(tool_calls) > 0 and hasattr(tool_calls[0], 'function'):
                    function_call = tool_calls[0].function
                    if function_call.name == "record_test_result":
                        arguments_str = function_call.arguments
                        if arguments_str:
                            try:
                                result_json = json.loads(arguments_str)
                                if "result" in result_json and "explanation" in result_json:
                                    result_status = result_json.get("result", "fail")
                                    failed_component = result_json.get("failed_component", "action" if result_status == "fail" else "none")
                                    explanation = result_json.get("explanation", "No explanation.")
                                    print("Successfully parsed function call from response.output.")
                                    return {"result": result_status, "failed_component": failed_component, "explanation": explanation}
                                else:
                                    explanation = f"Parsed function call JSON missing required fields ('result','explanation'). Parsed: {result_json}"
                            except json.JSONDecodeError as e:
                                explanation = f"Error parsing function call args JSON: {e}. Args: ```{arguments_str}```"
                        else:
                             explanation = "Function call arguments missing in response.output."
                    else:
                         explanation = f"Expected function call 'record_test_result' but received '{function_call.name}'."
                else:
                     explanation = "Tool calls found, but structure is unexpected or missing function details."
            # Check for direct text output if no function call was found
            elif hasattr(assistant_output, 'content') and isinstance(assistant_output.content, list) and \
                 hasattr(assistant_output.content[0], 'type') and assistant_output.content[0].type == 'output_text':
                 output_text = assistant_output.content[0].text
                 explanation = f"API call returned direct text output (function call likely failed or was not generated): {output_text}"
                 failed_component = "api_function_call_failed"
            # Check for explicit refusal message
            elif hasattr(assistant_output, 'content') and isinstance(assistant_output.content, list) and \
                 hasattr(assistant_output.content[0], 'type') and assistant_output.content[0].type == 'refusal':
                 refusal_text = assistant_output.content[0].refusal
                 explanation = f"API refused the request: {refusal_text}"
                 failed_component = "api_refusal"
            else:
                 explanation = f"Could not find expected tool call or text/refusal output in response.output[0]. Content: {getattr(assistant_output, 'content', 'N/A')}"
                 failed_component = "api_response_parsing"
        else:
            # Check for top-level output_text (SDK convenience property)
            if hasattr(response, 'output_text') and response.output_text:
                 explanation = f"API call returned direct text output via SDK property (function call likely failed): {response.output_text}"
                 failed_component = "api_function_call_failed"
            else:
                 explanation = f"API response object lacks 'output' list or 'output_text'. Response: {response}"
                 failed_component = "api_response_parsing"

    except (AttributeError, IndexError, KeyError, TypeError) as e:
        explanation = f"Error accessing API response structure: {e}. Response: {response}"
        failed_component = "api_response_parsing"

    # If we reached here, parsing failed
    print(f"Warning: API Response Parsing Failed: {explanation}")
    return {"result": "fail", "failed_component": failed_component, "explanation": explanation}

# --- run_test_pw function ---
def run_test_pw(page, test_def, comparison_url):
    """
    Execute a test using Playwright: navigate, wait, perform actions, capture, analyze.
    Returns a dict with test name, prompt, result, failed_component, and explanation.
    """
    test_name = test_def.get("name", "Unnamed Test")
    safe_test_name = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in test_name).rstrip('_')
    prompt = test_def.get("prompt", "")
    extra_wait_s = test_def.get("extra_wait_s")
    enabled = test_def.get("enabled", True)
    url_from_yaml = test_def.get("url")
    compare_to_production = test_def.get("compare_to_production", False)
    check_types = test_def.get("check_types", [])
    if check_types is None: check_types = []
    ticket = test_def.get("ticket", "")
    actions_to_perform = test_def.get("actions_before_capture", [])

    # Corrected URL comparison logic
    if comparison_url.startswith("[https://flybase.org](https://flybase.org)"):
        prod_label = "Production Screenshot"
        prod_text_label = "Production Text"
        prod_suffix = "_prod"
    elif comparison_url.startswith("[https://stage.flybase.org](https://stage.flybase.org)"):
        prod_label = "Staging Screenshot"
        prod_text_label = "Staging Text"
        prod_suffix = "_stage"
    else:
        print(f"Warning: Unrecognized COMPARISON_URL '{comparison_url}'. Defaulting labels to 'Production'.")
        prod_label = "Production Screenshot"
        prod_text_label = "Production Text"
        prod_suffix = "_prod"

    if not enabled:
        # ... (skip logic remains same) ...
        print(f"Skipping disabled test: {test_name}")
        return {"test_name": test_name, "result": "skipped", "failed_component": "none", "explanation": "Test is disabled.", "prompt": prompt}
    if not url_from_yaml:
        # ... (skip logic remains same) ...
        print(f"Skipping test '{test_name}': Missing required 'url' field in YAML.")
        return {"test_name": test_name, "result": "skipped", "failed_component": "none", "explanation": "Missing 'url'.", "prompt": prompt}

    preview_url = url_from_yaml
    prod_url = None
    if compare_to_production:
        try:
            parsed_target = urlparse(preview_url)
            path = parsed_target.path if parsed_target.path else "/"
            prod_path_query_fragment = urlunparse(('', '', path, parsed_target.params, parsed_target.query, parsed_target.fragment))
            base_comparison_url = comparison_url.rstrip('/') + '/'
            prod_url = urljoin(base_comparison_url, prod_path_query_fragment.lstrip('/'))
        except Exception as e:
            print(f"Warning: Could not construct production URL: {e}")
            compare_to_production = False
            prod_url = None

    print(f"\n=== Running Test: {test_name} ===")
    # ... (rest of the setup print statements) ...
    print(f"Preview URL: {preview_url}")
    effective_wait = extra_wait_s if extra_wait_s is not None else DEFAULT_POST_LOAD_WAIT_S
    print(f"Effective Initial Post-Load Wait: {effective_wait}s")
    if compare_to_production and prod_url:
        prod_base_label = prod_label.split()[0]
        print(f"Comparing against Base URL ({prod_base_label}): {prod_url}")
    else:
        print("Running as single page test (no production comparison).")
        compare_to_production = False # Ensure it's false if prod_url failed

    if actions_to_perform:
        print(f"Actions to perform before capture: {len(actions_to_perform)}")

    # --- Prepare USER message content list ---
    user_content_parts = []
    prompt_display = f"Test: {test_name}\nPrompt: {prompt}"
    if ticket: prompt_display += f"\nReference Ticket: {ticket}"
    user_content_parts.append({"type": "text", "text": prompt_display}) # Type will be transformed later

    action_failure = False
    action_failure_location = None
    try:
        # --- (Navigation, actions, capture logic remains the same, appending to user_content_parts) ---
        # --- It uses perform_actions_pw, capture_page_text_pw, capture_page_screenshot_pw ---
        # --- Importantly, it adds {"detail": "high"} to image_url dicts ---
        if compare_to_production and prod_url: # Comparison Test
            prod_base_label = prod_label.split()[0] # "Production" or "Staging"
            user_content_parts.append({"type": "text", "text": f"\n--- Comparing {prod_base_label} vs Preview ---"})
            user_content_parts.append({"type": "text", "text": f"{prod_base_label} URL: {prod_url}"})
            user_content_parts.append({"type": "text", "text": f"Preview URL: {preview_url}"})

            # Production Capture
            print(f"  -- Processing {prod_base_label} URL --")
            page.goto(prod_url, wait_until="load") # Rely on context default timeout
            wait_for_page_conditions_pw(page, prod_url, extra_wait_s)
            if actions_to_perform:
                 if not perform_actions_pw(page, actions_to_perform):
                      action_failure = True
                      action_failure_location = prod_base_label
                      print(f"    WARNING: Action failed on {action_failure_location}, capturing state anyway.")

            if "text" in check_types:
                prod_text = capture_page_text_pw(page)
                prod_text_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}{prod_suffix}.txt")
                try:
                    with open(prod_text_path, "w", encoding="utf-8") as f: f.write(prod_text)
                    print(f"    Saved text artifact: {prod_text_path}")
                except Exception as e: print(f"    Warning: Could not save {prod_base_label} text file: {e}")
                user_content_parts.append({"type": "text", "text": f"\n{prod_text_label}:\n```\n{prod_text}\n```"})

            if "picture" in check_types:
                prod_screenshot_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}{prod_suffix}.png")
                prod_screenshot_bytes = capture_page_screenshot_pw(page, prod_screenshot_path)
                prod_img_uri = encode_image_to_data_uri(prod_screenshot_bytes)
                if prod_img_uri:
                    user_content_parts.append({"type": "text", "text": f"\n{prod_label}:"})
                    user_content_parts.append({
                        "type": "image_url",
                        "image_url": { "url": prod_img_uri, "detail": "high" } # Using detail: high
                    })
                else: print(f"    Warning: Failed to get {prod_base_label} screenshot")

            # Preview Capture (only if prod actions didn't fail)
            if not action_failure:
                print("  -- Processing Preview URL --")
                page.goto(preview_url, wait_until="load")
                wait_for_page_conditions_pw(page, preview_url, extra_wait_s)
                if actions_to_perform:
                     if not perform_actions_pw(page, actions_to_perform):
                          action_failure = True
                          action_failure_location = "Preview"
                          print("    WARNING: Action failed on Preview, capturing state anyway.")

                if "text" in check_types:
                    preview_text = capture_page_text_pw(page)
                    preview_text_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_preview.txt")
                    try:
                         with open(preview_text_path, "w", encoding="utf-8") as f: f.write(preview_text)
                         print(f"    Saved text artifact: {preview_text_path}")
                    except Exception as e: print(f"    Warning: Could not save preview text file: {e}")
                    user_content_parts.append({"type": "text", "text": f"\nPreview State Text:\n```\n{preview_text}\n```"})

                if "picture" in check_types:
                    preview_screenshot_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_preview.png")
                    preview_screenshot_bytes = capture_page_screenshot_pw(page, preview_screenshot_path)
                    preview_img_uri = encode_image_to_data_uri(preview_screenshot_bytes)
                    if preview_img_uri:
                        user_content_parts.append({"type": "text", "text": "\nPreview State Screenshot:"})
                        user_content_parts.append({
                            "type": "image_url",
                            "image_url": { "url": preview_img_uri, "detail": "high" } # Using detail: high
                        })
                    else: print(f"    Warning: Failed to get Preview screenshot")
            else: # Actions failed on Production/Staging
                print(f"    Skipping Preview capture because action failed on {action_failure_location}.")
                user_content_parts.append({"type": "text", "text": f"\nPreview State Text:\n```\n(Skipped due to action failure on {action_failure_location})\n```"})
                user_content_parts.append({"type": "text", "text": "\nPreview State Screenshot: (Skipped)"})

        else: # Single Page Test Logic
            user_content_parts.append({"type": "text", "text": f"\n--- Single Page Test ---"})
            user_content_parts.append({"type": "text", "text": f"URL Tested: {preview_url}"})

            page.goto(preview_url, wait_until="load")
            wait_for_page_conditions_pw(page, preview_url, extra_wait_s)
            if actions_to_perform:
                if not perform_actions_pw(page, actions_to_perform):
                     action_failure = True
                     action_failure_location = "Single Page"
                     print("    WARNING: Action failed on Single Page, proceeding with capture.")

            if "text" in check_types:
                page_text = capture_page_text_pw(page)
                text_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_preview.txt")
                try:
                    with open(text_path, "w", encoding="utf-8") as f: f.write(page_text)
                    print(f"    Saved text artifact: {text_path}")
                except Exception as e: print(f"    Warning: Could not save text file: {e}")
                user_content_parts.append({"type": "text", "text": f"\nPreview Page Text:\n```\n{page_text}\n```"})

            if "picture" in check_types:
                preview_screenshot_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_preview.png")
                preview_screenshot_bytes = capture_page_screenshot_pw(page, preview_screenshot_path)
                preview_img_uri = encode_image_to_data_uri(preview_screenshot_bytes)
                if preview_img_uri:
                    user_content_parts.append({"type": "text", "text": "\nPreview State Screenshot:"})
                    user_content_parts.append({
                         "type": "image_url",
                         "image_url": { "url": preview_img_uri, "detail": "high" } # Using detail: high
                     })
                else:
                    print(f"    Warning: Failed to get screenshot for single page test {test_name}")

        # Handle Action Failure Reporting
        if action_failure:
             user_content_parts.append({"type": "text", "text": f"\n*** NOTE: An action failed on {action_failure_location}. The captured state might be incomplete or inaccurate. ***"})

    except PlaywrightTimeoutError as page_err:
        # ... (Error handling remains same) ...
        error_msg = f"Timeout Error during page load or action for test '{test_name}'. URL(s): {preview_url}";
        if prod_url: error_msg += f", {prod_url}"
        error_msg += f". Error: {page_err}"; print(f"    ERROR: {error_msg}")
        try:
            if page and not page.is_closed():
                 fail_screenshot_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_FAILURE_capture.png")
                 capture_page_screenshot_pw(page, fail_screenshot_path)
        except Exception as screen_err: print(f"    Could not take failure screenshot: {screen_err}")
        return {"test_name": test_name, "result": "fail", "failed_component": "page load/action", "explanation": error_msg, "prompt": prompt}
    except Exception as page_err:
        # ... (Error handling remains same) ...
        error_msg = f"Unexpected Error during page load, action, or data capture for test '{test_name}'. URL(s): {preview_url}";
        if prod_url: error_msg += f", {prod_url}"
        error_msg += f". Error: {type(page_err).__name__} - {page_err}"; print(f"    ERROR: {error_msg}")
        import traceback; traceback.print_exc()
        try:
             if page and not page.is_closed():
                 fail_screenshot_path = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_FAILURE_capture.png")
                 capture_page_screenshot_pw(page, fail_screenshot_path)
        except Exception as screen_err: print(f"    Could not take failure screenshot: {screen_err}")
        return {"test_name": test_name, "result": "fail", "failed_component": "page load/action/capture", "explanation": error_msg, "prompt": prompt}

    # --- Finalize and Call OpenAI API ---
    if not check_types:
        # ... (Skip logic remains same) ...
        print(f"Skipping OpenAI analysis for test '{test_name}' as no check_types were specified.")
        final_result = "pass" if not action_failure else "fail"
        final_component = "none" if not action_failure else "action"
        final_explanation = "No checks performed." if not action_failure else f"Action failed on {action_failure_location}. No checks performed."
        return {"test_name": test_name, "result": final_result, "failed_component": final_component, "explanation": final_explanation, "prompt": prompt}

    # Check capture success
    has_text_content = any("```" in part.get("text", "") for part in user_content_parts if isinstance(part, dict) and part.get("type") == "text" and "```" in part["text"])
    has_image_content = any(part.get("type") == "image_url" for part in user_content_parts if isinstance(part, dict))
    should_fail_due_to_capture = False
    capture_fail_reason = ""
    if "text" in check_types and not has_text_content and not action_failure:
        should_fail_due_to_capture = True
        capture_fail_reason = "Text check requested but no text data captured."
    if "picture" in check_types and not has_image_content and not action_failure:
        should_fail_due_to_capture = True
        capture_fail_reason += " Picture check requested but no image data captured." if capture_fail_reason else "Picture check requested but no image data captured."

    if should_fail_due_to_capture:
         print(f"Error: Capture failure for test '{test_name}'. Reason: {capture_fail_reason.strip()}")
         return {"test_name": test_name, "result": "fail", "failed_component": "capture", "explanation": f"Capture failure: {capture_fail_reason.strip()}", "prompt": prompt}

    # <<< Call updated call_openai_api function >>>
    # Pass system prompt and the user content parts separately
    response = call_openai_api(SYSTEM_PROMPT_CONTENT, user_content_parts)
    result_data = parse_api_response(response) # Use the updated parser

    # Override API result if action failed
    if action_failure:
        if result_data.get("result") != "fail" or result_data.get("failed_component") != "action":
            print(f"Overriding API result due to action failure on {action_failure_location}.")
            result_data["result"] = "fail"
            result_data["failed_component"] = "action"
            original_api_expl = result_data.get('explanation', 'No original API explanation.') if result_data.get("result") != "fail" else "(API also indicated failure)"
            result_data["explanation"] = f"Action failed on {action_failure_location}. Original API Explanation: {original_api_expl}"


    return {"test_name": test_name, "prompt": prompt, **result_data}


# --- main function remains the same ---
def main(config_path):
    try:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        print(f"Artifacts will be saved to container path: {ARTIFACTS_DIR}")
    except OSError as e: print(f"Error creating artifacts directory {ARTIFACTS_DIR}: {e}")

    tests, comparison_url = load_yaml_config(config_path)
    if not tests: print("No tests found/loaded."); return

    all_results = []
    with sync_playwright() as p:
        browser = None
        try:
            print("Launching browser...")
            browser = p.chromium.launch(headless=True)
            print("Creating browser context...")
            context = browser.new_context(
                viewport={'width': 1280, 'height': 1024},
                locale='en-US',
            )
            waf_secret = os.getenv("WAF_SECRET_HEADER")
            if waf_secret:
                 print("Setting WAF secret header...")
                 context.set_extra_http_headers({"x-waf-secret": waf_secret})
            context.set_default_navigation_timeout(60000)
            context.set_default_timeout(30000)
            page = context.new_page()
            print("Playwright browser page created.")

            for test_def in tests:
                if not isinstance(test_def, dict) or not test_def.get("name"):
                    print(f"Warning: Skipping invalid test definition: {test_def}"); continue
                result = None
                try:
                    result = run_test_pw(page, test_def, comparison_url)
                except Exception as err:
                    test_name_fallback = test_def.get("name", "Unknown Test")
                    prompt_fallback = test_def.get("prompt", "")
                    print(f"CRITICAL: Unhandled exception during test '{test_name_fallback}': {err}")
                    import traceback; traceback.print_exc()
                    result = {"test_name": test_name_fallback, "result": "fail", "failed_component": "test execution crash",
                              "explanation": f"Unhandled Exception CRASH: {err}", "prompt": prompt_fallback}
                if result: all_results.append(result)

            print("\nClosing Playwright browser context...")
            context.close()

        except Exception as pw_err:
            print(f"FATAL: Playwright setup or top-level execution error: {pw_err}")
            import traceback; traceback.print_exc()
        finally:
             if browser and browser.is_connected():
                 print("Ensuring browser is closed...")
                 browser.close()
             print("Playwright closed.")

    # Process Results
    results_filename = os.path.join(ARTIFACTS_DIR, "test_results.json")
    try:
        if all_results:
             os.makedirs(ARTIFACTS_DIR, exist_ok=True)
             with open(results_filename, "w", encoding="utf-8") as f: json.dump(all_results, f, indent=2)
             print(f"\nFull results saved to container path: {results_filename}")
        else: print("\nNo test results generated to save.")
    except Exception as e: print(f"\nError saving results to {results_filename}: {e}")

    # --- Test Summary (remains the same) ---
    final_passed = sum(1 for r in all_results if r.get("result") == "pass")
    final_failed = sum(1 for r in all_results if r.get("result") == "fail")
    final_skipped = sum(1 for r in all_results if r.get("result") == "skipped")

    print("\n--- TEST SUMMARY ---")
    print(f"Total tests defined in config: {len(tests)}")
    print(f"Total results recorded: {len(all_results)}")
    print(f"Passed: {final_passed}"); print(f"Failed: {final_failed}"); print(f"Skipped: {final_skipped}")

    if final_failed > 0:
        print("\n--- FAILED TEST DETAILS ---")
        failed_tests = sorted([r for r in all_results if r.get("result") == "fail"], key=lambda x: x.get('test_name', ''))
        for res in failed_tests:
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