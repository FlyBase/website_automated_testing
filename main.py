#!/usr/bin/env python3
"""
qa_vision_runner.py - End-to-end Playwright + OpenAI vision test runner
Updated April 2025 for the *Responses* API (vision + Structured Outputs)

FlyBase QA team
"""

###############################################################################
# IMPORTS
###############################################################################
import os
import sys
import time
import json
import base64
import yaml
import argparse
from urllib.parse import urlparse, urldefrag, urljoin, urlunparse

# OpenAI
from openai import OpenAI, OpenAIError
import jsonschema                # pip install jsonschema

# Playwright
from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError,
    Error as PlaywrightError,
)

###############################################################################
# GLOBAL CONSTANTS
###############################################################################
SYSTEM_PROMPT_CONTENT = """
You are a meticulous QA-automation assistant for FlyBase .org, a Drosophila genomics database.

**Task**
After evaluating the evidence for a single web-page test, return a JSON object that adheres *exactly* to the provided `test_result` schema (result / failed_component / explanation).  
• Produce **only** the JSON - no markdown, no extra keys, no commentary.  
• If the model refuses, return the refusal via the Structured-Outputs mechanism.

**You will receive**
1. **User Prompt** - the condition to verify (primary requirement).  
2. **Text Content** - code-fenced blocks labelled “Production Text”, “Preview State Text”, or “Preview Page Text”.  
3. **Screenshots** - images labelled “Production Screenshot” or “Preview State Screenshot”.  
   Occasionally screenshots may be missing; rely on text when needed.

**Evaluation guidelines**
• Focus strictly on the User Prompt to decide pass / fail.  
• If both Production and Preview assets are present, compare them; minor rendering differences are acceptable unless the prompt targets them.  
• Flag failures for missing data, error messages, wrong visual states, or explicit “Something is broken” pages.  
• If screenshots are present, check visual layout, colours, and absence of errors.  
• If an action failed note appears (`*** NOTE: An action failed … ***`), consider whether that incomplete state should fail the test.  
• Choose `failed_component` as “text”, “image”, “both”, “page load”, “action”, “capture”, or “none” (for passes).

**Output format (strict)**
```json
{
  "result":           "pass" | "fail",
  "failed_component": "text" | "image" | "both" | "none" | "page load" | "action" | "capture",
  "explanation":      "<concise justification referencing specific evidence>"
}
```
"""

ARTIFACTS_DIR = "/app/artifacts"
DEFAULT_POST_LOAD_WAIT_S = 1.5
CLICK_WAIT_TIMEOUT_S = 15
DEFAULT_WAIT_AFTER_CLICK_S = 1.0
DEFAULT_WAIT_AFTER_INPUT_S = 0.5
DEFAULT_WAIT_AFTER_KEY_S = 1.0

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    sys.exit("Please set the OPENAI_API_KEY environment variable.")

client = OpenAI()
MODEL_NAME = "gpt-4.1-mini"      # vision-capable Responses model

###############################################################################
# STRUCTURED-OUTPUTS JSON SCHEMA
###############################################################################
TEST_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "result":          {"type": "string", "enum": ["pass", "fail"]},
        "failed_component": {
            "type": "string",
            "enum": [
                "text", "image", "both", "none",
                "page load", "action",
                "api_error", "api_function_call_failed",
                "api_refusal", "api_response_parsing",
                "capture",
            ],
        },
        "explanation":     {"type": "string"},
    },
    "required": ["result", "failed_component", "explanation"],
    "additionalProperties": False,
}

###############################################################################
# PLAYWRIGHT HELPERS
###############################################################################
def wait_for_page_conditions_pw(page, url_loaded: str, extra_wait_s=None):
    """Wait for load-state, optional fragment scroll, then fixed delay."""
    _base, fragment = urldefrag(url_loaded)
    if fragment:
        try:
            loc = page.locator(f"#{fragment}")
            loc.wait_for(state="attached", timeout=10000)
            loc.scroll_into_view_if_needed(timeout=5000)
            page.wait_for_timeout(1000)
        except PlaywrightTimeoutError:
            print(f"    Fragment '#{fragment}' not found - continuing.")
    delay = (
        float(extra_wait_s)
        if isinstance(extra_wait_s, (int, float)) and extra_wait_s >= 0
        else DEFAULT_POST_LOAD_WAIT_S
    )
    page.wait_for_timeout(delay * 1000)


def perform_actions_pw(page, actions: list) -> bool:
    """Run click / input / key actions defined in YAML."""
    if not actions:
        return True
    timeout_ms = CLICK_WAIT_TIMEOUT_S * 1000
    for i, act in enumerate(actions):
        a_type = act.get("action")
        sel = act.get("selector")
        if a_type in {"click", "input_text", "press_key"} and not sel:
            print(f"    Action {i+1}: selector missing → abort.")
            return False
        locator = page.locator(sel) if sel else None
        try:
            if locator:
                locator.wait_for(state="attached", timeout=timeout_ms // 2)
                if locator.count() > 1:
                    locator = locator.first
                locator.scroll_into_view_if_needed(timeout=timeout_ms // 2)
                page.wait_for_timeout(300)
            if a_type == "click":
                locator.click(timeout=timeout_ms)
            elif a_type == "input_text":
                locator.fill(str(act.get("text", "")), timeout=timeout_ms)
            elif a_type == "press_key":
                locator.press(act.get("key", "Enter"), timeout=timeout_ms)
            else:
                print(f"    Unsupported action '{a_type}' - skipped.")
                continue
            pause = float(act.get("wait_after_s", DEFAULT_WAIT_AFTER_CLICK_S))
            if pause > 0:
                page.wait_for_timeout(int(pause * 1000))
        except Exception as e:
            print(f"    Action {a_type} failed: {e}")
            return False
    return True


def capture_page_text_pw(page):
    try:
        page.locator("body").wait_for(state="attached", timeout=5000)
        return page.locator("body").inner_text()
    except Exception as e:
        return f"Error capturing text: {e}"


def capture_page_screenshot_pw(page, path_):
    try:
        os.makedirs(os.path.dirname(path_), exist_ok=True)
        page.screenshot(path=path_)
        with open(path_, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"    Screenshot error: {e}")
        return None


def encode_image_to_data_uri(img_bytes, fmt="png") -> str:
    if img_bytes is None:
        return ""
    return f"data:image/{fmt};base64,{base64.b64encode(img_bytes).decode()}"


def load_yaml_config(cfg_path):
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("tests", []), cfg.get("COMPARISON_URL", "https://flybase.org")
    except Exception as e:
        sys.exit(f"Error loading YAML config {cfg_path}: {e}")

###############################################################################
# OPENAI COMMUNICATION  (Responses + Chat fallback)
###############################################################################
def _transform_parts(parts):
    out = []
    for p in parts:
        if p["type"] == "text":
            out.append({"type": "input_text", "text": p["text"]})
        elif p["type"] == "image_url":
            out.append({"type": "input_image", "image_url": p["image_url"]})
    return out


def call_openai_api(system_prompt: str, user_parts: list):
    """Send to Responses API; on failure, fall back to chat.completions."""
    msg = {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": time.strftime("Current timestamp: %Y-%m-%d %H:%M:%S %Z"),
            },
            *_transform_parts(user_parts),
        ],
    }

    # --- Primary path: Responses API -----------------------------------
    try:
        return client.responses.create(
            model=MODEL_NAME,
            instructions=system_prompt,
            input=[msg],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "test_result",
                    "schema": TEST_RESULT_SCHEMA,
                    "strict": True,
                }
            },
            temperature=0,
        )
    except OpenAIError as e:
        print(f"[Responses fallback] {e}")

    # --- Fallback path: Chat Completions with function-calling ----------
    tool_def = {
        "name": "record_test_result",
        "parameters": TEST_RESULT_SCHEMA,
        "description": "Return the test verdict in JSON.",
    }
    chat_content = [
        {"type": "text", "text": p["text"]} if p["type"] == "text" else p
        for p in msg["content"]
    ]
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chat_content},
        ],
        tools=[tool_def],
        tool_choice={"name": "record_test_result"},
        temperature=0,
    )


def parse_api_response(resp):
    """Return dict matching TEST_RESULT_SCHEMA or synthetic failure info."""
    if not resp:
        return {
            "result": "fail",
            "failed_component": "api_error",
            "explanation": "No response.",
        }

    # Responses path
    if hasattr(resp, "output_text") and resp.output_text:
        try:
            data = json.loads(resp.output_text)
            jsonschema.validate(data, TEST_RESULT_SCHEMA)
            return data
        except Exception as e:
            return {
                "result": "fail",
                "failed_component": "api_response_parsing",
                "explanation": f"JSON parse/validate error: {e}",
            }

    # Chat fallback path
    try:
        tool_call = resp.choices[0].message.tool_calls[0]
        data = json.loads(tool_call.function.arguments)
        jsonschema.validate(data, TEST_RESULT_SCHEMA)
        return data
    except Exception as e:
        return {
            "result": "fail",
            "failed_component": "api_response_parsing",
            "explanation": f"Chat fallback parse error: {e}",
        }

###############################################################################
# SINGLE-TEST EXECUTION
###############################################################################
def run_test_pw(page, test_def: dict, comparison_url: str):
    """Run one test definition and return unified result dict."""
    name = test_def.get("name", "Unnamed Test")
    safe = "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in name)
    prompt = test_def.get("prompt", "")
    extra_wait_s = test_def.get("extra_wait_s")
    enabled = test_def.get("enabled", True)
    url_preview = test_def.get("url")
    compare = bool(test_def.get("compare_to_production", False))
    checks = test_def.get("check_types", []) or []
    ticket = test_def.get("ticket", "")
    actions = test_def.get("actions_before_capture", [])

    if not enabled:
        return {
            "test_name": name,
            "prompt": prompt,
            "result": "skipped",
            "failed_component": "none",
            "explanation": "Test disabled.",
        }
    if not url_preview:
        return {
            "test_name": name,
            "prompt": prompt,
            "result": "skipped",
            "failed_component": "none",
            "explanation": "No URL provided.",
        }

    # Build production URL (if requested)
    prod_url = None
    if compare:
        try:
            parts = urlparse(url_preview)
            path = parts.path or "/"
            prod_rel = urlunparse(
                ("", "", path, parts.params, parts.query, parts.fragment)
            )
            prod_url = urljoin(comparison_url.rstrip("/") + "/", prod_rel.lstrip("/"))
        except Exception as e:
            print(f"    Production URL build error: {e}")
            compare = False

    # Prepare OpenAI user content
    user_parts = [{
        "type": "text",
        "text": f"Test: {name}\nPrompt: {prompt}\nTicket: {ticket}",
    }]

    action_failed = False
    action_where = ""

    def _capture(label_prefix: str):
        if "text" in checks:
            txt = capture_page_text_pw(page)
            path_txt = os.path.join(ARTIFACTS_DIR, f"{safe}{label_prefix}.txt")
            try:
                os.makedirs(os.path.dirname(path_txt), exist_ok=True)
                with open(path_txt, "w", encoding="utf-8") as f:
                    f.write(txt)
            except Exception:
                pass
            user_parts.append(
                {"type": "text", "text": f"\n{label_prefix.strip('_').title()} Text:\n```\n{txt}\n```"}
            )
        if "picture" in checks:
            path_img = os.path.join(ARTIFACTS_DIR, f"{safe}{label_prefix}.png")
            img_bytes = capture_page_screenshot_pw(page, path_img)
            uri = encode_image_to_data_uri(img_bytes)
            if uri:
                user_parts.append({"type": "text", "text": f"\n{label_prefix.strip('_').title()} Screenshot:"})
                user_parts.append(
                    {"type": "image_url", "image_url": {"url": uri, "detail": "high"}}
                )

    try:
        # -------------------- Production pass (if any) -------------------
        if compare and prod_url:
            print(f"\n=== {name} (comparison) ===")
            page.goto(prod_url, wait_until="load")
            wait_for_page_conditions_pw(page, prod_url, extra_wait_s)
            if actions and not perform_actions_pw(page, actions):
                action_failed = True
                action_where = "Production"
            _capture("_prod")

        # -------------------- Preview / single-page ----------------------
        print(f"\n=== {name} - Preview ===")
        page.goto(url_preview, wait_until="load")
        wait_for_page_conditions_pw(page, url_preview, extra_wait_s)
        if actions and not perform_actions_pw(page, actions):
            action_failed = True
            action_where = "Preview"
        _capture("_preview")

        # Note action failure
        if action_failed:
            user_parts.append(
                {
                    "type": "text",
                    "text": f"\n*** NOTE: An action failed on {action_where}. "
                    "Captured state may be incomplete. ***",
                }
            )

    except PlaywrightTimeoutError as e:
        return {
            "test_name": name,
            "prompt": prompt,
            "result": "fail",
            "failed_component": "page load",
            "explanation": f"Timeout during navigation/actions: {e}",
        }
    except Exception as e:
        return {
            "test_name": name,
            "prompt": prompt,
            "result": "fail",
            "failed_component": "page load",
            "explanation": f"Unhandled Playwright error: {e}",
        }

    # --------------- Call OpenAI (unless no checks requested) -----------
    if not checks:
        return {
            "test_name": name,
            "prompt": prompt,
            "result": "pass" if not action_failed else "fail",
            "failed_component": "action" if action_failed else "none",
            "explanation": "No checks requested.",
        }

    resp = call_openai_api(SYSTEM_PROMPT_CONTENT, user_parts)
    verdict = parse_api_response(resp)

    # Override if local action failed
    if action_failed:
        verdict["result"] = "fail"
        verdict["failed_component"] = "action"
        verdict["explanation"] = (
            f"Action failed on {action_where}. "
            f"OpenAI explanation: {verdict.get('explanation', 'n/a')}"
        )

    verdict.update({"test_name": name, "prompt": prompt})
    return verdict

###############################################################################
# MAIN DRIVER
###############################################################################
def main(cfg_path: str):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    tests, comp_url = load_yaml_config(cfg_path)
    if not tests:
        print("No tests in config.")
        return

    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 1024},
            locale="en-US",
        )
        waf_secret = os.getenv("WAF_SECRET_HEADER")
        if waf_secret:
            ctx.set_extra_http_headers({"x-waf-secret": waf_secret})
        ctx.set_default_navigation_timeout(60000)
        ctx.set_default_timeout(30000)
        page = ctx.new_page()

        for td in tests:
            if not isinstance(td, dict) or not td.get("name"):
                print(f"Invalid test entry skipped: {td}")
                continue
            try:
                results.append(run_test_pw(page, td, comp_url))
            except Exception as e:
                results.append(
                    {
                        "test_name": td.get("name", "Unknown"),
                        "prompt": td.get("prompt", ""),
                        "result": "fail",
                        "failed_component": "test execution crash",
                        "explanation": f"Unhandled runner exception: {e}",
                    }
                )

        ctx.close()
        browser.close()

    # ------------------------ Save results ------------------------------
    out_path = os.path.join(ARTIFACTS_DIR, "test_results.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {out_path}")
    except Exception as e:
        print(f"Error writing results file: {e}")

    # ------------------------ Summary -----------------------------------
    passed = sum(1 for r in results if r["result"] == "pass")
    failed = sum(1 for r in results if r["result"] == "fail")
    skipped = sum(1 for r in results if r["result"] == "skipped")
    print("\n--- SUMMARY ---")
    print(f"PASS  {passed}")
    print(f"FAIL  {failed}")
    print(f"SKIP  {skipped}")
    if failed:
        print("\nFailed tests:")
        for r in results:
            if r["result"] == "fail":
                print(f"  ▸ {r['test_name']} - {r['failed_component']}")
                print(f"    {r['explanation']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FlyBase Playwright QA tests using OpenAI vision."
    )
    parser.add_argument(
        "--config", default="test_config.yml", help="Path to YAML test configuration."
    )
    args = parser.parse_args()
    main(args.config)
