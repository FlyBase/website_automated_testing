import json
import os
import glob
from jinja2 import Environment, FileSystemLoader

# Assuming ARTIFACTS_DIR and DOCS_DIR are relative to where this script runs
# In the GH Action context, this script runs from the root, so these paths are correct
ARTIFACTS_DIR = "artifacts"
DOCS_DIR = "docs"

def main():
    # Ensure the docs directory exists (where index.html will be written)
    # The artifacts subdirectory within docs should be created by the workflow copy step
    os.makedirs(DOCS_DIR, exist_ok=True)

    # 1) Load test_results.json (expected inside ARTIFACTS_DIR)
    results_path = os.path.join(ARTIFACTS_DIR, "test_results.json")
    if not os.path.isfile(results_path):
        print(f"ERROR: No test_results.json found at {results_path}")
        # Consider exiting or creating an empty report
        # For now, just return to prevent further errors
        return

    try:
        with open(results_path, "r", encoding="utf-8") as f: # Added encoding
            test_results = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not parse {results_path}: {e}")
        return
    except Exception as e:
        print(f"ERROR: Could not read {results_path}: {e}")
        return

    # Check if test_results is a list (basic validation)
    if not isinstance(test_results, list):
        print(f"ERROR: Expected a list in {results_path}, but got {type(test_results)}")
        return


    # 2) Summaries
    passed_count = sum(1 for r in test_results if isinstance(r, dict) and r.get("result") == "pass")
    failed_count = sum(1 for r in test_results if isinstance(r, dict) and r.get("result") == "fail")
    skipped_count = sum(1 for r in test_results if isinstance(r, dict) and r.get("result") == "skipped")

    # 3) Build a data structure for the template
    tests_for_template = []
    for r in test_results:
        # Skip items that are not dictionaries (robustness)
        if not isinstance(r, dict):
            print(f"Warning: Skipping invalid entry in test_results: {r}")
            continue

        name = r.get("test_name", "Unnamed Test")
        status = r.get("result") # Might be None if JSON is malformed
        explanation = r.get("explanation", "")
        prompt = r.get("prompt", "") # <<< Get the prompt >>>

        # Generate a safe filename prefix based on the test name
        safe_test_name = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in name).rstrip('_')

        # --- Find associated artifacts ---
        # Search pattern relative to the script's location (root) -> ARTIFACTS_DIR
        artifact_search_path = f"{ARTIFACTS_DIR}/{safe_test_name}_*"

        # Grab text files
        text_files = [os.path.basename(p) for p in glob.glob(f"{artifact_search_path}.txt")]

        # Grab screenshots
        screenshots_prod = []
        screenshots_target = []
        screenshots_single = []

        for path in glob.glob(f"{artifact_search_path}.png"):
            filename = os.path.basename(path)
            # More robust checking for screenshot types
            if filename.endswith("_prod.png"):
                screenshots_prod.append(filename)
            elif filename.endswith("_target.png"):
                screenshots_target.append(filename)
            elif filename.endswith("_single.png"):
                screenshots_single.append(filename)
            # Add handling for the _FAILURE_capture.png? Might need a separate list.
            # elif filename.endswith("_FAILURE_capture.png"):
            #     screenshots_failure.append(filename)


        tests_for_template.append({
            "name": name,
            "status": status,
            "explanation": explanation,
            "prompt": prompt, # <<< Add prompt to template data >>>
            "text_files": text_files,
            "screenshots_prod": screenshots_prod,
            "screenshots_target": screenshots_target,
            "screenshots_single": screenshots_single
            # "screenshots_failure": screenshots_failure # If adding failure screenshots
        })

    # 4) Use Jinja to render an HTML file
    # Assuming 'scripts/templates' exists relative to the script's execution (root)
    try:
        # Ensure FileSystemLoader path is correct relative to execution context
        # In GH Actions, running `python scripts/generate_html_report.py` from the root,
        # the template path should be relative to the root.
        template_dir = os.path.join("scripts", "templates")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("report_template.html")
    except Exception as e:
        print(f"ERROR: Could not load Jinja template from {template_dir}: {e}")
        return


    html_output = template.render(
        passed_count=passed_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        tests=tests_for_template
    )

    # 5) Write index.html to docs folder
    output_html_path = os.path.join(DOCS_DIR, "index.html")
    try:
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_output)
        print(f"Generated {output_html_path} for GitHub Pages.")
    except Exception as e:
        print(f"ERROR: Could not write HTML report to {output_html_path}: {e}")


if __name__ == "__main__":
    main()