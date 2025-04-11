import json
import os
import glob
from jinja2 import Environment, FileSystemLoader

ARTIFACTS_DIR = "artifacts"
DOCS_DIR = "docs"

def main():
    os.makedirs(DOCS_DIR, exist_ok=True)

    results_path = os.path.join(ARTIFACTS_DIR, "test_results.json")
    if not os.path.isfile(results_path):
        print(f"ERROR: No test_results.json found at {results_path}")
        return

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            test_results = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not read or parse {results_path}: {e}")
        return

    if not isinstance(test_results, list):
        print(f"ERROR: Expected a list in {results_path}, but got {type(test_results)}")
        return

    passed_count = sum(1 for r in test_results if isinstance(r, dict) and r.get("result") == "pass")
    failed_count = sum(1 for r in test_results if isinstance(r, dict) and r.get("result") == "fail")
    skipped_count = sum(1 for r in test_results if isinstance(r, dict) and r.get("result") == "skipped")

    tests_for_template = []
    for r in test_results:
        if not isinstance(r, dict):
            print(f"Warning: Skipping invalid entry in test_results: {r}")
            continue

        name = r.get("test_name", "Unnamed Test")
        status = r.get("result")
        explanation = r.get("explanation", "")
        prompt = r.get("prompt", "")
        safe_test_name = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in name).rstrip('_')
        # Use os.path.join for robust path construction
        artifact_search_pattern = os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_*")

        # Find associated artifacts
        text_files = [os.path.basename(p) for p in glob.glob(f"{artifact_search_pattern}.txt")]
        screenshots_prod = []
        screenshots_preview = []
        # screenshots_single list removed

        # Use os.path.join here too
        for path in glob.glob(os.path.join(ARTIFACTS_DIR, f"{safe_test_name}_*.png")):
            filename = os.path.basename(path)
            if filename.endswith("_prod.png"):
                screenshots_prod.append(filename)
            elif filename.endswith("_preview.png"):
                screenshots_preview.append(filename)
            # --- REMOVED check for _single.png ---

        tests_for_template.append({
            "name": name,
            "status": status,
            "explanation": explanation,
            "prompt": prompt,
            "text_files": text_files,
            "screenshots_prod": screenshots_prod,
            "screenshots_preview": screenshots_preview
            # screenshots_single key removed
        })

    # Use Jinja to render an HTML file
    try:
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

    # Write index.html to docs folder
    output_html_path = os.path.join(DOCS_DIR, "index.html")
    try:
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_output)
        print(f"Generated {output_html_path} for GitHub Pages.")
    except Exception as e:
        print(f"ERROR: Could not write HTML report to {output_html_path}: {e}")

if __name__ == "__main__":
    main()