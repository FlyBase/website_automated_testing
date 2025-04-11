import json
import os
import glob
from jinja2 import Environment, FileSystemLoader

ARTIFACTS_DIR = "artifacts"
DOCS_DIR = "docs"

def main():
    os.makedirs(DOCS_DIR, exist_ok=True)

    # 1) Load test_results.json
    results_path = os.path.join(ARTIFACTS_DIR, "test_results.json")
    if not os.path.isfile(results_path):
        print(f"ERROR: No test_results.json found at {results_path}")
        return

    with open(results_path, "r") as f:
        test_results = json.load(f)

    # 2) Summaries
    passed_count = sum(1 for r in test_results if r.get("result") == "pass")
    failed_count = sum(1 for r in test_results if r.get("result") == "fail")
    skipped_count = sum(1 for r in test_results if r.get("result") == "skipped")

    # 3) Build a data structure for the template
    #    For each test, see if there's a single or compare artifacts: 
    #    e.g. *single.txt or *prod.txt / *target.txt / *prod.png / *target.png
    tests_for_template = []
    for r in test_results:
        name = r.get("test_name", "Unnamed Test")
        status = r.get("result")
        explanation = r.get("explanation", "")
        # We'll also see if there's a _prod/_target or _single file to link
        safe_test_name = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in name).rstrip('_')

        # Grab text files
        text_files = []
        for path in glob.glob(f"{ARTIFACTS_DIR}/{safe_test_name}_*.txt"):
            filename = os.path.basename(path)
            text_files.append(filename)
        
        # Grab screenshots
        screenshots_prod = []
        screenshots_target = []
        screenshots_single = []

        # Match patterns
        for path in glob.glob(f"{ARTIFACTS_DIR}/{safe_test_name}_*.png"):
            filename = os.path.basename(path)
            if "_prod.png" in filename:
                screenshots_prod.append(filename)
            elif "_target.png" in filename:
                screenshots_target.append(filename)
            elif "_single.png" in filename:
                screenshots_single.append(filename)

        tests_for_template.append({
            "name": name,
            "status": status,
            "explanation": explanation,
            "text_files": text_files,
            "screenshots_prod": screenshots_prod,
            "screenshots_target": screenshots_target,
            "screenshots_single": screenshots_single
        })

    # 4) Use Jinja to render an HTML file
    env = Environment(loader=FileSystemLoader("./scripts/templates"))  # or wherever your template is
    template = env.get_template("report_template.html")

    html_output = template.render(
        passed_count=passed_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        tests=tests_for_template
    )

    # 5) Write index.html to docs folder
    with open(os.path.join(DOCS_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_output)
    
    print("Generated docs/index.html for GitHub Pages.")


if __name__ == "__main__":
    main()
