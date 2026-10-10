"""
Extracts executed functions and module coverage from the pytest .coverage database.
"""

import json
from pathlib import Path
import coverage

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    cov_file = REPO_ROOT / ".coverage"
    if not cov_file.exists():
        print(f"Warning: {cov_file} does not exist.")
        return
        
    cov = coverage.Coverage(data_file=str(cov_file))
    cov.load()
    data = cov.get_data()
    
    executed_files = {}
    for filename in data.measured_files():
        try:
            rel = Path(filename).relative_to(REPO_ROOT).as_posix()
        except ValueError:
            continue
            
        if any(d in rel for d in [".venv", "tests"]):
            continue
            
        lines = data.lines(filename)
        executed_files[rel] = {
            "executed_lines_count": len(lines) if lines else 0,
            "status": "EXECUTED AND TESTED" if lines else "STATICALLY USED BUT NOT EXECUTED"
        }
        
    out_json = REPO_ROOT / "audit" / "runtime" / "executed_functions.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(executed_files, f, indent=2)
        
    print(f"Executed modules mapped: {len(executed_files)} modules saved to {out_json}.")

if __name__ == "__main__":
    main()
