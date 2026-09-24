"""
Processes raw vulture output and classifies dead code candidates according to audit categories:
CONFIRMED DEAD, PROBABLY DEAD, POSSIBLY ACTIVE, DYNAMICALLY USED, EXPERIMENTAL, LEGACY BUT INTENTIONAL, UNKNOWN.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
raw_file = REPO_ROOT / "audit" / "code" / "dead_code_raw.txt"
out_file = REPO_ROOT / "audit" / "code" / "dead_code.txt"

if not raw_file.exists():
    print(f"Error: {raw_file} does not exist.")
    exit(1)

with open(raw_file, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

classified = {
    "CONFIRMED DEAD": [],
    "PROBABLY DEAD": [],
    "POSSIBLY ACTIVE": [],
    "DYNAMICALLY USED": [],
    "EXPERIMENTAL": [],
    "LEGACY BUT INTENTIONAL": [],
    "UNKNOWN": []
}

for line in lines:
    line = line.strip()
    if not line or ":" not in line:
        continue
        
    parts = line.split(":", 2)
    file_path = parts[0]
    
    # Classification logic
    if "unified_engine" in file_path or "engine_simple" in file_path or "compositional_engine" in file_path:
        classified["PROBABLY DEAD"].append((line, "Orphaned legacy simulation engine (never reached by EngineFactory)"))
    elif "simulation/profile_generator.py" in file_path or "simulation/injection_schemes.py" in file_path:
        classified["LEGACY BUT INTENTIONAL"].append((line, "Deprecated legacy wrapper superseded by FastProfileGenerator"))
    elif "analysis/" in file_path or "scripts/" in file_path:
        classified["EXPERIMENTAL"].append((line, "Standalone analytical or post-processing tool"))
    elif "ui/" in file_path:
        if "on_" in line or "slot" in line.lower() or "signal" in line.lower():
            classified["POSSIBLY ACTIVE"].append((line, "PyQt signal handler or UI callback"))
        else:
            classified["CONFIRMED DEAD"].append((line, "Unused UI variable or helper method"))
    elif "100% confidence" in line and "unused variable" in line:
        classified["CONFIRMED DEAD"].append((line, "Local variable assigned but never read"))
    elif "unused import" in line:
        classified["CONFIRMED DEAD"].append((line, "Unused import statement"))
    else:
        classified["UNKNOWN"].append((line, "Requires runtime trace verification"))

with open(out_file, "w", encoding="utf-8") as f:
    f.write("CO2 EOR OPTIMIZER - CLASSIFIED DEAD CODE AUDIT\n")
    f.write(f"Total Raw Candidates: {len(lines)}\n")
    f.write("=" * 80 + "\n\n")
    
    for category, items in classified.items():
        f.write(f"## {category} ({len(items)} items)\n")
        f.write("-" * 50 + "\n")
        for item, note in items[:50]: # Top 50 per category for readability
            f.write(f"- {item}\n  Note: {note}\n")
        if len(items) > 50:
            f.write(f"  ... [{len(items) - 50} additional items omitted for brevity]\n")
        f.write("\n")

print(f"Classified dead code saved to {out_file}.")
