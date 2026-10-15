"""Parse audit/scientific_flaws.md and derive the severity x category cross-tab."""
import pathlib
import re
from collections import Counter

p = pathlib.Path(r"D:\rep\4.6\co2eor_optimizer\audit\scientific_flaws.md")
text = p.read_text(encoding="utf-8")

findings = {}  # id -> (severity, category)

# Format A: bullet findings  ### CRIT-01 — ... then - **Severity:** HIGH | **Category:** `SOFTWARE`
for m in re.finditer(r"^### ((?:CRIT|HIGH)-\d+)\b", text, re.M):
    fid = m.group(1)
    tail = text[m.end():m.end() + 600]
    sv = re.search(r"\*\*Severity:\*\*\s*([A-Z]+)", tail)
    ct = re.search(r"\*\*Category:\*\*\s*`?([A-Z]+)`?", tail)
    findings[fid] = (sv.group(1) if sv else "?", ct.group(1) if ct else "?")

# Format B: table rows  | **MED-01** | PROVENANCE | ...  and | **LOW-01** | PHYSICAL | ...
for m in re.finditer(r"^\| \*\*((?:MED|LOW)-\d+)\*\* \| ([A-Z]+) \|", text, re.M):
    findings[m.group(1)] = (m.group(2), "FIXED")  # placeholder, real order below

# table order is actually | ID | Cat |
rows = {}
for m in re.finditer(r"^\| \*\*((?:MED|LOW)-\d+)\*\* \| ([A-Z]+) \|", text, re.M):
    rows[m.group(1)] = m.group(2)

sev_of = {}
for fid in findings:
    sev_of[fid] = findings[fid][0]
for fid, cat in rows.items():
    sev_of[fid] = fid.split("-")[0]

# rebuild: severity from id prefix for MED/LOW, from bullet for CRIT/HIGH
cat_of = {}
for fid, (sv, ct) in findings.items():
    if fid.startswith(("MED", "LOW")):
        continue
    cat_of[fid] = ct
for fid, ct in rows.items():
    cat_of[fid] = ct

order = {"CRIT": 0, "HIGH": 1, "MED": 2, "LOW": 3}
sev_of = {fid: fid.split("-")[0] for fid in cat_of}

missing = set(re.findall(r"^### ((?:CRIT|HIGH)-\d+)", text, re.M)) - set(cat_of)
print("findings parsed:", len(cat_of))
print("missing category:", sorted(missing))

grid = Counter((sev_of[f], cat_of[f]) for f in cat_of)
cats = ["MATHEMATICAL", "PHYSICAL", "NUMERICAL", "SOFTWARE", "PROVENANCE", "FIXED?"]
sevs = ["CRIT", "HIGH", "MED", "LOW"]
print("\n| Category | CRIT | HIGH | MED | LOW | Total |")
print("|---|---:|---:|---:|---:|---:|")
totals = Counter()
for c in [x for x in cats if any(grid[(s, x)] for s in sevs)]:
    row = [grid[(s, c)] for s in sevs]
    for s, v in zip(sevs, row):
        totals[s] += v
    print(f"| {c} | " + " | ".join(str(v) for v in row) + f" | **{sum(row)}** |")
print("| **Total** | " + " | ".join(f"**{totals[s]}**" for s in sevs) +
      f" | **{sum(totals.values())}** |")
unknown = [(f, cat_of[f]) for f in cat_of if cat_of[f] not in cats]
if unknown:
    print("unknown category values:", unknown)
