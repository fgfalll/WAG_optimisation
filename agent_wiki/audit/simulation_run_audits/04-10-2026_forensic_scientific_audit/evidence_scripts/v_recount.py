"""Re-derive severity x category counts directly from audit/scientific_flaws.md."""
import io
import re
from collections import Counter

P = r"D:\rep\4.6\co2eor_optimizer\audit\scientific_flaws.md"
txt = io.open(P, encoding="utf-8").read()
lines = txt.splitlines()

findings = {}   # id -> (severity, category, status)
cur = None
for i, ln in enumerate(lines):
    m = re.match(r"^### (CRIT|HIGH|MED|LOW)-(\d+)\s*[—-]\s*(.*)$", ln)
    if m:
        cur = f"{m.group(1)}-{m.group(2)}"
        findings.setdefault(cur, [None, None, None])
        continue
    m2 = re.match(r"^\|\s*\*\*(MED|LOW)-(\d+)\*\*\s*\|\s*([A-Z]+)\s*\|", ln)
    if m2:
        fid = f"{m2.group(1)}-{m2.group(2)}"
        findings[fid] = ["MEDIUM" if m2.group(1) == "MED" else "LOW",
                         m2.group(3).strip(), None]
        tail = ln.rstrip().rstrip("|").rsplit("|", 1)[-1].strip()
        if tail and tail != "---":
            findings[fid][2] = tail
        continue
    if cur and cur in findings:
        m3 = re.match(r"^- \*\*Severity:\*\* ([A-Z]+)(?: \| \*\*Category:\*\* `?([A-Z]+)`?)?", ln)
        if m3 and findings[cur][0] is None:
            findings[cur][0] = m3.group(1)
            if m3.group(2):
                findings[cur][1] = m3.group(2)
        m3b = re.match(r"^- \*\*Category:\*\* `?([A-Z]+)`?", ln)
        if m3b and findings[cur][1] is None:
            findings[cur][1] = m3b.group(1)
        m4 = re.match(r"^- \*\*Status:\*\* (.*)$", ln)
        if m4:
            findings[cur][2] = m4.group(1).strip()

sev_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
cats = ["MATHEMATICAL", "PHYSICAL", "NUMERICAL", "SOFTWARE", "PROVENANCE"]

print("TOTAL parsed:", len(findings))
missing = [k for k, v in findings.items() if v[0] is None or v[1] is None]
print("MISSING severity/category:", missing)

grid = Counter()
for fid, (sev, cat, st) in findings.items():
    if sev and cat:
        grid[(cat, sev)] += 1

hdr = f"{'CAT':<13}" + "".join(f"{s[:4]:>7}" for s in sev_order) + f"{'TOT':>7}"
print(hdr)
print("-" * len(hdr))
coltot = Counter()
for c in cats:
    row = [grid[(c, s)] for s in sev_order]
    for s, v in zip(sev_order, row):
        coltot[s] += v
    print(f"{c:<13}" + "".join(f"{v:>7}" for v in row) + f"{sum(row):>7}")
print("-" * len(hdr))
print(f"{'Total':<13}" + "".join(f"{coltot[s]:>7}" for s in sev_order)
      + f"{sum(coltot.values()):>7}")

print("\n--- ID -> severity/category/status ---")
for fid in sorted(findings, key=lambda x: (sev_order.index(findings[x][0]) if findings[x][0] in sev_order else 9,
                                           int(x.split("-")[1]))):
    sev, cat, st = findings[fid]
    print(f"  {fid:<9} {str(sev):<9} {str(cat):<13} {st}")

print("\nSTATUS distribution:")
for st, n in Counter((v[2] or "?").split(" ")[0] for v in findings.values()).most_common():
    print(f"  {st:<14} {n}")
