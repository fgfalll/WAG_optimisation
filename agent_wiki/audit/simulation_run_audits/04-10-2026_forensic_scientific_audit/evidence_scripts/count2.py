import re, collections
txt = open(r"D:\rep\4.6\co2eor_optimizer\audit\scientific_flaws.md", encoding="utf-8").read()
data = []
for s in re.split(r"^### ", txt, flags=re.M)[1:]:
    m = re.match(r"(CRIT|HIGH)-(\d+)", s)
    if not m:
        continue
    cm = re.search(r"\*\*Category:\*\*\s*`?([A-Z]+)", s)
    data.append((m.group(0), cm.group(1) if cm else "??"))
for l in txt.splitlines():
    m = re.match(r"\|\s*\*\*((?:MED|LOW)-\d+)\*\*\s*\|\s*([A-Z]+)\s*\|", l)
    if m:
        data.append((m.group(1), m.group(2)))
order = ["CRIT", "HIGH", "MED", "LOW"]
bysev = collections.defaultdict(list)
for i, c in data:
    bysev[i.split("-")[0]].append((i, c))
for s in order:
    print(s, len(bysev[s]))
    print("  ", ", ".join("%s=%s" % (i, c[:4]) for i, c in bysev[s]))
# cross tab
ct = collections.defaultdict(lambda: collections.Counter())
for i, c in data:
    ct[c][i.split("-")[0]] += 1
for c in ["MATHEMATICAL", "PHYSICAL", "NUMERICAL", "SOFTWARE", "PROVENANCE"]:
    k = ct[c]
    print("| %s | %d | %d | %d | %d | %d |" % (c, k["CRIT"], k["HIGH"], k["MED"], k["LOW"], sum(k.values())))
