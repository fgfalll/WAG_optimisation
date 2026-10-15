import glob, re, collections
files = [f for f in glob.glob(r"D:\rep\4.6\co2eor_optimizer\tests\**\*.py", recursive=True)]
c = collections.Counter()
det = collections.Counter()
for f in files:
    s = open(f, encoding="utf-8", errors="replace").read()
    for pat, key in [(r"phd_hybrid", "phd_hybrid"),
                     (r"recovery_model_type\s*=\s*[\"']hybrid[\"']", "hybrid_kwarg"),
                     (r"[\"']hybrid[\"']", "hybrid_literal")]:
        n = len(re.findall(pat, s))
        c[key] += n
        if n:
            det[key] += n
print("tests/ totals:", dict(c))
for f in files:
    s = open(f, encoding="utf-8", errors="replace").read()
    for m in re.finditer(r".{0,60}[\"']hybrid[\"'].{0,40}", s):
        print("  HYBRID:", f.split("co2eor_optimizer")[-1], "|", m.group(0).replace("\n", " ")[:110])
