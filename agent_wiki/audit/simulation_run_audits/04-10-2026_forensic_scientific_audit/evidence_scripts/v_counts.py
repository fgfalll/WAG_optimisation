import glob, re, collections
files = [f for f in glob.glob(r"D:\rep\4.6\co2eor_optimizer\**\*.py", recursive=True)
         if not any(k in f for k in ("\\.venv\\", "\\venv\\", "\\audit\\", "\\scratch\\"))]
tot = collections.Counter()
byfile = collections.Counter()
for f in files:
    s = open(f, encoding="utf-8", errors="replace").read()
    for pat, key in [(r"phd_hybrid", "phd_hybrid"),
                     (r"recovery_model_type\s*=\s*[\"']hybrid[\"']", "hybrid_kwarg"),
                     (r"[\"']hybrid[\"']", "hybrid_literal")]:
        n = len(re.findall(pat, s))
        if n:
            tot[key] += n
            byfile[key + "@" + f.split("co2eor_optimizer")[-1]] += n
print(tot)
for k, v in sorted(byfile.items()):
    if k.startswith("hybrid_kwarg") or k.startswith("phd_hybrid"):
        print(" ", k, v)

# gravity_factor physics consumers (exclude optimizer/data/sensitivity/tests/ui)
print("\n--- gravity_factor in physics modules ---")
for f in files:
    if any(x in f for x in ("analytical_models", "surrogate_models", "surrogate_engine",
                            "profile_generator", "pvt_state", "objectives")):
        for i, l in enumerate(open(f, encoding="utf-8", errors="replace"), 1):
            if "gravity_factor" in l:
                print("  ", f.split("co2eor_optimizer")[-1], i, l.strip()[:90])
