"""Quantify tautological vs production-exercising tests in tests/scientific/."""
import ast
import pathlib

ROOT = pathlib.Path(r"D:\rep\4.6\co2eor_optimizer\tests\scientific")
PROD_PREFIXES = ("core", "analysis", "evaluation", "utils")


def prod_names(tree):
    """Module-level production imports."""
    out = set()
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith(PROD_PREFIXES):
            for a in n.names:
                out.add(a.asname or a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                nm = (a.asname or a.name).split(".")[0]
                if nm in PROD_PREFIXES:
                    out.add(nm)
    return out


def fn_local_names(fn):
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith(PROD_PREFIXES):
            for a in n.names:
                out.add(a.asname or a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                nm = (a.asname or a.name).split(".")[0]
                if nm in PROD_PREFIXES:
                    out.add(nm)
    return out


rows = []
for p in sorted(ROOT.rglob("test_*.py")):
    src = p.read_text(encoding="utf-8")
    tree = ast.parse(src)
    mod = prod_names(tree)
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]:
        body = ast.get_source_segment(src, fn) or ""
        local = fn_local_names(fn)
        avail = mod | local
        # strip import lines before searching for usage
        stripped = "\n".join(
            l for l in body.splitlines() if not l.strip().startswith(("from ", "import "))
        )
        used = sorted(s for s in avail if s in stripped)
        rows.append((str(p.relative_to(ROOT)), fn.name, sorted(avail), used))

n = len(rows)
taut = [r for r in rows if r[2] and not r[3]]
none_ = [r for r in rows if not r[2]]
ok = [r for r in rows if r[3]]
print(f"total test functions in tests/scientific: {n}")
print(f"A) file/function imports production code but never uses it (tautological): {len(taut)}")
print(f"B) no production import at all (self-contained math or pure mocks):        {len(none_)}")
print(f"C) exercises a production symbol:                                          {len(ok)}")
print("\n--- A) TAUTOLOGICAL ---")
for r in taut:
    print(f"  {r[0]} :: {r[1]}\n      imported-never-used: {r[2]}")
print("\n--- B) NO PRODUCTION IMPORT ---")
for r in none_:
    print(f"  {r[0]} :: {r[1]}")
print("\n--- C) EXERCISES PRODUCTION CODE ---")
for r in ok:
    print(f"  {r[0]} :: {r[1]} -> {r[3]}")
