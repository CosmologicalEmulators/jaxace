"""Summarize complete paired trials; fail for missing cells or regressions.

Usage: python scripts/summarize_bspline_parity_benchmarks.py results.json
The JSON must contain the full test_bspline_parity_benchmarks.py matrix.
"""
import json
from pathlib import Path
import sys
import numpy as np

data = json.loads(Path(sys.argv[1]).read_text())
groups = {}
for row in data["benchmarks"]:
    meta = row["extra_info"]
    key = meta["case"],meta["operation"]
    values = groups.setdefault(key,{})
    trial = meta["implementation"],meta["repeat"]
    if trial in values:
        raise RuntimeError(f"duplicate trial: {key} {trial}")
    values[trial] = row["stats"]["median"]

cases = ("tiny","small_matrix","production_vector","production_batch","long_queries","large_geometry")
operations = ("coefficients","gradient_values","gradient_sites","interpolation",
              "interpolation_sites_gradient","plan_construction","plan_application","plan_gradient")
expected = {(c,o) for c in cases for o in operations
            if not (c=="production_batch" and o=="plan_construction")}
assert set(groups)==expected, ("incomplete matrix",expected-set(groups),set(groups)-expected)
trials = {(name,i) for name in ("dense","banded_dense","compact") for i in (0,1)}
failed = False
print("case operation dense_ms banded_dense_ms compact_ms dense_over_compact banded_over_compact max_repeat_drift_pct status")
for key,values in groups.items():
    assert set(values)==trials, ("incomplete pair",key,set(values))
    means,drifts = {},[]
    for name in ("dense","banded_dense","compact"):
        pair = [values[name,i] for i in (0,1)]
        means[name] = np.mean(pair)
        drifts.append(abs(pair[1]-pair[0])/min(pair))
    slowdown = means["compact"]/means["dense"]-1
    drift = max(drifts)
    status = "UNSTABLE" if drift>.10 else "REGRESSION" if slowdown>.10 else "WARN" if max(slowdown,drift)>.05 else "PASS"
    failed |= status in ("UNSTABLE","REGRESSION")
    print(*key,*(f"{1000*means[name]:.6f}" for name in ("dense","banded_dense","compact")),
          f"{means['dense']/means['compact']:.3f}",f"{means['banded_dense']/means['compact']:.3f}",
          f"{100*drift:.2f}",status)
raise SystemExit(1 if failed else 0)
