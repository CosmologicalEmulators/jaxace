"""Supplement, never replace, the frozen Julia boundary derivatives.

ForwardDiff 1.4.6 compares Dual(hi, 1) > hi as true. Julia's basis_row then
takes its outside-zero branch even though the primal query is at the endpoint.
SciPy supplies the explicitly intended interior derivative at equality.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[1] / "tests/data/bspline_api"
names = ("polynomial", "simple", "double", "triple")
paths = [ROOT / f"{name}_grad_query_interior.txt" for name in names]
metadata = ROOT / "boundary_metadata.json"
if metadata.exists() or any(path.exists() for path in paths):
    raise RuntimeError("boundary supplements already exist; refusing overwrite")

record = {
    "scipy_version": scipy.__version__,
    "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "convention": "interior derivative at endpoints; right span at interior knots; zero outside",
    "julia_issue": "ForwardDiff 1.4.6 Dual(hi,1)>hi changes basis_row to the outside-zero branch",
    "cases": {},
}
for name, path in zip(names, paths):
    def load(field):
        return np.loadtxt(ROOT / f"{name}_{field}.txt")
    knots, q, cot = load("knots"), load("query"), load("query_cotangent")
    n = len(knots) - 4
    spline = BSpline(knots, np.eye(n), 3, extrapolate=False)
    inside = (q >= knots[3]) & (q <= knots[-4])
    matrix = np.zeros((len(q), n))
    derivative = np.zeros_like(matrix)
    matrix[inside] = spline(q[inside])
    derivative[inside] = spline(q[inside], nu=1)
    np.testing.assert_allclose(matrix, load("matrix"), rtol=2e-13, atol=2e-14)
    gradient = np.sum(derivative * cot, axis=1)
    original = load("grad_query")
    right = q == knots[-4]
    np.testing.assert_allclose(gradient[~right], original[~right], rtol=2e-11, atol=2e-12)
    np.testing.assert_array_equal(original[right], 0)
    np.savetxt(path, gradient)
    record["cases"][name] = {
        "right_endpoint": float(knots[-4]),
        "saved_julia_gradient": original[right].tolist(),
        "interior_gradient": gradient[right].tolist(),
        "inputs_sha256": {field: hashlib.sha256((ROOT / f"{name}_{field}.txt").read_bytes()).hexdigest()
                          for field in ("knots", "query", "query_cotangent", "grad_query", "matrix")},
    }
    print(name, "Julia right-end derivative", original[right], "interior", gradient[right])
metadata.write_text(json.dumps(record, indent=2) + "\n")
