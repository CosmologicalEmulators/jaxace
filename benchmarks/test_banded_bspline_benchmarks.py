"""Matched, synchronized old-dense versus banded workloads at N81×8192.

All sites, values and queries are dynamic JIT arguments. No constant matrix
factorization is substituted for changing geometry. The dense oracle is the
pre-port construction from utils.py; its basis recursion remains unchanged.
Run pinned and thermally stabilized; these tests do not certify host isolation.
"""
from pathlib import Path
import hashlib
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxace import CubicBSpline, CubicBSplinePlan, cubic_b_spline_interpolation
from tests.bspline_dense_reference import _cubic_b_spline_knots, _cubic_b_spline_basis_matrix

jax.config.update("jax_enable_x64", True)
DATA = Path(__file__).resolve().parents[1] / "tests/data/bspline_banded"


def temperature():
    paths = list(Path("/sys/devices/platform").glob("coretemp.*/hwmon/hwmon*/temp*_input"))
    return max(float(p.read_text()) / 1000 for p in paths) if paths else None


def throttle_counts():
    return {str(p): int(p.read_text()) for p in
            Path("/sys/devices/system/cpu").glob("cpu[0-9]*/thermal_throttle/*_throttle_count")}


def compute_snapshot():
    """Observe other Python/Julia CPU ticks, including passive monitors."""
    proc = Path("/proc")
    snapshot = {}
    for path in proc.glob("[0-9]*/comm"):
        try:
            name = path.read_text().strip()
            fields = (path.parent / "stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        pid = int(path.parent.name)
        if pid != os.getpid() and (name == "julia" or name.startswith("python")):
            snapshot[pid] = (int(fields[19]), int(fields[11]) + int(fields[12]))
    return snapshot


def check_compute_idle(before=None):
    """Pre/post activity observation, not a claim of continuous exclusivity."""
    if before is None:
        before = compute_snapshot()
        time.sleep(.2)
    after = compute_snapshot()
    activity = {pid: ticks - (before[pid][1] if pid in before and before[pid][0] == start else 0)
                for pid, (start, ticks) in after.items()}
    active = {pid: ticks for pid, ticks in activity.items() if ticks > 0}
    if active:
        raise RuntimeError(f"other Julia/Python CPU activity: {active}; timings rejected")
    return after


def cool():
    deadline, stable = time.monotonic() + 180, 0
    while True:
        value = temperature()
        if value is None:
            return None
        stable = stable + 1 if value <= 55 else 0
        if stable >= 2:
            return value
        if time.monotonic() > deadline:
            raise RuntimeError("CPU did not cool; no timing accepted")
        time.sleep(1)


def dense_coefficients(u, t):
    return jnp.linalg.solve(_cubic_b_spline_basis_matrix(_cubic_b_spline_knots(t), t), u)


def dense_evaluate(u, t, q):
    return _cubic_b_spline_basis_matrix(_cubic_b_spline_knots(t), q) @ dense_coefficients(u, t)


def dense_plan(t, q):
    knots = _cubic_b_spline_knots(t)
    a = _cubic_b_spline_basis_matrix(knots, t)
    e = _cubic_b_spline_basis_matrix(knots, q)
    return jnp.linalg.solve(a.T, e.T).T


@pytest.fixture(scope="module")
def workloads():
    inputs = np.loadtxt(DATA / "julia_n81_inputs.txt")
    t = jnp.asarray(inputs[:, 0])
    scales = jnp.linspace(.5, 1.5, 8192)
    u = jnp.asarray(inputs[:, 1:2]) * scales[None, :]
    q = jnp.linspace(t[0], t[-1], 129)
    band = lambda x, y: CubicBSpline(x, y).coefficients
    functions = {
        "coefficients": (dense_coefficients, band, (u, t)),
        "gradient_values": (
            jax.grad(lambda x, y: jnp.mean(dense_coefficients(x, y)**2)),
            jax.grad(lambda x, y: jnp.mean(band(x, y)**2)), (u, t)),
        "gradient_sites": (
            jax.grad(lambda y, x: jnp.mean(dense_coefficients(x, y)**2)),
            jax.grad(lambda y, x: jnp.mean(band(x, y)**2)), (t, u)),
        "interpolation": (dense_evaluate, cubic_b_spline_interpolation, (u, t, q)),
        "plan_construction": (dense_plan, lambda t, q: CubicBSplinePlan(t, q).operator, (t, q)),
    }
    compiled = {}
    for name, (old, new, args) in functions.items():
        old, new = jax.jit(old), jax.jit(new)
        reference, candidate = old(*args).block_until_ready(), new(*args).block_until_ready()
        np.testing.assert_allclose(candidate, reference, rtol=2e-10, atol=2e-12)
        compiled[name] = (old, new, args)
    return compiled


@pytest.mark.parametrize("operation,implementation,repeat", [
    (operation, implementation, repeat)
    for operation in ("coefficients", "gradient_values", "gradient_sites", "interpolation", "plan_construction")
    for implementation, repeat in (("dense", 0), ("banded", 0), ("banded", 1), ("dense", 1))
])
def test_banded_runtime(benchmark, workloads, operation, implementation, repeat):
    old, new, args = workloads[operation]
    fn = old if implementation == "dense" else new

    def run():
        return fn(*args).block_until_ready()

    before = cool()
    processes = check_compute_idle()
    counts = throttle_counts()
    benchmark.pedantic(run, rounds=30, iterations=1, warmup_rounds=5)
    after = temperature()
    counters_unchanged = counts == throttle_counts()
    root = Path(__file__).resolve().parents[1]
    hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
              for p in ("jaxace/utils.py", "jaxace/_bspline.py", "jaxace/_bspline_banded.py",
                        "tests/bspline_dense_reference.py")}
    benchmark.extra_info.update({"N": 81, "M": 129 if operation == "plan_construction" else 8192, "Q": 129,
                                 "device": str(jax.devices()[0]),
                                 "jax_version": jax.__version__, "source_sha256": hashes,
                                 "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                                 "dynamic_geometry": True, "dtype": "float64",
                                 "repeat": repeat, "before_c": before, "after_c": after,
                                 "throttle_counts_unchanged": counters_unchanged})
    check_compute_idle(processes)
    assert counters_unchanged, "throttling during trial; reject these measurements"
    assert after is None or max(before, after) < 95, "hot trial; reject these measurements"
