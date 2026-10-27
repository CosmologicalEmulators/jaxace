"""Three-way paired benchmarks for the full compact-plan port.

Requires the preserved pre-port source snapshot at
.benchmarks/julia_parity_20261026/before. Dense baseline is pinned to 368596b.
Compile/validate first, then cool and measure A/B/C/C/B/A with synchronized
pytest-benchmark rounds. CPU affinity and absence of competing compute matter.
"""
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxace
from benchmarks.test_banded_bspline_benchmarks import (
    cool, temperature, throttle_counts, check_compute_idle,
)

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / ".benchmarks/julia_parity_20261026/before/jaxace"
CASES = {
    "tiny": (4, 7, 1),
    "small_matrix": (8, 17, 3),
    "production_vector": (81, 129, 1),
    "production_batch": (81, 129, 8192),
    "long_queries": (81, 4096, 3),
    "large_geometry": (512, 129, 3),
}
OPERATIONS = (
    "coefficients", "gradient_values", "gradient_sites", "interpolation",
    "interpolation_sites_gradient", "plan_construction", "plan_application", "plan_gradient",
)
TRIALS = (("dense",0),("banded_dense",0),("compact",0),
          ("compact",1),("banded_dense",1),("dense",1))


@pytest.fixture(scope="module")
def implementations():
    source = subprocess.check_output(["git","show","368596b:jaxace/utils.py"], cwd=ROOT, text=True)
    dense = types.ModuleType("_jaxace_dense_parity_reference")
    sys.modules[dense.__name__] = dense
    exec(compile(source,"git:368596b:jaxace/utils.py","exec"),dense.__dict__)
    if not (BASELINE / "utils.py").is_file():
        raise RuntimeError("missing preserved pre-compact baseline; do not substitute current code")
    package = types.ModuleType("_jaxace_precompact_parity_reference")
    package.__path__ = [str(BASELINE)]
    sys.modules[package.__name__] = package
    before = importlib.import_module(package.__name__ + ".utils")
    return {"dense":dense, "banded_dense":before, "compact":jaxace}, source


def arguments(case):
    n, count, columns = CASES[case]
    if n == 81:
        source = np.loadtxt(ROOT / "tests/data/bspline_banded/julia_n81_inputs.txt")
        t = jnp.asarray(source[:,0])
    else:
        t = jnp.linspace(0.,1.,n)**1.35
    q = jnp.linspace(t[0],t[-1],count)
    u = (jnp.sin(1.3*t)+.2*jnp.cos(.7*t))[:,None] * jnp.linspace(.5,1.5,columns)[None,:]
    if columns == 1:
        u = u[:,0]
    return t,q,u


@pytest.fixture(scope="module")
def compiled_cases(implementations):
    modules, dense_source = implementations
    cache = {}

    def prepare(case, operation):
        key = case,operation
        if key in cache:
            return cache[key]
        t,q,u = arguments(case)
        functions = {}
        for label,module in modules.items():
            coefficients = lambda x,y,module=module: module.CubicBSpline(x,y).coefficients
            interpolation = module.cubic_b_spline_interpolation
            if operation == "coefficients":
                f,args = coefficients,(u,t)
            elif operation == "gradient_values":
                f,args = jax.grad(lambda x,y,coefficients=coefficients: jnp.mean(coefficients(x,y)**2)),(u,t)
            elif operation == "gradient_sites":
                f,args = jax.grad(lambda y,x,coefficients=coefficients: jnp.mean(coefficients(x,y)**2)),(t,u)
            elif operation == "interpolation":
                f,args = interpolation,(u,t,q)
            elif operation == "interpolation_sites_gradient":
                f,args = jax.grad(lambda y,x,z,interpolation=interpolation: jnp.mean(interpolation(x,y,z)**2)),(t,u,q)
            elif operation == "plan_construction":
                f,args = module.CubicBSplinePlan,(t,q)
            else:
                plan = jax.block_until_ready(module.CubicBSplinePlan(t,q))
                if operation == "plan_application":
                    f,args = (lambda p,x:p(x)),(plan,u)
                else:
                    f,args = jax.grad(lambda x,p:jnp.mean(p(x)**2)),(u,plan)
            compiled = jax.jit(f)
            result = jax.block_until_ready(compiled(*args))
            comparison = jax.block_until_ready(result(u)) if operation == "plan_construction" else result
            functions[label] = compiled,args,np.asarray(comparison)
        reference = functions["dense"][2]
        for label in ("banded_dense","compact"):
            np.testing.assert_allclose(functions[label][2],reference,rtol=2e-10,atol=2e-12,
                                       err_msg=f"{case} {operation} {label}")
        cache[key] = functions
        return functions

    return prepare


@pytest.mark.parametrize("case,operation,implementation,repeat", [
    (case,operation,implementation,repeat)
    for case in CASES for operation in OPERATIONS
    # Same geometry: no need to duplicate construction just because M differs.
    if not (case == "production_batch" and operation == "plan_construction")
    for implementation,repeat in TRIALS
])
def test_parity_runtime(benchmark, implementations, compiled_cases, case, operation, implementation, repeat):
    fn,args,_ = compiled_cases(case,operation)[implementation]
    before = cool()
    processes = check_compute_idle()
    counters = throttle_counts()
    benchmark.pedantic(lambda:jax.block_until_ready(fn(*args)), rounds=30, iterations=1, warmup_rounds=5)
    after = temperature()
    check_compute_idle(processes)
    assert counters == throttle_counts(), "thermal throttling; reject this trial"
    assert after is None or max(before,after) < 95, "hot trial; reject this trial"
    n,q,m = CASES[case]
    paths = [ROOT / p for p in ("jaxace/_bspline.py","jaxace/_bspline_banded.py",
                                "jaxace/utils.py","tests/bspline_dense_reference.py")]
    paths += [BASELINE / "utils.py", BASELINE / "_bspline_banded.py"]
    benchmark.extra_info.update({
        "case":case, "operation":operation, "implementation":implementation, "repeat":repeat,
        "N":n,"Q":q,"M":None if operation=="plan_construction" else m,
        "dtype":"float64","jax_version":jax.__version__,"device":str(jax.devices()[0]),
        "cpu_affinity":sorted(os.sched_getaffinity(0)),"before_c":before,"after_c":after,
        "dynamic_geometry":operation not in ("plan_application","plan_gradient"),
        "prepared_plan_passed_as_argument":operation in ("plan_application","plan_gradient"),
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "dense_commit":"368596b","dense_source_sha256":hashlib.sha256(implementations[1].encode()).hexdigest(),
    })
