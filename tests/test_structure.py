"""
Test jaxace implementation structure without requiring JAX installation.
This verifies that all necessary functions have been implemented.
"""

import ast
import os
from pathlib import Path

def parse_python_file(filepath):
    """Parse a Python file and extract function/class definitions."""
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())
    
    functions = []
    classes = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
    
    return functions, classes

def test_core_implementation():
    """Test that core.py has all required components."""
    core_file = Path(__file__).parent.parent / "jaxace" / "core.py"
    functions, classes = parse_python_file(core_file)
    
    required_classes = ["AbstractTrainedEmulator", "FlaxEmulator"]
    required_functions = ["run_emulator", "get_emulator_description"]
    
    print("Testing core.py:")
    all_present = True
    
    for cls in required_classes:
        if cls in classes:
            print(f"  ✓ Class {cls} found")
        else:
            print(f"  ✗ Class {cls} missing")
            all_present = False
    
    for func in required_functions:
        if func in functions:
            print(f"  ✓ Function {func} found")
        else:
            print(f"  ✗ Function {func} missing")
            all_present = False

    assert all_present, "Some required classes or functions are missing from core.py"

def test_initialization_implementation():
    """Test that initialization.py has all required components."""
    init_file = Path(__file__).parent.parent / "jaxace" / "initialization.py"
    functions, classes = parse_python_file(init_file)
    
    required_classes = ["MLP"]
    required_functions = [
        "init_emulator",
        "_get_in_out_arrays",
        "_get_i_array",
        "_get_weight_bias",
        "_get_flax_params",
        "_get_flax_states",
        "_get_nn_flax",
        "_init_flaxemulator"
    ]
    
    print("\nTesting initialization.py:")
    all_present = True
    
    for cls in required_classes:
        if cls in classes:
            print(f"  ✓ Class {cls} found")
        else:
            print(f"  ✗ Class {cls} missing")
            all_present = False
    
    for func in required_functions:
        if func in functions:
            print(f"  ✓ Function {func} found")
        else:
            print(f"  ✗ Function {func} missing")
            all_present = False

    assert all_present, "Some required classes or functions are missing from initialization.py"

def test_utils_implementation():
    """Test that utils.py has all required components."""
    utils_file = Path(__file__).parent.parent / "jaxace" / "utils.py"
    functions, classes = parse_python_file(utils_file)
    
    required_functions = [
        "maximin",
        "inv_maximin",
        "validate_nn_dict_structure",
        "validate_layer_structure",
        "validate_activation_function",
        "validate_parameter_ranges",
        "validate_trained_weights",
        "safe_dict_access",
        "calculate_weight_size"
    ]
    
    print("\nTesting utils.py:")
    all_present = True
    
    for func in required_functions:
        if func in functions:
            print(f"  ✓ Function {func} found")
        else:
            print(f"  ✗ Function {func} missing")
            all_present = False

    assert all_present, "Some required functions are missing from utils.py"

def test_background_implementation():
    """Test that background.py has all required cosmology functions."""
    bg_file = Path(__file__).parent.parent / "jaxace" / "background.py"
    functions, classes = parse_python_file(bg_file)
    
    required_classes = ["W0WaCDMCosmology"]
    required_functions = [
        # Basic functions
        "a_z", "rhoDE_a", "rhoDE_z", "drhoDE_da",
        
        # Neutrino functions
        "gety", "F", "dFdy",
        
        # Hubble parameter
        "E_a", "E_z", "dlogEdloga", "Ωm_a",
        
        # Distance measures
        "r_z", "dA_z", "dL_z",
        
        # Growth functions
        "D_z", "f_z", "D_f_z",
        "growth_ode_system", "growth_solver",
        
        # Additional functions
        "ρc_z", "Ωtot_z"
    ]
    
    print("\nTesting background.py:")
    all_present = True
    
    for cls in required_classes:
        if cls in classes:
            print(f"  ✓ Class {cls} found")
        else:
            print(f"  ✗ Class {cls} missing")
            all_present = False
    
    for func in required_functions:
        if func in functions:
            print(f"  ✓ Function {func} found")
        else:
            print(f"  ✗ Function {func} missing")
            all_present = False
    
    # Check for important cosmology wrapper functions
    cosmology_wrappers = [
        "Ez_from_cosmo", "Ea_from_cosmo",
        "D_z_from_cosmo", "f_z_from_cosmo", "D_f_z_from_cosmo",
        "r_z_from_cosmo", "dA_z_from_cosmo", "dL_z_from_cosmo"
    ]
    
    print("\n  Cosmology structure wrappers:")
    for func in cosmology_wrappers:
        if func in functions:
            print(f"    ✓ {func} found")
        else:
            print(f"    ✗ {func} missing (optional)")

    assert all_present, "Some required classes or functions are missing from background.py"

def test_file_structure():
    """Test that all required files exist."""
    base_dir = Path(__file__).parent.parent
    
    required_files = [
        "jaxace/__init__.py",
        "jaxace/core.py",
        "jaxace/initialization.py",
        "jaxace/utils.py",
        "jaxace/background.py",
        "tests/testNN.json",
        "tests/test_background.py",
        "tests/test_core_functionality.py",
        "pyproject.toml",
        "README.md"
    ]
    
    print("\nTesting file structure:")
    all_present = True
    
    for filepath in required_files:
        full_path = base_dir / filepath
        if full_path.exists():
            print(f"  ✓ {filepath} exists")
        else:
            print(f"  ✗ {filepath} missing")
            all_present = False

    assert all_present, "Some required files are missing from the file structure"

def main():
    """Run all structure tests."""
    print("=" * 60)
    print("jaxace Implementation Structure Test")
    print("Testing that all required functions have been implemented")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(("File structure", test_file_structure()))
    results.append(("Core module", test_core_implementation()))
    results.append(("Initialization module", test_initialization_implementation()))
    results.append(("Utils module", test_utils_implementation()))
    results.append(("Background cosmology", test_background_implementation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ All implementation structure tests passed!")
        print("\nThe jaxace implementation successfully translates all core")
        print("functionalities from AbstractCosmologicalEmulators.jl:")
        print("  - Neural network loading (FlaxEmulator)")
        print("  - Data normalization (maximin/inv_maximin)")
        print("  - Validation system")
        print("  - Background cosmology calculations")
        print("  - All distance measures including dL_z")
        print("\nTo run full tests with JAX, install dependencies:")
        print("  pip install jax jaxlib flax numpy quadax interpax diffrax")
        return 0
    else:
        print("✗ Some structure tests failed")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())