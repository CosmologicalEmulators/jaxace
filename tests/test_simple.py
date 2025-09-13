"""
Simple tests that don't require external dependencies to verify basic functionality.
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that basic imports work."""
    try:
        from jaxace.utils import validate_nn_dict_structure, safe_dict_access
        print("✓ Basic utils imported successfully")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        assert False, f"Import failed: {e}"

def test_validation():
    """Test validation functions that don't require JAX."""
    from jaxace.utils import validate_nn_dict_structure, validate_layer_structure

    # Valid structure
    valid_dict = {
        "n_input_features": 10,
        "n_output_features": 100,
        "n_hidden_layers": 2,
        "layers": {
            "layer_1": {"n_neurons": 64, "activation_function": "tanh"},
            "layer_2": {"n_neurons": 32, "activation_function": "relu"}
        }
    }

    try:
        validate_nn_dict_structure(valid_dict)
        print("✓ Valid NN dict passed validation")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        assert False, f"Validation failed: {e}"

    # Test invalid structure
    invalid_dict = valid_dict.copy()
    del invalid_dict["n_input_features"]

    try:
        validate_nn_dict_structure(invalid_dict)
        print("✗ Invalid dict should have failed validation")
        assert False, "Invalid dict should have failed validation"
    except ValueError:
        print("✓ Invalid dict correctly rejected")

def test_safe_dict_access():
    """Test safe dictionary access."""
    from jaxace.utils import safe_dict_access

    test_dict = {
        "level1": {
            "level2": {
                "level3": "value"
            }
        }
    }

    # Valid access
    result = safe_dict_access(test_dict, "level1", "level2", "level3")
    if result == "value":
        print("✓ Safe dict access works for valid path")
    else:
        print(f"✗ Expected 'value', got {result}")
        assert False, f"Expected 'value', got {result}"

    # Invalid access with default
    result = safe_dict_access(test_dict, "level1", "invalid", default="default")
    if result == "default":
        print("✓ Safe dict access returns default for invalid path")
    else:
        print(f"✗ Expected 'default', got {result}")
        assert False, f"Expected 'default', got {result}"

def test_nn_json():
    """Test that test NN JSON file is valid."""
    test_file = Path(__file__).parent / "testNN.json"

    if not test_file.exists():
        print(f"✗ Test NN file not found: {test_file}")
        assert False, f"Test NN file not found: {test_file}"

    try:
        with open(test_file, 'r') as f:
            nn_dict = json.load(f)

        from jaxace.utils import validate_nn_dict_structure
        validate_nn_dict_structure(nn_dict)
        print("✓ Test NN JSON is valid")
    except Exception as e:
        print(f"✗ Test NN JSON validation failed: {e}")
        assert False, f"Test NN JSON validation failed: {e}"

def main():
    """Run simple tests."""
    print("Running simple jaxace tests (no JAX dependencies)...")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Validation", test_validation),
        ("Safe Dict Access", test_safe_dict_access),
        ("NN JSON", test_nn_json)
    ]
    
    for name, test_func in tests:
        print(f"\nTesting {name}:")
        if not test_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All simple tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())