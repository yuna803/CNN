"""
Test script to verify the code structure is correct
"""

# Test 1: Check if all files exist
import os

required_files = [
    'config.py',
    'dataset.py',
    'model.py',
    'utils.py',
    'main.py'
]

print("Checking if required files exist...")
for file in required_files:
    if os.path.exists(file):
        print(f"✓ {file} exists")
    else:
        print(f"✗ {file} missing")

# Test 2: Check if all required functions are defined
print("\nChecking if required functions are defined...")

# Check dataset.py
print("\nChecking dataset.py...")
try:
    from dataset import data_preprocess, create_mini_batches, create_dataloaders, one_hot_encode
    print("✓ data_preprocess function exists")
    print("✓ create_mini_batches function exists")
    print("✓ create_dataloaders function exists")
    print("✓ one_hot_encode function exists")
except ImportError as e:
    print(f"✗ Error importing dataset.py: {e}")

# Check model.py
print("\nChecking model.py...")
try:
    from model import build_cnn_model, CNN
    print("✓ build_cnn_model function exists")
    print("✓ CNN class exists")
except ImportError as e:
    print(f"✗ Error importing model.py: {e}")

# Check utils.py
print("\nChecking utils.py...")
try:
    from utils import (
        compile_model, train_model, train_with_different_optimizers,
        plot_performance, evaluate_model
    )
    print("✓ compile_model function exists")
    print("✓ train_model function exists")
    print("✓ train_with_different_optimizers function exists")
    print("✓ plot_performance function exists")
    print("✓ evaluate_model function exists")
except ImportError as e:
    print(f"✗ Error importing utils.py: {e}")

# Check main.py
print("\nChecking main.py...")
try:
    from main import main
    print("✓ main function exists")
except ImportError as e:
    print(f"✗ Error importing main.py: {e}")

print("\nStructure test completed!")
