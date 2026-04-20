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
        print("File exists: " + file)
    else:
        print("File missing: " + file)

print("Structure test completed!")
