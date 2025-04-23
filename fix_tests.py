#!/usr/bin/env python3
# Script to add debug_files_only attribute to all test files
import os
import re

def add_debug_files_only_to_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all occurrences of debug attribute setting
    pattern = r'(\s+args\.debug\s*=\s*(True|False)(?!\s*\n\s+args\.debug_files_only))'
    
    # Replace with debug and debug_files_only
    replacement = r'\1\n        args.debug_files_only = False'
    modified_content = re.sub(pattern, replacement, content)
    
    # Write back to the file if changes were made
    if content != modified_content:
        with open(file_path, 'w') as f:
            f.write(modified_content)
        print(f"Updated {file_path}")
    else:
        print(f"No changes needed in {file_path}")

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and file.startswith('test_'):
                file_path = os.path.join(root, file)
                add_debug_files_only_to_file(file_path)

if __name__ == "__main__":
    tests_dir = "/home/jacsan/utv/lang/tests"
    process_directory(tests_dir)
    print("Done processing all test files")
