"""
Simple utility script to fix backtick syntax errors in Python files.
"""
import os
import re

def fix_backticks(file_path):
    """Remove backtick code blocks from Python files."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove triple backticks
    fixed_content = re.sub(r'```(?:[a-z]*\n|\n)', '', content)
    fixed_content = fixed_content.replace('```', '')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"Fixed backticks in {file_path}")

if __name__ == "__main__":
    # Fix lego_cli.py
    fix_backticks('/c:/Users/User/Projects_Unprotected/LEGO_Bricks_ML_Vision/lego_cli.py')
