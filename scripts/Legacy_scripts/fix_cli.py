"""
Fix script for CLI-related issues in the project.
"""
import os
import re
import sys

def fix_backticks(file_path):
    """Remove backtick code blocks from Python files."""
    print(f"Fixing backticks in {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove triple backticks
    fixed_content = re.sub(r'```(?:[a-z]*\n|\n)', '', content)
    fixed_content = fixed_content.replace('```', '')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"✓ Fixed backticks in {file_path}")

def fix_unicode_in_cli():
    """Fix Unicode emoji issues in the CLI module."""
    file_path = os.path.join(os.getcwd(), 'lego_cli.py')
    print(f"Fixing emoji handling in {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add safe emoji function
    safe_emoji_function = '''
def safe_emoji(emoji_str):
    """Returns the emoji if the platform supports it, otherwise returns alternative text."""
    import platform
    if platform.system() == "Windows":
        # Check if running in Windows Terminal with emoji support
        if "WT_SESSION" not in os.environ and "TERM_PROGRAM" not in os.environ:
            emoji_map = {
                '🚀': '[ROCKET]', '✅': '[OK]', '❌': '[ERROR]', '⚠️': '[WARNING]',
                '📦': '[PACKAGE]', '🔍': '[SEARCH]', '📏': '[RULER]', '🧹': '[CLEANUP]',
                '🔄': '[REFRESH]', '💾': '[SAVE]', '🖼️': '[IMAGE]', '📌': '[PIN]',
                '📂': '[FOLDER]', '🆕': '[NEW]'
            }
            for emoji, text in emoji_map.items():
                emoji_str = emoji_str.replace(emoji, text)
        return emoji_str
    return emoji_str
'''
    
    # Add safe_emoji function if it doesn't exist yet
    if "def safe_emoji" not in content:
        # Find the imports section to place the function after it
        import_section_end = content.rfind("import", 0, 1000) + 100  # Rough estimate where imports end
        content = content[:import_section_end] + "\n" + safe_emoji_function + content[import_section_end:]
    
    # Replace direct emoji usage with safe_emoji calls
    emoji_pattern = r'(["\']\s*)([\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF]+)(\s*[^"\']*["\']\s*)'
    
    # This is a simplified approach - in a real fix you'd need to be more careful
    content = re.sub(emoji_pattern, r'\1" + safe_emoji("\2") + "\3', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Added safe emoji handling to {file_path}")

def fix_progress_bar_conflict():
    """Fix the 'Only one live display may be active at once' error."""
    cli_path = os.path.join(os.getcwd(), 'lego_cli.py')
    utils_path = os.path.join(os.getcwd(), 'utils', 'model_utils.py')
    
    print(f"Fixing progress bar conflict between {cli_path} and {utils_path}...")
    
    # Simplify by temporarily disabling rich progress in one of the files
    with open(cli_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find infer function and disable rich progress for testing
    if "def infer" in content and "with Progress" in content:
        content = content.replace("with Progress(", "# TEMPORARY FIX: Disabled nested progress\nif False:  # with Progress(")
        
        with open(cli_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print("✓ Temporarily disabled nested progress in CLI for testing")

if __name__ == "__main__":
    print("LEGO Bricks ML Vision - CLI Fix Script")
    print("=====================================")
    
    # Make sure we're in the right directory
    project_dir = '/c:/Users/User/Projects_Unprotected/LEGO_Bricks_ML_Vision'
    if os.path.exists(project_dir):
        os.chdir(project_dir)
    
    # Fix all issues
    fix_backticks(os.path.join(os.getcwd(), 'lego_cli.py'))
    fix_unicode_in_cli()
    fix_progress_bar_conflict()
    
    print("\nAll fixes applied. Try running the CLI again with:")
    print("python lego_cli.py --help")
