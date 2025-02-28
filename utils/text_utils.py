"""
Text utilities for the LEGO Bricks ML Vision project.
"""
import sys
import platform

def safe_emoji(emoji_str):
    """
    Returns the emoji if the platform supports it, otherwise returns a fallback text.
    
    Args:
        emoji_str: String containing emoji characters
        
    Returns:
        String with emojis if supported, or text alternatives if not
    """
    # Check if Windows terminal supports Unicode properly
    if platform.system() == "Windows":
        # Check if running in Windows Terminal which has better Unicode support
        if "WT_SESSION" not in os.environ and "TERM_PROGRAM" not in os.environ:
            # Classic console without proper Unicode support
            emoji_map = {
                '🚀': '[ROCKET]',
                '✅': '[OK]',
                '❌': '[ERROR]',
                '⚠️': '[WARNING]',
                '📦': '[PACKAGE]',
                '🔍': '[SEARCH]',
                '📏': '[RULER]',
                '🧹': '[CLEANUP]',
                '🔄': '[REFRESH]',
                '💾': '[SAVE]',
                '🖼️': '[IMAGE]',
                '📌': '[PIN]',
                '📂': '[FOLDER]',
                '🆕': '[NEW]',
                # Add more emoji mappings as needed
            }
            
            for emoji, text in emoji_map.items():
                emoji_str = emoji_str.replace(emoji, text)
            
            return emoji_str
    
    # Platform supports emoji properly
    return emoji_str
