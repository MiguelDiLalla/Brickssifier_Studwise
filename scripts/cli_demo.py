"""
LEGO Bricks ML Vision - CLI Demo Script

This script demonstrates how to use the LEGO Bricks ML Vision command-line interface.
It contains examples for the most common operations:
- Running brick detection on sample images
- Running stud detection on brick crops
- Running the full inference pipeline
- Exploring data processing utilities

Usage:
    python scripts/cli_demo.py

Note: This script assumes you're running from the project root directory.
"""
import os
import sys
import subprocess
import time

# Add project root to PATH if needed
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Try to import rich_utils for enhanced console output
try:
    from utils.rich_utils import (
        RICH_AVAILABLE, console, create_status_panel,
        display_results_table, create_progress
    )
except ImportError:
    # Fall back to direct rich import
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown
        from rich.syntax import Syntax
        from rich.table import Table
        RICH_AVAILABLE = True
        console = Console()
    except ImportError:
        RICH_AVAILABLE = False
        print("Note: Install 'rich' package for enhanced output: pip install rich")

def print_header(title):
    """
    Print a section header with consistent formatting.
    
    Args:
        title (str): The section title to display
    """
    if RICH_AVAILABLE:
        create_status_panel(title)
    else:
        print("\n" + "="*40)
        print(title)
        print("="*40 + "\n")

def run_command(command, description=None):
    """
    Run a CLI command and print its output with rich formatting.
    
    Args:
        command (list): Command to run as a list of strings
        description (str, optional): Description of what the command does
    """
    if description and RICH_AVAILABLE:
        console.print(f"[italic yellow]{description}[/italic yellow]")
    
    cmd_str = " ".join(command)
    
    if RICH_AVAILABLE:
        syntax = Syntax(cmd_str, "bash", theme="monokai", 
                       line_numbers=False, word_wrap=True)
        console.print(syntax)
        console.print("[bold green]Running command...[/bold green]")
        
        # Add small delay for readability
        time.sleep(0.5)
    else:
        print(f"\n\n{'='*40}")
        print(f"Running: {cmd_str}")
        print(f"{'='*40}\n")
    
    # Run the command
    result = subprocess.run(command, capture_output=RICH_AVAILABLE)
    
    if RICH_AVAILABLE:
        if result.returncode == 0:
            if result.stdout:
                console.print(result.stdout.decode('utf-8', errors='replace'))
        else:
            console.print("[bold red]Command failed with error:[/bold red]")
            if result.stderr:
                console.print(result.stderr.decode('utf-8', errors='replace'))
        
        console.print("[bold green]Command completed.[/bold green]\n")

def display_available_examples():
    """
    Display a table of available CLI examples.
    """
    if not RICH_AVAILABLE:
        return
    
    table = Table(title="Available CLI Examples")
    table.add_column("Category", style="cyan")
    table.add_column("Command", style="green")
    table.add_column("Description", style="yellow")
    
    # Help commands
    table.add_row("Help", "lego_cli.py --help", 
                 "Display main help information")
    table.add_row("Help", "lego_cli.py data-processing --help", 
                 "Display data processing help")
    
    # Detection commands
    table.add_row("Detection", "lego_cli.py detect-bricks", 
                 "Detect bricks in an image")
    table.add_row("Detection", "lego_cli.py detect-studs", 
                 "Detect studs on a brick image")
    table.add_row("Detection", "lego_cli.py infer", 
                 "Run full detection pipeline")
    
    # Training commands
    table.add_row("Training", "lego_cli.py train", 
                 "Train a detection model")
    
    # Data processing commands  
    table.add_row("Data", "lego_cli.py data-processing labelme-to-yolo", 
                 "Convert LabelMe to YOLO format")
    
    console.print(table)

def main():
    """
    Run a series of example CLI commands to demonstrate functionality.
    
    This function shows several key features:
    1. Basic CLI help and information
    2. Brick detection with visualization
    3. Stud detection and dimension classification
    4. Full inference pipeline
    5. Data processing utilities
    """
    # Display intro
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold green]LEGO[/bold green] [bold blue]Bricks[/bold blue] "
            "[bold yellow]ML[/bold yellow] [bold red]Vision[/bold red] CLI Demo",
            subtitle="Example Usage Guide"))
        
        console.print(Markdown("""
        This script demonstrates the key functionality of the LEGO Bricks ML Vision CLI.
        It will run several example commands to show how the system works.
        
        > Note: Make sure you've installed all requirements with `pip install -r requirements.txt`
        """))
        
        # Show examples table
        display_available_examples()
    else:
        print_header("LEGO Bricks ML Vision CLI Demo")
        print("This script demonstrates the key functionality of the LEGO Bricks ML Vision CLI.")
        print("It will run several example commands to show how the system works.")
        print("\n> Note: Make sure you've installed all requirements with `pip install -r requirements.txt`\n")
    
    # Test help
    run_command(["python", "lego_cli.py", "--help"], "Display main help information")
    
    # Test brick detection
    test_image = "presentation/Test_images/BricksPics/image_10.jpg"
    if os.path.exists(test_image):
        run_command([
            "python", "lego_cli.py", "detect-bricks",
            "--image", test_image,
            "--conf", "0.25",
            "--save-annotated",
            "--save-json"
        ], "Detect bricks in an image")
    
    # Test stud detection
    test_studs_image = "presentation/Test_images/StudsPics/image_10_LegoBrick_0_c87.jpg"
    if os.path.exists(test_studs_image):
        run_command([
            "python", "lego_cli.py", "detect-studs",
            "--image", test_studs_image,
            "--save-annotated"
        ], "Detect studs on a brick image")
    
    # Test full pipeline
    run_command([
        "python", "lego_cli.py", "infer",
        "--image", test_image
    ], "Run full detection pipeline")
    
    # Test data processing commands
    run_command(["python", "lego_cli.py", "data-processing", "--help"], "Display data processing help")

if __name__ == "__main__":
    main()
