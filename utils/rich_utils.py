"""
Rich Integration Utilities for LEGO Bricks ML Vision

This module provides consistent rich console output components for the project,
making the CLI experience more intuitive and visually appealing.

Key features:
- Progress bars with consistent styling
- Result tables for model outputs
- Status panels and spinners
- Error formatting and display
"""

import sys
from typing import List, Dict, Any, Optional, Union, Callable

# Try importing rich
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, 
        BarColumn, TimeElapsedColumn, TaskProgressColumn
    )
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.traceback import install as install_rich_traceback
    
    # Install rich traceback handler for better error display
    install_rich_traceback(show_locals=False, width=100)
    
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' package for enhanced output: pip install rich")

def create_progress() -> Union[Progress, None]:
    """
    Creates a consistently styled progress bar for use across the project.
    
    Returns:
        Progress or None: A rich Progress object if rich is available, None otherwise
    """
    if not RICH_AVAILABLE:
        return None
        
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn()
    )

def create_status_panel(title: str, message: str = "") -> None:
    """
    Displays a status panel with the given title and message.
    
    Args:
        title (str): The panel title
        message (str, optional): The message to display inside the panel
    """
    if not RICH_AVAILABLE:
        print(f"\n{title}")
        if message:
            print(message)
        return
        
    console.print(Panel(
        message,
        title=f"[bold blue]{title}[/bold blue]",
        border_style="blue"
    ))

def display_results_table(title: str, data: Dict[str, Any]) -> None:
    """
    Displays a formatted table with the analysis results.
    
    Args:
        title (str): The table title
        data (dict): Dictionary of result data to display (key-value pairs)
    """
    if not RICH_AVAILABLE:
        print(f"\n{title}")
        for key, value in data.items():
            print(f"{key}: {value}")
        return
        
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in data.items():
        # Format different value types appropriately
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        elif isinstance(value, list) and len(value) > 0:
            formatted_value = ", ".join(str(item) for item in value[:3])
            if len(value) > 3:
                formatted_value += f"... (+{len(value)-3} more)"
        else:
            formatted_value = str(value)
            
        table.add_row(key, formatted_value)
    
    console.print(table)

def display_error(message: str, exception: Optional[Exception] = None) -> None:
    """
    Displays an error message with optional exception details.
    
    Args:
        message (str): The error message
        exception (Exception, optional): The exception object
    """
    if not RICH_AVAILABLE:
        print(f"ERROR: {message}")
        if exception:
            print(f"Exception: {str(exception)}")
        return
        
    console.print(f"[bold red]ERROR:[/bold red] {message}")
    if exception:
        console.print_exception()
