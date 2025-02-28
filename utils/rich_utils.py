"""
Rich Utilities for LEGO Bricks ML Vision

This module provides enhanced console output helpers using the rich library.
It handles pretty-printing of results, progress bars, and status indicators.

Author: Miguel DiLalla
"""

import logging
from typing import List, Dict, Any, Optional, Union

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logging.warning("Rich package not available. Install with: pip install rich")

# Initialize console if rich is available
console = Console() if RICH_AVAILABLE else None

def create_status_panel(title: str, status: str = None) -> None:
    """
    Creates a status panel with optional status message.
    
    Args:
        title: Panel title
        status: Optional status message
    """
    if not RICH_AVAILABLE:
        logging.info(f"{title} - {status if status else ''}")
        return
    
    message = status if status else ""
    console.print(Panel(message, title=title, border_style="blue"))

def create_progress(**kwargs) -> Union[Progress, None]:
    """
    Creates a rich progress bar with standard columns.
    
    Returns:
        Progress object if rich is available, None otherwise
    """
    if not RICH_AVAILABLE:
        return None
        
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold green]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        **kwargs
    )

def display_results_table(results: Dict[str, Any], title: str = "Results") -> None:
    """
    Displays results in a formatted table.
    
    Args:
        results: Dictionary of results to display
        title: Table title
    """
    if not RICH_AVAILABLE:
        logging.info(f"{title}:")
        for key, value in results.items():
            logging.info(f"  {key}: {value}")
        return
    
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in results.items():
        # Format value based on type
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        elif isinstance(value, (list, tuple)) and len(value) > 5:
            formatted_value = f"[{len(value)} items]"
        else:
            formatted_value = str(value)
            
        table.add_row(key, formatted_value)
    
    console.print(table)

def setup_rich_logging():
    """
    Sets up logging with rich handler if available.
    """
    if not RICH_AVAILABLE:
        return
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)]
    )
