from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console

RICH_AVAILABLE = True
console = Console()

def create_progress():
    """Creates and returns a Rich progress bar with custom formatting.
    
    Returns:
        Progress: Configured Rich progress bar with spinner, bar, and time tracking
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        transient=True
    )

class ProgressContext:
    """Context manager for tracking progress of operations.
    
    Automatically creates a progress bar that completes when the context exits.
    """
    def __init__(self, description: str, total: int = 1):
        self.description = description
        self.total = total
        self.progress = None
        self.task_id = None
        
    def __enter__(self):
        if RICH_AVAILABLE:
            self.progress = create_progress()
            self.progress.start()
            self.task_id = self.progress.add_task(self.description, total=self.total)
            return self.progress
        else:
            logger.info(f"Starting: {self.description}")
            return None
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.update(self.task_id, completed=self.total)
            self.progress.stop()
        else:
            logger.info(f"Completed: {self.description}")
