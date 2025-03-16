import os
import sys
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich.prompt import Prompt
import nbformat
from nbconvert import MarkdownExporter
import subprocess
from dataclasses import dataclass
from typing import List
import shutil

# Initialize rich console
console = Console()

@dataclass
class ExportError:
    """Store export error details"""
    type: str
    message: str
    details: str = ""

class NotebookExporter:
    def __init__(self):
        self.errors = []
    
    def add_error(self, error_type: str, message: str, details: str = ""):
        """Add error to collection"""
        self.errors.append(ExportError(error_type, message, details))

def show_usage():
    """Display usage information when script starts."""
    console.print("\n[bold blue]Jupyter Notebook Markdown Exporter[/bold blue]")
    console.print("\n[yellow]Description:[/yellow]")
    console.print("This tool exports Jupyter notebooks to Markdown format while preserving executed cell outputs.")
    console.print("\n[yellow]Default behavior:[/yellow]")
    console.print("- Input: notebooks/Train_Pipeline.ipynb")
    console.print("- Output: Same location as input with .md extension")
    console.print("- Images will be copied to an 'images' subdirectory")
    console.print("\n[yellow]Usage:[/yellow]")
    console.print("1. Press Enter to use defaults")
    console.print("2. Or enter a custom notebook path")
    console.print("3. The Markdown file will be created in the same location\n")

def get_default_notebook_path():
    """Get the default training pipeline notebook path."""
    return Path("notebooks") / "Train_Pipeline.ipynb"

def validate_notebook_path(notebook_path: Path) -> bool:
    """Validate that the notebook exists and has .ipynb extension."""
    return notebook_path.exists() and notebook_path.suffix == '.ipynb'

def check_dependencies():
    """Check if required dependencies are installed."""
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Checking dependencies...", total=None)
        try:
            subprocess.run(["jupyter-nbconvert", "--version"], capture_output=True, check=True)
            progress.update(task, description="[green]All dependencies found!")
            return True
        except Exception:
            progress.update(task, description="[red]Missing dependencies!")
            console.print("\n[red]Please install required package:[/red]")
            console.print("- Jupyter nbconvert (pip install nbconvert)")
            return False

def resolve_notebook_images(notebook_path: Path, nb, exporter):
    """Resolve and copy images to a local directory."""
    notebook_dir = notebook_path.parent
    repo_root = notebook_dir.parent
    
    # Create images directory next to the markdown file
    images_dir = notebook_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            lines = cell.source.split('\n')
            for i, line in enumerate(lines):
                if '![' in line and '](' in line:
                    try:
                        # Extract image path
                        img_path_start = line.find('](') + 2
                        img_path_end = line.find(')', img_path_start)
                        rel_img_path = line[img_path_start:img_path_end]
                        
                        # Resolve original image path
                        if Path(rel_img_path).is_absolute():
                            src_img_path = Path(rel_img_path)
                        elif rel_img_path.startswith('..'):
                            src_img_path = (repo_root / rel_img_path.lstrip('./')).resolve()
                        else:
                            src_img_path = (notebook_dir / rel_img_path).resolve()
                        
                        if src_img_path.exists():
                            # Copy image to images directory
                            dest_img_path = images_dir / src_img_path.name
                            shutil.copy2(src_img_path, dest_img_path)
                            
                            # Update markdown with new relative path
                            rel_path = f"images/{src_img_path.name}"
                            lines[i] = line[:img_path_start] + rel_path + line[img_path_end:]
                        else:
                            exporter.add_error("ImageNotFound", f"Image not found: {src_img_path}")
                    except Exception as e:
                        exporter.add_error("ImageProcessing", f"Failed to process image: {rel_img_path}", str(e))
            
            cell.source = '\n'.join(lines)
    
    return nb

def export_notebook_to_markdown(notebook_path: Path):
    """Export the notebook to Markdown."""
    exporter = NotebookExporter()
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Loading notebook...", total=None)
        
        try:
            # Load and process notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            progress.update(task, description="Processing images...")
            nb = resolve_notebook_images(notebook_path, nb, exporter)
            
            # Configure and run export
            progress.update(task, description="Converting to Markdown...")
            markdown_exporter = MarkdownExporter()
            markdown_data, _ = markdown_exporter.from_notebook_node(nb)
            
            # Save Markdown
            output_path = notebook_path.with_suffix('.md')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_data)
            
            progress.update(task, description="[green]Export completed!")
            
            # Show errors if any occurred
            if exporter.errors:
                console.print("\n[yellow]Warnings during export:[/yellow]")
                for err in exporter.errors:
                    console.print(f"- {err.type}: {err.message}")
                    if err.details:
                        console.print(f"  Details: {err.details}")
            
            return output_path
            
        except Exception as e:
            progress.update(task, description="[red]Export failed!")
            raise

def main():
    """Main entry point for the notebook export script."""
    show_usage()
    
    if not check_dependencies():
        sys.exit(1)
    
    default_path = get_default_notebook_path()
    notebook_path = Path(Prompt.ask("Enter notebook path", default=str(default_path)))
    
    if not validate_notebook_path(notebook_path):
        console.print(f"[red]Error: Invalid notebook path: {notebook_path}")
        sys.exit(1)
    
    try:
        output_path = export_notebook_to_markdown(notebook_path)
        console.print(f"\n[green]Successfully exported to:[/green] {output_path}")
        console.print(f"[green]Images copied to:[/green] {output_path.parent / 'images'}")
    except Exception as e:
        console.print(f"\n[red]Error during export:[/red] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
