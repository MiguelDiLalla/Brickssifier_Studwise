import os
import sys
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich.prompt import Prompt
import nbformat
from nbconvert import PDFExporter
import subprocess
import logging
from rich.logging import RichHandler

# Initialize rich console and logging
console = Console()
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("notebook_export")

def show_usage():
    """Display usage information when script starts."""
    console.print("\n[bold blue]Jupyter Notebook PDF Exporter[/bold blue]")
    console.print("\n[yellow]Description:[/yellow]")
    console.print("This tool exports Jupyter notebooks to PDF format while preserving executed cell outputs.")
    console.print("\n[yellow]Default behavior:[/yellow]")
    console.print("- Input: notebooks/Train_Pipeline.ipynb")
    console.print("- Output: Same location as input with .pdf extension")
    console.print("\n[yellow]Usage:[/yellow]")
    console.print("1. Press Enter to use defaults")
    console.print("2. Or enter a custom notebook path")
    console.print("3. The PDF will be created in the same location\n")

def get_default_notebook_path():
    """Get the default training pipeline notebook path."""
    return Path("notebooks") / "Train_Pipeline.ipynb"

def validate_notebook_path(notebook_path: Path) -> bool:
    """Validate that the notebook exists and has .ipynb extension."""
    return notebook_path.exists() and notebook_path.suffix == '.ipynb'

def check_dependencies():
    """Check if required dependencies are installed with detailed logging."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Checking dependencies...", total=None)
        
        try:
            # Check for nbconvert
            log.debug("Checking nbconvert installation...")
            nbconvert_result = subprocess.run(
                ["jupyter-nbconvert", "--version"], 
                capture_output=True, 
                text=True,
                check=True
            )
            log.debug(f"nbconvert version: {nbconvert_result.stdout.strip()}")
            
            # Check for MiKTeX installation with detailed output
            log.debug("Checking MiKTeX installation...")
            try:
                miktex_result = subprocess.run(
                    ["miktex", "--version"], 
                    capture_output=True, 
                    text=True,
                    shell=True
                )
                log.debug(f"MiKTeX check output:\n{miktex_result.stdout}")
                if miktex_result.stderr:
                    log.warning(f"MiKTeX stderr: {miktex_result.stderr}")
            except Exception as e:
                log.error(f"MiKTeX check failed: {e}")
                log.debug("Trying alternative MiKTeX paths...")
                # Try common MiKTeX installation paths
                miktex_paths = [
                    r"C:\Program Files\MiKTeX\miktex\bin\x64",
                    r"C:\Program Files (x86)\MiKTeX\miktex\bin",
                ]
                found = False
                for path in miktex_paths:
                    if Path(path).exists():
                        log.debug(f"Found MiKTeX path: {path}")
                        found = True
                if not found:
                    log.error("No MiKTeX installation found in common paths")
            
            # Check XeLaTeX with detailed output
            log.debug("Checking XeLaTeX installation...")
            latex_result = subprocess.run(
                ["xelatex", "--version"], 
                capture_output=True, 
                text=True,
                check=True
            )
            log.debug(f"XeLaTeX version info:\n{latex_result.stdout}")
            if latex_result.stderr:
                log.warning(f"XeLaTeX stderr: {latex_result.stderr}")
            
            progress.update(task, description="[green]All dependencies found!")
            return True
            
        except FileNotFoundError as e:
            log.error(f"Dependency not found: {e}")
            progress.update(task, description="[red]Missing dependencies!")
            console.print("\n[red]Please check your installation:")
            if "miktex" in str(e):
                console.print("1. MiKTeX is not properly installed or not in PATH")
                console.print("   - Download from: https://miktex.org/download")
                console.print("   - Verify installation in Control Panel")
                console.print("   - Add to PATH: C:\\Program Files\\MiKTeX\\miktex\\bin\\x64")
            elif "xelatex" in str(e):
                console.print("1. XeLaTeX not found. Please:")
                console.print("   - Open MiKTeX Console")
                console.print("   - Go to Packages")
                console.print("   - Search and install 'xetex' package")
            return False
        except subprocess.CalledProcessError as e:
            log.error(f"Dependency check failed with error code {e.returncode}")
            log.error(f"Command output: {e.output if hasattr(e, 'output') else 'No output'}")
            log.error(f"Error output: {e.stderr if hasattr(e, 'stderr') else 'No stderr'}")
            progress.update(task, description="[red]Dependency check failed!")
            return False

def resolve_notebook_images(notebook_path: Path, nb):
    """Resolve relative image paths in notebook to absolute paths."""
    log.debug("Resolving image paths in notebook...")
    notebook_dir = notebook_path.parent
    
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            # Find markdown image references
            lines = cell.source.split('\n')
            for i, line in enumerate(lines):
                if '![' in line and '](' in line:
                    # Extract image path
                    img_path_start = line.find('](') + 2
                    img_path_end = line.find(')', img_path_start)
                    rel_img_path = line[img_path_start:img_path_end]
                    
                    if rel_img_path.startswith('..'):
                        # Convert relative path to absolute
                        abs_img_path = notebook_dir.joinpath(rel_img_path).resolve()
                        log.debug(f"Converting image path: {rel_img_path} -> {abs_img_path}")
                        
                        if abs_img_path.exists():
                            # Update the path in the notebook
                            lines[i] = line[:img_path_start] + str(abs_img_path) + line[img_path_end:]
                        else:
                            log.error(f"Image not found: {abs_img_path}")
            
            cell.source = '\n'.join(lines)
    return nb

def export_notebook_to_pdf(notebook_path: Path):
    """Export the notebook to PDF with execution output and debug logging."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Loading notebook...", total=None)
        
        try:
            # Load and configure exporter
            log.debug(f"Loading notebook from: {notebook_path}")
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Resolve image paths before export
            nb = resolve_notebook_images(notebook_path, nb)
            
            progress.update(task, description="Configuring PDF export...")
            pdf_exporter = PDFExporter()
            log.debug("PDF Exporter configuration:")
            log.debug(f"Template: {pdf_exporter.template_name}")
            log.debug(f"LaTeX command: {pdf_exporter.latex_command}")
            
            # Export to PDF
            progress.update(task, description="Converting to PDF...")
            try:
                pdf_data, resources = pdf_exporter.from_notebook_node(nb)
            except Exception as e:
                log.error(f"PDF export failed: {str(e)}")
                log.debug("Checking LaTeX environment:")
                latex_log_path = Path(r"C:\Users\User\AppData\Local\MiKTeX\miktex\log\xelatex.log")
                if latex_log_path.exists():
                    log.debug("LaTeX log content:")
                    log.debug(latex_log_path.read_text(errors='ignore'))
                raise
            
            # Save PDF
            output_path = notebook_path.with_suffix('.pdf')
            with open(output_path, 'wb') as f:
                f.write(pdf_data)
            
            progress.update(task, description="[green]Export completed!")
            return output_path
            
        except Exception as e:
            log.exception("Export failed with exception:")
            raise

def main():
    """Main entry point for the notebook export script."""
    show_usage()
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Get notebook path from user or use default
    default_path = get_default_notebook_path()
    
    notebook_path = Path(Prompt.ask(
        "Enter notebook path",
        default=str(default_path)
    ))
    
    # Validate notebook path
    if not validate_notebook_path(notebook_path):
        console.print(f"[red]Error: Invalid notebook path: {notebook_path}")
        sys.exit(1)
    
    try:
        # Export notebook to PDF
        output_path = export_notebook_to_pdf(notebook_path)
        console.print(f"\n[green]Successfully exported to:[/green] {output_path}")
    except Exception as e:
        console.print(f"\n[red]Error during export:[/red] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
