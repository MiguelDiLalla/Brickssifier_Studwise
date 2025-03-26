import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

# Initialize Rich console
console = Console()

def get_jpeg_files():
    """Get all JPEG files in the current directory."""
    current_dir = Path(__file__).parent
    return list(current_dir.glob("*.jpeg")) + list(current_dir.glob("*.jpg"))

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    output_dir = Path(__file__).parent / "cropped"
    output_dir.mkdir(exist_ok=True)
    return output_dir

def clean_metadata(current_dir):
    """Clean metadata from images using the CLI tool."""
    def find_cli_recursively(start_path: Path, max_depth: int = 5) -> Path:
        """Recursively search for lego_cli.py in parent directories."""
        current = start_path
        for _ in range(max_depth):
            cli_path = current / "lego_cli.py"
            if cli_path.exists():
                console.print(f"[green]Found lego_cli.py at:[/green] {cli_path}")
                return cli_path
            if current.parent == current:  # Reached root directory
                break
            current = current.parent
        
        raise FileNotFoundError(
            f"Could not find lego_cli.py in parent directories (searched {max_depth} levels up)"
        )

    try:
        cli_path = find_cli_recursively(Path(__file__).parent)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        return False
    
    console.print("\n[bold yellow]Running metadata cleanup...[/bold yellow]")
    
    cmd = [
        "python",
        str(cli_path),
        "metadata",
        "clean-batch",
        str(current_dir),
        "--force"  # Skip confirmation prompt
    ]
    
    try:
        process = subprocess.run(
            cmd,
            check=True,
            text=True
        )
        console.print("[green]Metadata cleanup completed successfully[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print("[red]Error during metadata cleanup[/red]")
        console.print(f"Error code: {e.returncode}")
        return False
    except Exception as e:
        console.print(f"[red]Failed to clean metadata:[/red] {str(e)}")
        return False

def process_images(jpeg_files):
    """Process each image using the LEGO CLI tool."""
    cli_path = Path(__file__).parent.parent.parent.parent / "lego_cli.py"
    output_dir = ensure_output_dir()
    
    for image_path in track(jpeg_files, description="Processing images"):
        cmd = [
            "python",
            str(cli_path),
            "detect-bricks",
            "--image", str(image_path),
            "--output", str(output_dir),
            "--save-cropped",
            "--conf", "0.40",
            "--no-save-annotated",
            "--no-save-json"
        ]
        
        try:
            # Run subprocess without capturing output to see CLI logs
            process = subprocess.run(
                cmd,
                check=True,
                text=True
            )
            console.print(f"[green]Successfully processed[/green]: {image_path.name}")
            
            # Clean up empty subfolder
            image_name = image_path.stem
            empty_subfolder = output_dir / image_name
            if empty_subfolder.exists() and empty_subfolder.is_dir():
                try:
                    empty_subfolder.rmdir()  # This will only remove the directory if it's empty
                    console.print(f"[dim]Cleaned up empty subfolder for[/dim]: {image_name}")
                except OSError:
                    # If the folder is not empty, we leave it as is
                    pass
                    
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error processing[/red]: {image_path.name}")
            console.print(f"Error code: {e.returncode}")
        except Exception as e:
            console.print(f"[red]Failed to process[/red]: {image_path.name}")
            console.print(f"Exception: {str(e)}")

def main():
    """Main execution function."""
    try:
        console.print(Panel.fit("LEGO Brick Detection - Batch Processing", style="bold blue"))
        
        # Get JPEG files
        jpeg_files = get_jpeg_files()
        if not jpeg_files:
            console.print("[yellow]No JPEG files found in the current directory[/yellow]")
            return
        
        # Show found files
        console.print("\n[bold]Found JPEG files:[/bold]")
        for file in jpeg_files:
            console.print(f"  • {file.name}")
        
        # Create output directory
        output_dir = ensure_output_dir()
        console.print(f"\n[bold]Output directory:[/bold] {output_dir}")
        
        # Clean metadata first
        if not clean_metadata(Path(__file__).parent):
            console.print("[yellow]Warning: Metadata cleanup failed, but continuing with processing...[/yellow]")
        
        # Process images
        console.print("\n[bold]Starting processing...[/bold]")
        process_images(jpeg_files)
        
        console.print("\n[bold green]Processing complete![/bold green]")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]An error occurred:[/red] {str(e)}")

if __name__ == "__main__":
    main()