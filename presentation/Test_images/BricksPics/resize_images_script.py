from pathlib import Path
from PIL import Image
from rich.console import Console
from rich.progress import track

# Initialize Rich console for nice output
console = Console()

def get_jpeg_files():
    """Get all JPEG files in the current directory."""
    current_dir = Path(__file__).parent
    return list(current_dir.glob("*.jpeg")) + list(current_dir.glob("*.jpg"))

def resize_image(image_path):
    """Resize image to 1/4 of its original dimensions."""
    try:
        with Image.open(image_path) as img:
            # Get current dimensions
            width, height = img.size
            # Calculate new dimensions (1/4 of original)
            new_width = width // 4
            new_height = height // 4
            
            # Resize image with high-quality downsampling
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save back to the same path, overwriting original
            resized_img.save(image_path, quality=95, optimize=True)
            return True
    except Exception as e:
        console.print(f"[red]Error processing {image_path.name}:[/red] {str(e)}")
        return False

def main():
    """Main execution function."""
    try:
        console.print("\n[bold blue]LEGO Brick Image Resizer[/bold blue]")
        
        # Get JPEG files
        jpeg_files = get_jpeg_files()
        if not jpeg_files:
            console.print("[yellow]No JPEG files found in the current directory[/yellow]")
            return
        
        # Show found files
        console.print("\n[bold]Found JPEG files:[/bold]")
        for file in jpeg_files:
            console.print(f"  • {file.name}")
        
        # Process images
        console.print("\n[bold]Resizing images to 1/4 dimensions...[/bold]")
        
        successful = 0
        for image_path in track(jpeg_files, description="Resizing images"):
            if resize_image(image_path):
                successful += 1
        
        console.print(f"\n[bold green]Resizing complete![/bold green]")
        console.print(f"Successfully resized {successful} of {len(jpeg_files)} images")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]An error occurred:[/red] {str(e)}")

if __name__ == "__main__":
    main()