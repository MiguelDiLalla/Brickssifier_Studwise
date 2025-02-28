#!/usr/bin/env python
"""
LEGO Bricks ML Vision - Command Line Interface

This module provides a comprehensive CLI for the LEGO Bricks ML Vision project,
integrating detection, training, and data processing capabilities.

Usage:
    lego_cli.py [OPTIONS] COMMAND [ARGS]...

Author: Miguel DiLalla
"""

import os
import sys
import logging
import click
import time
import glob
import json
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.progress import Progress, track, SpinnerColumn, TextColumn
from rich.status import Status
from rich.text import Text
from rich.style import Style
from rich.layout import Layout
from rich.progress import *


# Try importing rich for enhanced console output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' package for enhanced output: pip install rich")

# Import project modules
from utils.model_utils import (
    detect_bricks, detect_studs, run_full_algorithm, read_exif, 
    write_exif, clean_exif_metadata, setup_utils
)
import utils.data_utils as data_utils

# Version information
__version__ = "1.0.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "cli.log"), mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Initialize configuration
config = setup_utils(repo_download=False)

def print_version(ctx, param, value):
    """Print version information and exit."""
    if not value or ctx.resilient_parsing:
        return
    if RICH_AVAILABLE:
        console.print(f"[bold green]LEGO Bricks ML Vision[/bold green] version [yellow]{__version__}[/yellow]")
    else:
        click.echo(f"LEGO Bricks ML Vision version {__version__}")
    ctx.exit()

def validate_output_dir(ctx, param, value):
    """Validate and create output directory if it doesn't exist."""
    if value:
        os.makedirs(value, exist_ok=True)
    return value

#
# CLI Groups and Commands
#

@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug output.')
@click.option('--version', is_flag=True, callback=print_version, expose_value=False, is_eager=True,
              help='Show version information and exit.')
def cli(debug):
    """LEGO Bricks ML Vision - Command Line Interface

    This tool provides commands for detecting LEGO bricks, classifying their dimensions,
    training models, and processing datasets.
    """
    # Set debug level if requested
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

@cli.command('detect-bricks')
@click.option('--image', required=True, type=click.Path(exists=True),
              help='Path to input image or directory of images.')
@click.option('--output', type=click.Path(), callback=validate_output_dir,
              help='Directory to save output. Will be created if it does not exist.')
@click.option('--conf', type=float, default=0.25,
              help='Confidence threshold for detections (0-1).')
@click.option('--save-annotated/--no-save-annotated', default=True,
              help='Save annotated images with detection visualization.')
@click.option('--save-json/--no-save-json', default=False, 
              help='Save detection results as JSON files.')
@click.option('--clean-exif/--no-clean-exif', default=False,
              help='Clean EXIF metadata before processing.')
def detect_bricks_cmd(image, output, conf, save_annotated, save_json, clean_exif):
    """Detect LEGO bricks in images using the trained model.
    
    This command runs the brick detection model on the specified image(s)
    and saves the results.
    """
    if RICH_AVAILABLE:
        console.print(Panel.fit("[bold blue]LEGO Brick Detection[/bold blue]"))
    
    # Ensure output directory has a default value
    if not output:
        output = os.path.join(os.getcwd(), "results", "brick_detection")
        os.makedirs(output, exist_ok=True)
        logger.info(f"Using default output directory: {output}")
    
    # Handle single image or directory input
    images_to_process = []
    if os.path.isdir(image):
        images_to_process = glob.glob(os.path.join(image, "*.jpg")) + \
                           glob.glob(os.path.join(image, "*.png"))
        logger.info(f"Found {len(images_to_process)} images in directory {image}")
    else:
        images_to_process = [image]
        
    if not images_to_process:
        logger.error("No images found to process.")
        sys.exit(1)
        
    for img_path in images_to_process:
        logger.info(f"Processing image: {img_path}")
        
        # Clean EXIF if requested
        if clean_exif:
            logger.info(f"Cleaning EXIF metadata for {img_path}")
            clean_exif_metadata(img_path)
        
        # Create image-specific output subfolder
        img_name = os.path.basename(img_path)
        img_output = os.path.join(output, os.path.splitext(img_name)[0])
        os.makedirs(img_output, exist_ok=True)
        
        # Run detection
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[bold green]{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"[green]Detecting bricks in {img_name}...", total=100)
                
                # Run detection with progress updates
                progress.update(task, advance=10)
                result = detect_bricks(
                    img_path,
                    conf=conf,
                    save_json=save_json, 
                    save_annotated=save_annotated,
                    output_folder=img_output
                )
                progress.update(task, advance=90)
                
        else:
            # Run detection without progress display
            result = detect_bricks(
                img_path,
                conf=conf,
                save_json=save_json, 
                save_annotated=save_annotated,
                output_folder=img_output
            )
        
        # Check detection results
        if result is None:
            logger.error(f"Detection failed for {img_path}")
            continue
            
        boxes = result.get("boxes", [])
        if len(boxes) == 0:
            logger.warning(f"No bricks detected in {img_path}")
        else:
            logger.info(f"Detected {len(boxes)} bricks in {img_path}")
            
        # Save annotated image if requested
        if save_annotated:
            annotated_path = os.path.join(img_output, "annotated_image.jpg")
            logger.info(f"Saved annotated image to {annotated_path}")
            
        # Print path to results
        if RICH_AVAILABLE:
            console.print(f"[green]Results saved to:[/green] {img_output}")
        else:
            click.echo(f"Results saved to: {img_output}")

@cli.command('detect-studs')
@click.option('--image', required=True, type=click.Path(exists=True),
              help='Path to input image or directory of images.')
@click.option('--output', type=click.Path(), callback=validate_output_dir,
              help='Directory to save output. Will be created if it does not exist.')
@click.option('--conf', type=float, default=0.25,
              help='Confidence threshold for detections (0-1).')
@click.option('--save-annotated/--no-save-annotated', default=True,
              help='Save annotated images with detection visualization.')
@click.option('--clean-exif/--no-clean-exif', default=False,
              help='Clean EXIF metadata before processing.')
def detect_studs_cmd(image, output, conf, save_annotated, clean_exif):
    """Detect studs on LEGO bricks and classify dimensions.
    
    This command runs the stud detection model on the specified brick image(s),
    counts the studs, and classifies the brick dimensions.
    """
    if RICH_AVAILABLE:
        console.print(Panel.fit("[bold blue]LEGO Stud Detection & Classification[/bold blue]"))
    
    # Ensure output directory has a default value
    if not output:
        output = os.path.join(os.getcwd(), "results", "stud_detection")
        os.makedirs(output, exist_ok=True)
        logger.info(f"Using default output directory: {output}")
    
    # Handle single image or directory input
    images_to_process = []
    if os.path.isdir(image):
        images_to_process = glob.glob(os.path.join(image, "*.jpg")) + \
                           glob.glob(os.path.join(image, "*.png"))
        logger.info(f"Found {len(images_to_process)} images in directory {image}")
    else:
        images_to_process = [image]
        
    if not images_to_process:
        logger.error("No images found to process.")
        sys.exit(1)
        
    for img_path in images_to_process:
        logger.info(f"Processing image: {img_path}")
        
        # Clean EXIF if requested
        if clean_exif:
            logger.info(f"Cleaning EXIF metadata for {img_path}")
            clean_exif_metadata(img_path)
        
        # Create image-specific output subfolder
        img_name = os.path.basename(img_path)
        img_output = os.path.join(output, os.path.splitext(img_name)[0])
        os.makedirs(img_output, exist_ok=True)
        
        # Run detection
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[bold green]{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"[green]Detecting studs in {img_name}...", total=100)
                
                # Run detection with progress updates
                progress.update(task, advance=10)
                result = detect_studs(
                    img_path,
                    conf=conf,
                    save_annotated=save_annotated,
                    output_folder=img_output
                )
                progress.update(task, advance=90)
                
        else:
            # Run detection without progress display
            result = detect_studs(
                img_path,
                conf=conf,
                save_annotated=save_annotated,
                output_folder=img_output
            )
        
        # Check detection results
        if result is None:
            logger.error(f"Detection failed for {img_path}")
            continue
            
        # Display dimension classification results
        dimension = result.get("dimension", "Unknown")
        if RICH_AVAILABLE:
            console.print(f"[green]Brick Dimension:[/green] [bold yellow]{dimension}[/bold yellow]")
            console.print(f"[green]Results saved to:[/green] {img_output}")
        else:
            click.echo(f"Brick Dimension: {dimension}")
            click.echo(f"Results saved to: {img_output}")

@cli.command('infer')
@click.option('--image', required=True, type=click.Path(exists=True),
              help='Path to input image or directory of images.')
@click.option('--output', type=click.Path(), callback=validate_output_dir,
              help='Directory to save output. Will be created if it does not exist.')
@click.option('--save-annotated/--no-save-annotated', default=True,
              help='Save annotated images with detection visualization.')
@click.option('--force-run/--no-force-run', default=False,
              help='Force re-running detection even if cached results exist.')
def infer_cmd(image, output, save_annotated, force_run):
    """Run the full inference pipeline: brick detection, stud detection, and classification.
    
    This command runs the complete analysis pipeline on the input image(s).
    """
    if RICH_AVAILABLE:
        console.print(Panel.fit("[bold blue]LEGO Full Inference Pipeline[/bold blue]"))
    
    # Ensure output directory has a default value
    if not output:
        output = os.path.join(os.getcwd(), "results", "full_inference")
        os.makedirs(output, exist_ok=True)
        logger.info(f"Using default output directory: {output}")
    
    # Handle single image or directory input
    images_to_process = []
    if os.path.isdir(image):
        images_to_process = glob.glob(os.path.join(image, "*.jpg")) + \
                           glob.glob(os.path.join(image, "*.png"))
        logger.info(f"Found {len(images_to_process)} images in directory {image}")
    else:
        images_to_process = [image]
        
    if not images_to_process:
        logger.error("No images found to process.")
        sys.exit(1)
        
    for img_path in images_to_process:
        logger.info(f"Processing image: {img_path}")
        
        # Create image-specific output subfolder
        img_name = os.path.basename(img_path)
        img_output = os.path.join(output, os.path.splitext(img_name)[0])
        os.makedirs(img_output, exist_ok=True)
        
        # Run full algorithm
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[bold green]{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"[green]Running full inference on {img_name}...", total=100)
                
                # First phase: brick detection
                progress.update(task, description=f"[green]Phase 1: Detecting bricks in {img_name}...", advance=10)
                
                # Second phase: stud detection and classification
                progress.update(task, description=f"[green]Phase 2: Analyzing studs...", advance=30)
                
                result = run_full_algorithm(
                    img_path,
                    save_annotated=save_annotated,
                    output_folder=img_output,
                    force_ran=force_run
                )
                
                progress.update(task, advance=60)
                
        else:
            # Run full algorithm without progress display
            result = run_full_algorithm(
                img_path,
                save_annotated=save_annotated,
                output_folder=img_output,
                force_ran=force_run
            )
        
        # Check results
        if result is None:
            logger.error(f"Inference failed for {img_path}")
            continue
            
        # Count detected bricks and get dimension info
        brick_results = result.get("brick_results", {})
        studs_results = result.get("studs_results", [])
        
        boxes = brick_results.get("boxes", [])
        num_bricks = len(boxes)
        
        # Format and display results
        if RICH_AVAILABLE:
            table = Table(title=f"Analysis Results for {img_name}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Bricks detected", str(num_bricks))
            
            for i, stud_result in enumerate(studs_results):
                dimension = stud_result.get("dimension", "Unknown")
                table.add_row(f"Brick {i+1} dimension", dimension)
            
            console.print(table)
            console.print(f"[green]Results saved to:[/green] {img_output}")
        else:
            click.echo(f"Bricks detected: {num_bricks}")
            for i, stud_result in enumerate(studs_results):
                dimension = stud_result.get("dimension", "Unknown")
                click.echo(f"Brick {i+1} dimension: {dimension}")
            click.echo(f"Results saved to: {img_output}")

@cli.command('train')
@click.option('--mode', type=click.Choice(['bricks', 'studs']), required=True,
              help='Training mode: "bricks" or "studs".')
@click.option('--epochs', type=int, default=20,
              help='Number of training epochs.')
@click.option('--batch-size', type=int, default=16,
              help='Batch size for training.')
@click.option('--show-results/--no-show-results', default=True,
              help='Display results after training.')
@click.option('--cleanup/--no-cleanup', default=True,
              help='Remove cached datasets and logs after training.')
@click.option('--force-extract', is_flag=True,
              help='Force re-extraction of dataset.')
@click.option('--use-pretrained', is_flag=True,
              help='Use LEGO-trained model instead of YOLOv8n.')
def train_cmd(mode, epochs, batch_size, show_results, cleanup, force_extract, use_pretrained):
    """Train a detection model for either bricks or studs.
    
    This command runs the full training pipeline, including dataset preparation,
    model training, and results export.
    """
    if RICH_AVAILABLE:
        console.print(Panel.fit(f"[bold blue]Training {mode.capitalize()} Detection Model[/bold blue]"))
        
    # Import train module functions dynamically to avoid circular imports
    from train import (
        setup_logging, detect_hardware, setup_execution_structure,
        unzip_dataset, validate_dataset, split_dataset, 
        augment_data, select_model, train_model,
        export_logs, zip_and_download_results, display_last_training_session
    )
    
    # Set up training
    setup_logging()
    device = detect_hardware()
    logger.info(f"Using device: {device}")
    
    # Run training pipeline
    setup_execution_structure()
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold green]{task.fields[percentage]:.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            # Dataset preparation phase
            prep_task = progress.add_task(
                "[green]Preparing dataset...", 
                total=100, 
                percentage=0
            )
            
            # Extract dataset
            progress.update(prep_task, advance=10, percentage=10)
            dataset_path = unzip_dataset(mode, force_extract)
            
            # Validate dataset
            progress.update(prep_task, advance=10, percentage=20)
            validate_dataset(mode)
            
            # Split dataset
            progress.update(prep_task, advance=20, percentage=40)
            dataset_yolo_path = split_dataset(mode)
            
            # Augment data
            progress.update(prep_task, advance=20, percentage=60)
            augment_data(dataset_yolo_path)
            
            # Select model
            progress.update(prep_task, advance=10, percentage=70)
            model_path = select_model(mode, use_pretrained)
            
            # Complete preparation
            progress.update(prep_task, completed=100, percentage=100)
            
            # Training phase
            train_task = progress.add_task(
                f"[green]Training {mode} detection model...", 
                total=100, 
                percentage=0
            )
            
            # Train model with external progress tracking
            session_results = train_model(dataset_yolo_path, model_path, device, epochs, batch_size)
            
            # Complete training
            progress.update(train_task, completed=100, percentage=100)
            
            # Results phase
            results_task = progress.add_task(
                "[green]Processing results...", 
                total=100, 
                percentage=0
            )
            
            # Export logs
            progress.update(results_task, advance=30, percentage=30)
            export_logs()
            
            # Zip and prepare download
            progress.update(results_task, advance=40, percentage=70)
            zip_and_download_results()
            
            # Complete results
            progress.update(results_task, completed=100, percentage=100)
    else:
        # Run without rich progress display
        logger.info("Preparing dataset...")
        dataset_path = unzip_dataset(mode, force_extract)
        validate_dataset(mode)
        dataset_yolo_path = split_dataset(mode)
        augment_data(dataset_yolo_path)
        model_path = select_model(mode, use_pretrained)
        
        logger.info(f"Training {mode} detection model...")
        session_results = train_model(dataset_yolo_path, model_path, device, epochs, batch_size)
        
        logger.info("Processing results...")
        export_logs()
        zip_and_download_results()
    
    # Display training results if requested
    if show_results:
        if RICH_AVAILABLE:
            console.print("\n[bold green]Training Results:[/bold green]")
        else:
            click.echo("\nTraining Results:")
            
        display_last_training_session(session_results)
    
    # Clean up if requested
    if cleanup:
        from train import cleanup_after_training
        if RICH_AVAILABLE:
            console.print("\n[yellow]Cleaning up temporary files...[/yellow]")
        else:
            click.echo("\nCleaning up temporary files...")
        cleanup_after_training()
    
    if RICH_AVAILABLE:
        console.print(Panel.fit("[bold green]Training completed successfully![/bold green]"))
    else:
        click.echo("Training completed successfully!")

@cli.group('data-processing')
def data_processing():
    """Commands for dataset processing and visualization."""
    pass

@data_processing.command('labelme-to-yolo')
@click.option('--input', required=True, type=click.Path(exists=True),
              help='Input folder containing LabelMe JSON files.')
def labelme_to_yolo_cmd(input):
    """Convert LabelMe JSON annotations to YOLO format."""
    # Create args object to match data_utils expected interface
    class Args:
        def __init__(self, input):
            self.input = input
    
    args = Args(input)
    
    if RICH_AVAILABLE:
        console.print(Panel.fit("[bold blue]Converting LabelMe to YOLO Format[/bold blue]"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[green]Converting annotations...", total=100)
            progress.update(task, advance=10)
            
            data_utils.convert_labelme_to_yolo(args)
            
            progress.update(task, completed=100)
            
        console.print("[bold green]Conversion completed![/bold green]")
        
    else:
        logger.info("Converting LabelMe to YOLO format...")
        data_utils.convert_labelme_to_yolo(args)
        logger.info("Conversion completed!")
        
@data_processing.command('keypoints-to-bboxes')
@click.option('--input', required=True, type=click.Path(exists=True),
              help='Input folder containing keypoints JSON files.')
@click.option('--area-ratio', type=float, default=0.4,
              help='Total area ratio for bounding boxes.')
def keypoints_to_bboxes_cmd(input, area_ratio):
    """Convert keypoints to bounding boxes."""
    # Create args object to match data_utils expected interface
    class Args:
        def __init__(self, input, area_ratio):
            self.input = input
            self.area_ratio = area_ratio
    
    args = Args(input, area_ratio)
    
    if RICH_AVAILABLE:
        console.print(Panel.fit("[bold blue]Converting Keypoints to Bounding Boxes[/bold blue]"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[green]Converting keypoints...", total=100)
            progress.update(task, advance=10)
            
            data_utils.convert_keypoints_to_bboxes(args)
            
            progress.update(task, completed=100)
            
        console.print("[bold green]Conversion completed![/bold green]")
        
    else:
        logger.info("Converting keypoints to bounding boxes...")
        data_utils.convert_keypoints_to_bboxes(args)
        logger.info("Conversion completed!")

@data_processing.command('visualize')
@click.option('--input', required=True, type=click.Path(exists=True),
              help='Path to a single image or folder of images.')
@click.option('--labels', required=True, type=click.Path(exists=True),
              help='Folder containing YOLO .txt labels.')
@click.option('--grid-size', default="3x4",
              help='Grid size for visualization (e.g., 3x4).')
def visualize_cmd(input, labels, grid_size):
    """Visualize YOLO annotations on images."""
    # Create args object to match data_utils expected interface
    class Args:
        def __init__(self, input, labels, grid_size):
            self.input = input
            self.labels = labels
            self.grid_size = grid_size
    
    args = Args(input, labels, grid_size)
    
    if RICH_AVAILABLE:
        console.print(Panel.fit("[bold blue]Visualizing YOLO Annotations[/bold blue]"))
    
    logger.info("Visualizing YOLO annotations...")
    data_utils.visualize_yolo_annotation(args)
    logger.info("Visualization completed!")

@cli.command('cleanup')
@click.option('--all', 'all_files', is_flag=True, help='Clean all temporary files, including results.')
@click.option('--logs-only', is_flag=True, help='Clean only log files.')
@click.option('--cache-only', is_flag=True, help='Clean only cache files.')
@click.option('--results-only', is_flag=True, help='Clean only results files.')
def cleanup_cmd(all_files, logs_only, cache_only, results_only):
    """Clean up temporary files and directories."""
    # Import cleanup function from train module
    from train import cleanup_after_training
    
    # Define folders to clean
    to_clean = []
    
    if all_files or logs_only:
        to_clean.append("logs")
    if all_files or cache_only:
        to_clean.append("cache")
    if all_files or results_only:
        to_clean.append("results")
        
    if not to_clean:
        # Default behavior
        to_clean = ["cache", "logs"]
    
    if RICH_AVAILABLE:
        console.print(Panel.fit("[bold yellow]Cleaning Up Temporary Files[/bold yellow]"))
        
        # Display what will be cleaned
        table = Table(title="Folders to Clean")
        table.add_column("Folder", style="cyan")
        table.add_column("Status", style="yellow")
        
        for folder in to_clean:
            folder_path = os.path.join(os.getcwd(), folder)
            if os.path.exists(folder_path):
                table.add_row(folder, "Will be removed")
            else:
                table.add_row(folder, "Not found")
                
        console.print(table)
        
        # Confirm cleanup
        if click.confirm("Do you want to proceed with cleanup?", default=True):
            with console.status("[bold green]Cleaning up..."):
                for folder in to_clean:
                    folder_path = os.path.join(os.getcwd(), folder)
                    if os.path.exists(folder_path):
                        import shutil
                        shutil.rmtree(folder_path)
                        console.print(f"[green]✓[/green] Removed {folder}")
                    else:
                        console.print(f"[yellow]⚠[/yellow] {folder} not found")
            console.print("[bold green]Cleanup completed![/bold green]")
        else:
            console.print("[yellow]Cleanup cancelled.[/yellow]")
    else:
        # Non-rich version
        for folder in to_clean:
            folder_path = os.path.join(os.getcwd(), folder)
            if os.path.exists(folder_path):
                click.echo(f"Will remove: {folder}")
            else:
                click.echo(f"Not found: {folder}")
                
        if click.confirm("Do you want to proceed with cleanup?", default=True):
            for folder in to_clean:
                folder_path = os.path.join(os.getcwd(), folder)
                if os.path.exists(folder_path):
                    import shutil
                    shutil.rmtree(folder_path)
                    click.echo(f"Removed {folder}")
                else:
                    click.echo(f"{folder} not found")
            click.echo("Cleanup completed!")
        else:
            click.echo("Cleanup cancelled.")

@cli.command('version')
def version_cmd():
    """Display version and system information."""
    if RICH_AVAILABLE:
        table = Table(title="LEGO Bricks ML Vision System Information")
        table.add_column("Component", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Version", __version__)
        table.add_row("Python", sys.version.split()[0])
        
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            table.add_row("CUDA", cuda_version)
            table.add_row("GPU", device_name)
        else:
            table.add_row("CUDA", "Not available")
            table.add_row("GPU", "Not available")
            
        import platform
        table.add_row("OS", platform.platform())
        
        console.print(table)
    else:
        click.echo(f"LEGO Bricks ML Vision v{__version__}")
        click.echo(f"Python: {sys.version.split()[0]}")
        
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            click.echo(f"CUDA: {cuda_version}")
            click.echo(f"GPU: {device_name}")
        else:
            click.echo("CUDA: Not available")
            click.echo("GPU: Not available")
            
        import platform
        click.echo(f"OS: {platform.platform()}")


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Execute the CLI
    cli()