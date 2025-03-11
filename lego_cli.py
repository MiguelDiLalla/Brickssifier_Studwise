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
import random
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
from rich.box import ROUNDED


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

# Import project modules - Updated to use new modular structure
from utils.config_utils import setup_utils, config
from utils.detection_utils import detect_bricks, detect_studs
from utils.pipeline_utils import run_full_algorithm, batch_process
from utils.exif_utils import read_exif, write_exif, clean_exif_metadata
from utils.visualization_utils import display_conversion_summary
import utils.data_utils as data_utils
from utils.batch_utils import process_batch_inference, display_batch_results
from utils.visualization_utils import create_batch_visualization

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

def display_rich_help():
    """Display a rich formatted help showcase of all commands and options."""
    if not RICH_AVAILABLE:
        return False
        
    console.print(Panel.fit(
        "[bold blue]LEGO Bricks ML Vision CLI[/bold blue]\n" +
        "[dim]A comprehensive tool for LEGO brick detection and analysis[/dim]",
        border_style="blue"
    ))
    
    # Main commands table
    commands_table = Table(
        title="Available Commands",
        show_header=True,
        header_style="bold cyan",
        box=ROUNDED
    )
    commands_table.add_column("Command", style="green")
    commands_table.add_column("Description", style="yellow")
    commands_table.add_column("Options", style="cyan")
    
    # Command groups and their commands
    command_groups = {
        "Detection Commands": [
            ("detect-bricks", "Detect LEGO bricks in images", [
                "--image PATH", "--output PATH", "--conf FLOAT",
                "--save-annotated/--no-save-annotated",
                "--save-json/--no-save-json",
                "--clean-exif/--no-clean-exif"
            ]),
            ("detect-studs", "Detect studs on LEGO bricks", [
                "--image PATH", "--output PATH", "--conf FLOAT",
                "--save-annotated/--no-save-annotated",
                "--clean-exif/--no-clean-exif"
            ]),
            ("infer", "Run full detection pipeline", [
                "--image PATH", "--output PATH",
                "--save-annotated/--no-save-annotated",
                "--force-run/--no-force-run"
            ]),
            ("batch-inference", "Run batch inference to generate YOLO annotations", [
                "--input PATH", "--output PATH"
            ])
        ],
        "Training Commands": [
            ("train", "Train detection models", [
                "--mode [bricks|studs]", "--epochs INT", "--batch-size INT",
                "--show-results/--no-show-results", "--cleanup/--no-cleanup",
                "--force-extract", "--use-pretrained"
            ])
        ],
        "Data Processing": [
            ("data-processing labelme-to-yolo", "Convert LabelMe to YOLO format", [
                "--input PATH"
            ]),
            ("data-processing keypoints-to-bboxes", "Convert keypoints to bboxes", [
                "--input PATH", "--area-ratio FLOAT"
            ]),
            ("data-processing visualize", "Visualize annotations", [
                "--input PATH", "--labels PATH", "--grid-size TEXT"
            ]),
            ("data-processing demo-conversion", "Generate visual demonstration of the annotation conversion pipeline", [
                "--input PATH", "--samples INT", "--output PATH"
            ])
        ],
        "Metadata Commands": [
            ("metadata inspect", "Inspect image metadata", [
                "IMAGE"
            ]),
            ("metadata clean-batch", "Clean metadata from images", [
                "FOLDER", "--force"
            ])
        ],
        "Utility Commands": [
            ("cleanup", "Clean temporary files", [
                "--all", "--logs-only", "--cache-only", "--results-only"
            ]),
            ("version", "Show version information", [])
        ]
    }
    
    for group_name, commands in command_groups.items():
        # Add group header
        commands_table.add_row(
            f"[bold]{group_name}[/bold]", "", "",
            style="bold white on blue"
        )
        
        # Add commands in group
        for cmd, desc, opts in commands:
            options_str = "\n".join(opts) if opts else "None"
            commands_table.add_row(cmd, desc, options_str)
        
        # Add separator
        commands_table.add_row("", "", "")
    
    console.print(commands_table)
    
    # Global options panel
    console.print(Panel(
        "[bold]Global Options:[/bold]\n" +
        "  [cyan]--debug/--no-debug[/cyan]  Enable debug output\n" +
        "  [cyan]--version[/cyan]           Show version information\n" +
        "  [cyan]--help[/cyan]             Show this help message",
        title="Options",
        border_style="green"
    ))
    
    return True

@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug output.')
@click.option('--version', is_flag=True, callback=print_version, expose_value=False, is_eager=True,
              help='Show version information and exit.')
def cli(debug):
    """LEGO Bricks ML Vision - Command Line Interface
    
    This tool provides commands for detecting LEGO bricks, classifying their dimensions,
    training models, and processing datasets.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Show rich help if --help is used
    ctx = click.get_current_context()
    if '--help' in sys.argv and RICH_AVAILABLE:
        if display_rich_help():
            ctx.exit()

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

    Arguments:
        --image PATH          Input image or directory [required]
        --output PATH        Output directory (default: ./results/brick_detection)
        --conf FLOAT         Detection confidence threshold (default: 0.25)
        --save-annotated     Save visualization of detections (default: True)
        --save-json         Save detection results as JSON (default: False)
        --clean-exif        Remove EXIF metadata before processing (default: False)

    Example:
        lego_cli.py detect-bricks --image photos/test.jpg --conf 0.3 --save-json
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
                    force_rerun=force_run
                )
                
                progress.update(task, advance=60)
                
        else:
            # Run full algorithm without progress display
            result = run_full_algorithm(
                img_path,
                save_annotated=save_annotated,
                output_folder=img_output,
                force_rerun=force_run
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
@click.option('--output', type=click.Path(),
              help='Output folder for YOLO annotations (default: cache/datasets/processed/YOLO_labels).')
@click.option('--clean/--no-clean', default=False,
              help='Clean output directory before processing.')
def labelme_to_yolo_cmd(input, output, clean):
    """Convert LabelMe JSON annotations to YOLO format.
    
    Processes all JSON files in input folder and generates corresponding YOLO txt files.
    Original files are preserved, and a conversion summary is displayed.
    """
    if not output:
        output = os.path.join("cache", "datasets", "processed", "YOLO_labels")
    
    class Args:
        def __init__(self, input, output, clean):
            self.input = input
            self.output = output
            self.clean = clean
    
    args = Args(input, output, clean)
    
    if RICH_AVAILABLE:
        console.print(Panel.fit("[bold blue]Converting LabelMe to YOLO Format[/bold blue]"))
        
        with Progress() as progress:
            task = progress.add_task("[green]Converting annotations...", total=100)
            results = data_utils.convert_labelme_to_yolo(args)
            progress.update(task, completed=100)
            
        if results:  # Only display summary if results are returned
            display_conversion_summary(results, console)
    else:
        results = data_utils.convert_labelme_to_yolo(args)
        logger.info(f"Processed {results['total']} files")
        logger.info(f"Success: {results['success']}, Failed: {results['failed']}")

@data_processing.command('keypoints-to-bboxes')
@click.option('--input', required=True, type=click.Path(exists=True),
              help='Input folder containing keypoints JSON files.')
@click.option('--output', type=click.Path(),
              help='Output folder for bounding boxes (default: cache/datasets/processed/bboxes).')
@click.option('--area-ratio', type=float, default=0.4,
              help='Total area ratio for bounding boxes.')
@click.option('--clean/--no-clean', default=False,
              help='Clean output directory before processing.')
def keypoints_to_bboxes_cmd(input, output, area_ratio, clean):
    """Convert keypoints from LabelMe JSON into bounding boxes.
    
    Processes all JSON files in input folder and generates corresponding bounding box annotations.
    Original files are preserved, and a conversion summary is displayed.
    """
    if not output:
        output = os.path.join("cache", "datasets", "processed", "bboxes")
    
    class Args:
        def __init__(self, input, output, area_ratio, clean):
            self.input = input
            self.output = output
            self.area_ratio = area_ratio
            self.clean = clean
    
    args = Args(input, output, area_ratio, clean)
    
    if RICH_AVAILABLE:
        console.print(Panel.fit("[bold blue]Converting Keypoints to Bounding Boxes[/bold blue]"))
        
        with Progress() as progress:
            task = progress.add_task("[green]Converting keypoints...", total=100)
            results = data_utils.convert_keypoints_to_bboxes(args)
            progress.update(task, completed=100)
            
        display_conversion_summary(results, console)
    else:
        results = data_utils.convert_keypoints_to_bboxes(args)
        logger.info(f"Processed {results['total']} files")
        logger.info(f"Success: {results['success']}, Failed: {results['failed']}")

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

@data_processing.command('demo-conversion')
@click.option('--input', required=True, type=click.Path(exists=True),
              help='Input folder containing LabelMe JSON files.')
@click.option('--samples', default=3, type=int,
              help='Number of sample images to process (default: 3).')
@click.option('--output', type=click.Path(),
              help='Output folder for demo results (default: results/demo_conversion).')
def demo_conversion_cmd(input, samples, output):
    """Generate visual demonstration of the annotation conversion pipeline.
    
    Creates a side-by-side comparison showing:
    1. Original image in grayscale
    2. LabelMe annotations visualization
    3. YOLO format annotations visualization
    
    Results are saved in a timestamped folder within the output directory.
    """
    if not output:
        output = os.path.join("results", "demo_conversion")
    
    # Create timestamped output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output, f"demo_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    if RICH_AVAILABLE:
        console.print(Panel.fit("[bold blue]Annotation Conversion Demo[/bold blue]"))
        
        with Progress() as progress:
            # List and sample JSON files
            json_files = glob.glob(os.path.join(input, "*.json"))
            if len(json_files) == 0:
                console.print("[red]No JSON files found in input folder[/red]")
                return
                
            selected_files = random.sample(json_files, min(samples, len(json_files)))
            
            # Process each sample
            task = progress.add_task(
                "[green]Processing samples...", 
                total=len(selected_files)
            )
            
            for json_file in selected_files:
                demo_result = data_utils.create_conversion_demo(
                    json_file, 
                    output_folder
                )
                if demo_result:
                    console.print(f"[green]✓[/green] Processed {os.path.basename(json_file)}")
                else:
                    console.print(f"[red]✗[/red] Failed to process {os.path.basename(json_file)}")
                progress.update(task, advance=1)
                
        console.print(f"\n[bold green]Demo results saved to:[/bold green] {output_folder}")
    else:
        # Non-rich version
        json_files = glob.glob(os.path.join(input, "*.json"))
        if len(json_files) == 0:
            click.echo("No JSON files found in input folder")
            return
            
        selected_files = random.sample(json_files, min(samples, len(json_files)))
        
        with click.progressbar(selected_files, label='Processing samples') as bar:
            for json_file in bar:
                data_utils.create_conversion_demo(json_file, output_folder)
                
        click.echo(f"\nDemo results saved to: {output_folder}")

@cli.command('cleanup')
@click.option('--all', 'all_files', is_flag=True, help='Clean all temporary files, including results.')
@click.option('--logs-only', is_flag=True, help='Clean only log files.')
@click.option('--cache-only', is_flag=True, help='Clean only cache files.')
@click.option('--results-only', is_flag=True, help='Clean only results files.')
def cleanup_cmd(all_files, logs_only, cache_only, results_only):
    """Clean up temporary files and directories."""
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

    def clean_folder(folder_path):
        """Helper function to clean a folder safely"""
        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                return True
            
            # Remove all contents but keep folder
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                try:
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        import shutil
                        shutil.rmtree(item_path)
                except Exception as e:
                    logger.error(f"Error removing {item}: {e}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error handling folder {folder_path}: {e}")
            return False
    
    if RICH_AVAILABLE:
        console.print(Panel.fit("[bold yellow]Cleaning Up Folder Contents[/bold yellow]"))
        
        # Display what will be cleaned
        table = Table(title="Folders to Clean")
        table.add_column("Folder", style="cyan")
        table.add_column("Status", style="yellow")
        
        for folder in to_clean:
            folder_path = os.path.join(os.getcwd(), folder)
            if os.path.exists(folder_path):
                table.add_row(folder, "Will be emptied")
            else:
                table.add_row(folder, "Will be created")
                
        console.print(table)
        
        # Confirm cleanup
        if click.confirm("Do you want to proceed with cleanup?", default=True):
            with console.status("[bold green]Cleaning up...") as status:
                success = True
                for folder in to_clean:
                    folder_path = os.path.join(os.getcwd(), folder)
                    if clean_folder(folder_path):
                        console.print(f"[green]✓[/green] Processed {folder}")
                    else:
                        console.print(f"[red]✗[/red] Failed to process {folder}")
                        success = False
                
                if success:
                    console.print("[bold green]Cleanup completed successfully![/bold green]")
                else:
                    console.print("[bold yellow]Cleanup completed with some errors.[/bold yellow]")
        else:
            console.print("[yellow]Cleanup cancelled.[/yellow]")
    else:
        # Non-rich version
        for folder in to_clean:
            folder_path = os.path.join(os.getcwd(), folder)
            if os.path.exists(folder_path):
                click.echo(f"Will empty: {folder}")
            else:
                click.echo(f"Will create: {folder}")
                
        if click.confirm("Do you want to proceed with cleanup?", default=True):
            success = True
            for folder in to_clean:
                folder_path = os.path.join(os.getcwd(), folder)
                if clean_folder(folder_path):
                    click.echo(f"Processed {folder}")
                else:
                    click.echo(f"Failed to process {folder}")
                    success = False
            
            if success:
                click.echo("Cleanup completed successfully!")
            else:
                click.echo("Cleanup completed with some errors.")
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

@cli.group('metadata')
def metadata():
    """Commands for inspecting and managing image metadata."""
    pass

@metadata.command('inspect')
@click.argument('image', type=click.Path(exists=True))
def inspect_metadata_cmd(image):
    """Display detailed metadata information for a single image."""
    from utils.exif_utils import read_exif
    
    if RICH_AVAILABLE:
        console.print(Panel.fit(f"[bold blue]Metadata Inspection for {os.path.basename(image)}[/bold blue]"))
        
        metadata = read_exif(image)
        if not metadata:
            console.print("[yellow]No metadata found in image.[/yellow]")
            return

        # Create a rich table view of the metadata
        table = Table(show_header=False, box=ROUNDED)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        
        def add_dict_to_table(d, prefix=''):
            for key, value in d.items():
                if isinstance(value, dict):
                    table.add_row(f"{prefix}{key}", "[blue]<nested>[/blue]")
                    add_dict_to_table(value, prefix + '  ')
                else:
                    table.add_row(f"{prefix}{key}", str(value))
        
        add_dict_to_table(metadata)
        console.print(table)
    else:
        metadata = read_exif(image)
        if not metadata:
            click.echo("No metadata found in image.")
            return
        click.echo(json.dumps(metadata, indent=2))

@metadata.command('clean-batch')
@click.argument('folder', type=click.Path(exists=True))
@click.option('--force', is_flag=True, help='Skip confirmation prompt.')
def clean_batch_metadata_cmd(folder, force):
    """Clean metadata from all images in a folder."""
    from utils.exif_utils import read_exif, clean_exif_metadata
    
    # Find all images in folder
    images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        images.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    
    if not images:
        if RICH_AVAILABLE:
            console.print("[yellow]No images found in folder.[/yellow]")
        else:
            click.echo("No images found in folder.")
        return
    
    # Check which images have metadata
    images_with_meta = []
    for img in images:
        if read_exif(img):
            images_with_meta.append(img)
    
    if not images_with_meta:
        if RICH_AVAILABLE:
            console.print("[yellow]No images with metadata found.[/yellow]")
        else:
            click.echo("No images with metadata found.")
        return
    
    # Show summary
    if RICH_AVAILABLE:
        console.print(Panel.fit(f"[bold blue]Batch Metadata Cleaning[/bold blue]"))
        table = Table(title="Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")
        table.add_row("Total images", str(len(images)))
        table.add_row("Images with metadata", str(len(images_with_meta)))
        console.print(table)
    else:
        click.echo(f"Total images: {len(images)}")
        click.echo(f"Images with metadata: {len(images_with_meta)}")
    
    # Ask for confirmation unless --force is used
    if not force:
        if not click.confirm("Do you want to clean metadata from these images?"):
            if RICH_AVAILABLE:
                console.print("[yellow]Operation cancelled.[/yellow]")
            else:
                click.echo("Operation cancelled.")
            return
    
    # Clean metadata
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[green]Cleaning metadata...", total=len(images_with_meta))
            
            for img in images_with_meta:
                clean_exif_metadata(img)
                progress.update(task, advance=1)
                
        console.print("[bold green]Metadata cleaning completed![/bold green]")
    else:
        with click.progressbar(images_with_meta, label='Cleaning metadata') as bar:
            for img in bar:
                clean_exif_metadata(img)
        click.echo("Metadata cleaning completed!")

@cli.command('batch-inference')
@click.option('--input', required=True, type=click.Path(exists=True),
              help='Input folder containing images to process.')
@click.option('--output', type=click.Path(),
              help='Output folder for YOLO annotations (default: results/batch_inference).')
def batch_inference_cmd(input, output):
    """
    Run batch inference to generate YOLO annotations.
    
    Processes all images in the input folder:
    1. Detects LEGO bricks
    2. For each brick, detects studs and classifies dimensions
    3. Generates YOLO format annotations for valid detections
    4. Saves metadata including class mappings
    
    Example:
        lego_cli.py batch-inference --input dataset/images --output dataset/labels
    """
    if not output:
        output = os.path.join("results", "batch_inference")
    
    # Get dimensions mapping from config
    dimensions_map = config.get("BRICKS_DIMENSIONS_CLASSES", {})
    
    if RICH_AVAILABLE:
        console.print(Panel.fit("[bold blue]Batch Inference Processing[/bold blue]"))
        
        # Show dimensions mapping
        map_table = Table(title="Class ID Mapping")
        map_table.add_column("ID", style="cyan")
        map_table.add_column("Dimension", style="green")
        
        for class_id, dimension in dimensions_map.items():
            map_table.add_row(str(class_id), dimension)
        
        console.print(map_table)
        
        # Process with progress tracking
        with Progress() as progress:
            task = progress.add_task(
                "[green]Processing images...",
                total=len(list(Path(input).glob('*.[jp][pn][g]')))
            )
            
            stats = process_batch_inference(
                input,
                output,
                dimensions_map,
                progress
            )
        
        # Display results
        display_batch_results(stats, console)
        
        console.print(f"\n[bold green]Results saved to:[/bold green] {output}")
        
    else:
        # Non-rich version
        logger.info("Processing images from: %s", input)
        stats = process_batch_inference(input, output, dimensions_map)
        logger.info("Processing complete:")
        logger.info("- Processed %d images", stats['processed_images'])
        logger.info("- Found %d bricks (%d valid, %d unknown)",
                   stats['total_bricks'],
                   stats['valid_bricks'],
                   stats['unknown_bricks'])
        logger.info("Results saved to: %s", output)

@cli.command('visualize-batch')
@click.argument('metadata_json', type=click.Path(exists=True))
@click.option('--output', type=click.Path(), help='Output folder for visualization (defaults to metadata output folder).')
@click.option('--samples', default=6, type=int, help='Number of sample images to visualize (default: 6).')
def visualize_batch_cmd(metadata_json, output, samples):
    """
    Create grid visualizations of randomly selected images with their annotations.
    
    Takes a batch inference metadata JSON file and creates one or more 3x3 grid 
    visualizations showing the YOLO annotations with proper class labels.
    """
    # Load metadata
    try:
        with open(metadata_json, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata JSON: {e}")
        return

    # Use output folder from metadata if not specified
    if not output:
        output = metadata['config']['output_folder']
    os.makedirs(output, exist_ok=True)

    console = Console()
    console.print("[bold blue]Batch Visualization[/bold blue]")
    
    with Progress() as progress:
        task = progress.add_task("[green]Creating visualizations...", total=100)
        
        try:
            results = create_batch_visualization(
                metadata_json=metadata_json,
                output_folder=output,
                num_samples=samples
            )
            progress.update(task, completed=100)
            
            if results:
                console.print(f"\n[green]Created {len(results)} visualization grids:[/green]")
                for path in results:
                    console.print(f"  • {path}")
            else:
                console.print("[red]Failed to create visualizations[/red]")
        except Exception as e:
            console.print(f"[red]Error creating visualizations: {e}[/red]")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Execute the CLI
    cli()