Command-line interfaces
Script parameterization
Task scheduling
Process automation
Batch processing
Logging and monitoring
Configuration management
Email and notification integration

# Automation & Scripting for Data Professionals

Welcome to the Automation & Scripting module! This guide focuses on leveraging Python to automate data processes and create robust scripts - essential skills that will help data governance and analytics professionals save time and ensure consistency in their workflows.

## Why Automation & Scripting Matter

Automation is a critical skill for data professionals because it allows you to:
- Save time by eliminating repetitive manual tasks
- Ensure consistency and reduce human error in data processes
- Create reproducible workflows for data governance
- Schedule routine tasks to run automatically
- Document data transformations and lineage
- Scale your data operations efficiently
- Focus your time on analysis rather than preparation

## Module Overview

This module covers key automation and scripting techniques:

1. [Command-line Interfaces](#command-line-interfaces)
2. [Script Parameterization](#script-parameterization)
3. [Task Scheduling](#task-scheduling)
4. [Process Automation](#process-automation)
5. [Batch Processing](#batch-processing)
6. [Logging and Monitoring](#logging-and-monitoring)
7. [Configuration Management](#configuration-management)
8. [Email and Notification Integration](#email-and-notification-integration)
9. [Mini-Project: Data Governance Automation Suite](#mini-project-data-governance-automation-suite)

## Command-line Interfaces

Building usable command-line scripts:

```python
import argparse
import pandas as pd
import os
import sys
from datetime import datetime

def create_basic_cli():
    """Create a basic command-line interface for a data processing script"""
    
    # Create the parser
    parser = argparse.ArgumentParser(
        description='Process data files with various operations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Add arguments
    parser.add_argument(
        '--input', '-i', 
        required=True,
        help='Path to input data file (CSV or Excel)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Path to output file (defaults to input_processed.csv)'
    )
    
    parser.add_argument(
        '--operation', '-op',
        choices=['validate', 'clean', 'transform', 'all'],
        default='all',
        help='Operation to perform:\n'
             '  validate: Check data for issues\n'
             '  clean: Clean and standardize data\n'
             '  transform: Apply transformations\n'
             '  all: Perform all operations (default)'
    )
    
    parser.add_argument(
        '--log', '-l',
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        help='Set the logging level (default: info)'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Process arguments
    input_file = args.input
    
    # Generate default output filename if not provided
    if args.output:
        output_file = args.output
    else:
        # Get the base filename without extension
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_processed.csv"
    
    # Validate input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    # Set up logging based on specified level
    import logging
    log_levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    
    logging.basicConfig(
        level=log_levels[args.log],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=f"{base_name}_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    # If verbose, also log to console
    if args.verbose:
        console = logging.StreamHandler()
        console.setLevel(log_levels[args.log])
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
    
    logging.info(f"Starting processing with operation: {args.operation}")
    logging.info(f"Input file: {input_file}")
    logging.info(f"Output file: {output_file}")
    
    # Load configuration if provided
    config = {}
    if args.config:
        if not os.path.exists(args.config):
            logging.error(f"Configuration file '{args.config}' not found.")
            sys.exit(1)
        
        logging.info(f"Loading configuration from {args.config}")
        try:
            import json
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
            sys.exit(1)
    
    # Load data
    try:
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(input_file)
        else:
            logging.error(f"Unsupported file format: {input_file}")
            sys.exit(1)
        
        logging.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        sys.exit(1)
    
    # Process data based on specified operation
    if args.operation in ['validate', 'all']:
        df = validate_data(df, config)
    
    if args.operation in ['clean', 'all']:
        df = clean_data(df, config)
    
    if args.operation in ['transform', 'all']:
        df = transform_data(df, config)
    
    # Save the processed data
    try:
        df.to_csv(output_file, index=False)
        logging.info(f"Saved processed data to {output_file}")
    except Exception as e:
        logging.error(f"Error saving data: {str(e)}")
        sys.exit(1)
    
    logging.info("Processing completed successfully")
    
    # Return success
    return 0

def validate_data(df, config):
    """Validate data for issues"""
    logging.info("Validating data...")
    # Implement validation logic here
    return df

def clean_data(df, config):
    """Clean and standardize data"""
    logging.info("Cleaning data...")
    # Implement cleaning logic here
    return df

def transform_data(df, config):
    """Apply transformations to data"""
    logging.info("Transforming data...")
    # Implement transformation logic here
    return df

if __name__ == "__main__":
    sys.exit(create_basic_cli())
```

## Creating a Rich Command-line Interface

Building a more user-friendly CLI with progress bars, colors, and interactive elements:

```python
import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime
import argparse
import json
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm
from rich import print as rprint

def create_rich_cli():
    """Create a rich command-line interface for data processing"""
    
    # Initialize rich console
    console = Console()
    
    # Print header
    console.print(Panel("🔄 [bold blue]Data Processing Toolkit[/bold blue] 🔄", 
                        subtitle="Automate your data workflows"))
    
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Process data files with advanced options',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Add arguments
    parser.add_argument('--input', '-i', required=True, help='Path to input data file')
    parser.add_argument('--output', '-o', help='Path to output file')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        console.print(f"[bold red]Error:[/bold red] Input file '{args.input}' not found.")
        return 1
    
    # Set default output file if not provided
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_processed.csv"
    
    # Load configuration if provided
    config = {}
    if args.config:
        if not os.path.exists(args.config):
            console.print(f"[bold red]Error:[/bold red] Configuration file '{args.config}' not found.")
            return 1
        
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
                
            console.print(f"[green]Configuration loaded from {args.config}[/green]")
        except Exception as e:
            console.print(f"[bold red]Error loading configuration:[/bold red] {str(e)}")
            return 1
    
    # Show processing options
    console.print("\n[bold]Processing Options:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Option")
    table.add_column("Value")
    
    table.add_row("Input file", args.input)
    table.add_row("Output file", args.output)
    table.add_row("Configuration", args.config if args.config else "Default")
    table.add_row("Verbose mode", "Enabled" if args.verbose else "Disabled")
    
    console.print(table)
    
    # Confirm processing
    if not Confirm.ask("\nProceed with processing?"):
        console.print("[yellow]Operation cancelled by user[/yellow]")
        return 0
    
    # Start processing
    try:
        console.print("\n[bold]Loading data...[/bold]")
        
        # Determine file type and load data
        if args.input.endswith('.csv'):
            df = pd.read_csv(args.input)
        elif args.input.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(args.input)
        else:
            console.print(f"[bold red]Error:[/bold red] Unsupported file format: {args.input}")
            return 1
        
        console.print(f"[green]Loaded dataset with {len(df)} rows and {len(df.columns)} columns[/green]")
        
        # Show data sample
        console.print("\n[bold]Data Sample:[/bold]")
        sample_table = Table(show_header=True, header_style="bold cyan")
        
        # Add columns
        for column in df.columns[:5]:  # Limit to first 5 columns for display
            sample_table.add_column(column)
        
        # Add rows
        for _, row in df.head(5).iterrows():  # Show first 5 rows
            sample_table.add_row(*[str(row[col]) for col in df.columns[:5]])
        
        console.print(sample_table)
        
        if len(df.columns) > 5:
            console.print(f"[dim](Showing only first 5 of {len(df.columns)} columns)[/dim]")
        
        # Process data with progress bars
        with Progress(
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(bar_width=50),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            # Validate data
            validate_task = progress.add_task("[cyan]Validating data...[/cyan]", total=100)
            
            # Simulate processing steps
            for i in range(101):
                time.sleep(0.01)  # Simulate work
                progress.update(validate_task, completed=i)
            
            # Data cleaning
            clean_task = progress.add_task("[green]Cleaning data...[/green]", total=100)
            
            for i in range(101):
                time.sleep(0.015)  # Simulate work
                progress.update(clean_task, completed=i)
            
            # Transformation
            transform_task = progress.add_task("[magenta]Transforming data...[/magenta]", total=100)
            
            for i in range(101):
                time.sleep(0.012)  # Simulate work
                progress.update(transform_task, completed=i)
        
        # Save results
        console.print("\n[bold]Saving processed data...[/bold]")
        df.to_csv(args.output, index=False)
        
        # Show processing summary
        console.print(Panel(f"[bold green]Processing Complete![/bold green]\n"
                            f"Input file: {args.input}\n"
                            f"Output file: {args.output}\n"
                            f"Rows processed: {len(df)}\n"
                            f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
        
        return 0
        
    except Exception as e:
        console.print(f"[bold red]Error during processing:[/bold red] {str(e)}")
        import traceback
        if args.verbose:
            console.print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(create_rich_cli())
```

## Script Parameterization

Designing flexible scripts with configuration options:

```python
import pandas as pd
import yaml
import json
import os
import sys
from typing import Dict, Any, Optional, List, Union
import logging

class ConfigurableScript:
    """Base class for configurable data processing scripts"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the script with configuration
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        # Default configuration
        self.config = {
            'input': {
                'file_path': None,
                'file_type': 'csv',
                'encoding': 'utf-8',
                'delimiter': ',',
                'sheet_name': 0,
                'header': 0
            },
            'output': {
                'file_path': None,
                'file_type': 'csv',
                'encoding': 'utf-8',
                'delimiter': ',',
                'index': False
            },
            'processing': {
                'operations': ['validate', 'clean', 'transform'],
                'batch_size': 10000,
                'skip_errors': False
            },
            'logging': {
                'level': 'info',
                'file': None,
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
        
        # Set up logging
        self._setup_logging()
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine file type from extension
        _, ext = os.path.splitext(config_path)
        
        try:
            if ext.lower() in ['.yml', '.yaml']:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
            elif ext.lower() == '.json':
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {ext}")
            
            # Update configuration (recursive dictionary update)
            self._update_dict(self.config, loaded_config)
            
        except Exception as e:
            raise ValueError(f"Error loading configuration: {str(e)}")
    
    def _update_dict(self, d: Dict, u: Dict) -> Dict:
        """Recursively update a dictionary with another dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _setup_logging(self) -> None:
        """Set up logging based on configuration"""
        log_config = self.config['logging']
        
        # Map string levels to logging levels
        level_map = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        level = level_map.get(log_config['level'].lower(), logging.INFO)
        
        # Configure logging
        logging.basicConfig(
            level=level,
            format=log_config['format'],
            filename=log_config['file']
        )
        
        # Add console handler if no file specified
        if not log_config['file']:
            console = logging.StreamHandler()
            console.setLevel(level)
            formatter = logging.Formatter(log_config['format'])
            console.setFormatter(formatter)
            logging.getLogger('').addHandler(console)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Logging initialized")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data based on configuration
        
        Returns:
            Pandas DataFrame with loaded data
        """
        input_config = self.config['input']
        file_path = input_config['file_path']
        
        if not file_path:
            raise ValueError("Input file path not specified in configuration")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        self.logger.info(f"Loading data from {file_path}")
        
        try:
            file_type = input_config['file_type'].lower()
            
            if file_type == 'csv':
                df = pd.read_csv(
                    file_path,
                    encoding=input_config['encoding'],
                    delimiter=input_config['delimiter'],
                    header=input_config['header']
                )
            elif file_type in ['xls', 'xlsx', 'excel']:
                df = pd.read_excel(
                    file_path,
                    sheet_name=input_config['sheet_name'],
                    header=input_config['header']
                )
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def save_data(self, df: pd.DataFrame) -> None:
        """
        Save data based on configuration
        
        Args:
            df: Pandas DataFrame to save
        """
        output_config = self.config['output']
        file_path = output_config['file_path']
        
        if not file_path:
            # Generate default output path based on input path
            input_path = self.config['input']['file_path']
            base_name = os.path.splitext(input_path)[0]
            file_path = f"{base_name}_processed.{output_config['file_type']}"
            self.config['output']['file_path'] = file_path
        
        self.logger.info(f"Saving data to {file_path}")
        
        try:
            file_type = output_config['file_type'].lower()
            
            if file_type == 'csv':
                df.to_csv(
                    file_path,
                    encoding=output_config['encoding'],
                    sep=output_config['delimiter'],
                    index=output_config['index']
                )
            elif file_type in ['xls', 'xlsx', 'excel']:
                df.to_excel(
                    file_path,
                    index=output_config['index']
                )
            else:
                raise ValueError(f"Unsupported output file type: {file_type}")
            
            self.logger.info(f"Saved {len(df)} rows to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process data according to configuration
        
        Args:
            df: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        operations = self.config['processing']['operations']
        self.logger.info(f"Processing data with operations: {operations}")
        
        processed_df = df.copy()
        
        try:
            for operation in operations:
                if operation == 'validate':
                    processed_df = self.validate(processed_df)
                elif operation == 'clean':
                    processed_df = self.clean(processed_df)
                elif operation == 'transform':
                    processed_df = self.transform(processed_df)
                else:
                    self.logger.warning(f"Unknown operation: {operation}")
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            if not self.config['processing']['skip_errors']:
                raise
            return df
    
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data (to be implemented by subclasses)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validated DataFrame
        """
        self.logger.info("Validation not implemented")
        return df
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data (to be implemented by subclasses)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Cleaning not implemented")
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data (to be implemented by subclasses)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        self.logger.info("Transformation not implemented")
        return df
    
    def run(self) -> None:
        """Run the script end-to-end"""
        self.logger.info("Starting script execution")
        
        try:
            # Load data
            df = self.load_data()
            
            # Process data
            processed_df = self.process_data(df)
            
            # Save data
            self.save_data(processed_df)
            
            self.logger.info("Script execution completed successfully")
            
        except Exception as e:
            self.logger.error(f"Script execution failed: {str(e)}")
            raise

# Example implementation
class CustomerDataProcessor(ConfigurableScript):
    """Process customer data with validation, cleaning, and transformation"""
    
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate customer data"""
        self.logger.info("Validating customer data")
        
        # Check required columns
        required_columns = ['customer_id', 'name', 'email']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            if not self.config['processing']['skip_errors']:
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Log validation metrics
        null_counts = df[required_columns].isnull().sum()
        self.logger.info(f"Null counts in required columns: {null_counts.to_dict()}")
        
        return df
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean customer data"""
        self.logger.info("Cleaning customer data")
        
        # Make a copy to avoid modifying the input
        cleaned_df = df.copy()
        
        # Example: Clean email addresses
        if 'email' in cleaned_df.columns:
            # Convert to lowercase
            cleaned_df['email'] = cleaned_df['email'].str.lower()
            
            # Remove whitespace
            cleaned_df['email'] = cleaned_df['email'].str.strip()
            
            # Count cleaned emails
            changed_count = (df['email'] != cleaned_df['email']).sum()
            self.logger.info(f"Cleaned {changed_count} email addresses")
        
        return cleaned_df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform customer data"""
        self.logger.info("Transforming customer data")
        
        # Make a copy to avoid modifying the input
        transformed_df = df.copy()
        
        # Example: Add a full_name column if first_name and last_name exist
        if 'first_name' in df.columns and 'last_name' in df.columns:
            transformed_df['full_name'] = df['first_name'] + ' ' + df['last_name']
            self.logger.info("Added full_name column")
        
        return transformed_df

# Example usage
if __name__ == "__main__":
    # Check if configuration file provided
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        processor = CustomerDataProcessor(config_path)
    else:
        # Use default configuration
        processor = CustomerDataProcessor()
        
        # Set input file path
        processor.config['input']['file_path'] = 'customer_data.csv'
    
    # Run the processor
    processor.run()
```

Example YAML configuration file (`config.yaml`):

```yaml
input:
  file_path: data/customers.csv
  file_type: csv
  encoding: utf-8
  delimiter: ','
  header: 0

output:
  file_path: data/customers_processed.csv
  file_type: csv
  encoding: utf-8
  delimiter: ','
  index: false

processing:
  operations:
    - validate
    - clean
    - transform
  batch_size: 5000
  skip_errors: false

logging:
  level: info
  file: logs/processing.log
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
```

## Task Scheduling

Setting up scheduled data processes:

```python
import schedule
import time
import subprocess
import os
import datetime
import logging
from pathlib import Path
import sys
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_scheduler")

class DataTaskScheduler:
    """Schedule and manage data processing tasks"""
    
    def __init__(self, tasks_dir='tasks'):
        """
        Initialize the task scheduler
        
        Args:
            tasks_dir: Directory containing task configuration files
        """
        self.tasks_dir = Path(tasks_dir)
        self.jobs = []
        
        # Create tasks directory if it doesn't exist
        if not self.tasks_dir.exists():
            self.tasks_dir.mkdir(parents=True)
            logger.info(f"Created tasks directory: {self.tasks_dir}")
    
    def load_tasks(self):
        """Load tasks from configuration files"""
        logger.info("Loading tasks...")
        
        task_files = list(self.tasks_dir.glob("*.py"))
        task_configs = list(self.tasks_dir.glob("*.yaml")) + list(self.tasks_dir.glob("*.json"))
        
        logger.info(f"Found {len(task_files)} task scripts and {len(task_configs)} task configurations")
        
        # Schedule each task
        schedule.clear()
        self.jobs = []
        
        for task_file in task_files:
            # Get matching config file (same name, different extension)
            task_name = task_file.stem
            matching_configs = [
                f for f in task_configs 
                if f.stem == task_name or f.stem == f"{task_name}_config"
            ]
            
            config_file = matching_configs[0] if matching_configs else None
            
            # Schedule the task
            self._schedule_task(task_file, config_file)
    
    def _schedule_task(self, task_file, config_file=None):
        """
        Schedule a task based on its configuration
        
        Args:
            task_file: Path to the task script
            config_file: Optional path to task configuration
        """
        # Default schedule is daily at midnight
        schedule_type = "daily"
        schedule_time = "00:00"
        schedule_interval = None
        
        # Try to extract schedule from script comments
        with open(task_file, 'r') as f:
            script_content = f.read()
            
            # Look for schedule comment
            import re
            schedule_match = re.search(r'#\s*schedule:\s*(\w+)', script_content)
            time_match = re.search(r'#\s*time:\s*(\d{1,2}:\d{2})', script_content)
            interval_match = re.search(r'#\s*interval:\s*(\d+)\s*(minutes|hours)', script_content)
            
            if schedule_match:
                schedule_type = schedule_match.group(1).lower()
            
            if time_match:
                schedule_time = time_match.group(1)
            
            if interval_match:
                schedule_interval = int(interval_match.group(1))
                interval_unit = interval_match.group(2)
        
        # Create job function
        task_name = task_file.stem
        
        def run_task():
            logger.info(f"Running task: {task_name}")
            
            try:
                cmd = [sys.executable, str(task_file)]
                
                if config_file:
                    cmd.extend(['--config', str(config_file)])
                
                # Run the process
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True
                )
                
                if result.returncode == 0:
                    logger
if result.returncode == 0:
                    logger.info(f"Task {task_name} completed successfully")
                    if result.stdout:
                        logger.info(f"Output: {result.stdout}")
                else:
                    logger.error(f"Task {task_name} failed with return code {result.returncode}")
                    if result.stderr:
                        logger.error(f"Error: {result.stderr}")
                
                # Record task execution
                self._record_execution(task_name, result.returncode == 0)
                
            except Exception as e:
                logger.error(f"Error running task {task_name}: {str(e)}")
                self._record_execution(task_name, False, error=str(e))
        
        # Schedule based on type
        job = None
        
        if schedule_type == "daily":
            hour, minute = schedule_time.split(":")
            job = schedule.every().day.at(schedule_time).do(run_task)
            logger.info(f"Scheduled task {task_name} to run daily at {schedule_time}")
            
        elif schedule_type == "weekly":
            job = schedule.every().week.at(schedule_time).do(run_task)
            logger.info(f"Scheduled task {task_name} to run weekly at {schedule_time}")
            
        elif schedule_type == "monthly":
            # For monthly, we need to use a custom approach
            def monthly_job():
                # Check if it's the first day of the month
                if datetime.datetime.now().day == 1:
                    run_task()
            
            # Schedule to check daily
            job = schedule.every().day.at(schedule_time).do(monthly_job)
            logger.info(f"Scheduled task {task_name} to run monthly at {schedule_time}")
            
        elif schedule_type == "interval" and schedule_interval:
            if interval_unit == "minutes":
                job = schedule.every(schedule_interval).minutes.do(run_task)
                logger.info(f"Scheduled task {task_name} to run every {schedule_interval} minutes")
            elif interval_unit == "hours":
                job = schedule.every(schedule_interval).hours.do(run_task)
                logger.info(f"Scheduled task {task_name} to run every {schedule_interval} hours")
        
        # Store the job
        if job:
            self.jobs.append({
                'name': task_name,
                'file': task_file,
                'config': config_file,
                'job': job,
                'type': schedule_type,
                'time': schedule_time,
                'interval': f"{schedule_interval} {interval_unit}" if schedule_interval else None
            })
    
    def _record_execution(self, task_name, success, error=None):
        """
        Record task execution for reporting
        
        Args:
            task_name: Name of the task
            success: Whether execution was successful
            error: Optional error message
        """
        execution_log = Path("execution_log.csv")
        
        # Create log file if it doesn't exist
        if not execution_log.exists():
            pd.DataFrame(columns=[
                'timestamp', 'task', 'success', 'error'
            ]).to_csv(execution_log, index=False)
        
        # Append execution record
        record = pd.DataFrame([{
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'task': task_name,
            'success': success,
            'error': error or ''
        }])
        
        record.to_csv(execution_log, mode='a', header=False, index=False)
    
    def run_pending(self):
        """Run pending scheduled tasks"""
        schedule.run_pending()
    
    def run_task_now(self, task_name):
        """
        Run a specific task immediately
        
        Args:
            task_name: Name of the task to run
        """
        for job in self.jobs:
            if job['name'] == task_name:
                logger.info(f"Running task {task_name} immediately")
                job['job'].run()
                return True
        
        logger.error(f"Task {task_name} not found")
        return False
    
    def generate_schedule_report(self):
        """Generate a report of scheduled tasks"""
        report = "# Scheduled Tasks Report\n"
        report += f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add task table
        report += "| Task | Schedule | Next Run | Last Status |\n"
        report += "|------|----------|----------|-------------|\n"
        
        # Load execution log
        execution_log = Path("execution_log.csv")
        executions = pd.DataFrame()
        
        if execution_log.exists():
            executions = pd.read_csv(execution_log)
        
        for job in self.jobs:
            # Get schedule info
            if job['type'] == 'interval':
                schedule_info = f"Every {job['interval']}"
            else:
                schedule_info = f"{job['type'].title()} at {job['time']}"
            
            # Get next run time
            next_run = job['job'].next_run
            next_run_str = next_run.strftime('%Y-%m-%d %H:%M:%S') if next_run else 'N/A'
            
            # Get last status
            last_status = 'Never run'
            if not executions.empty:
                task_executions = executions[executions['task'] == job['name']]
                if not task_executions.empty:
                    last_execution = task_executions.iloc[-1]
                    if last_execution['success']:
                        last_status = '✅ Success'
                    else:
                        last_status = f"❌ Failed: {last_execution['error']}"
            
            report += f"| {job['name']} | {schedule_info} | {next_run_str} | {last_status} |\n"
        
        # Add execution history
        if not executions.empty:
            report += "\n## Recent Executions\n\n"
            report += "| Timestamp | Task | Status | Error |\n"
            report += "|-----------|------|--------|-------|\n"
            
            # Get last 10 executions
            recent = executions.tail(10).iloc[::-1]
            
            for _, row in recent.iterrows():
                status = '✅ Success' if row['success'] else '❌ Failed'
                report += f"| {row['timestamp']} | {row['task']} | {status} | {row['error']} |\n"
        
        return report
    
    def start(self):
        """Start the scheduler main loop"""
        logger.info("Starting scheduler...")
        
        # Load tasks
        self.load_tasks()
        
        try:
            while True:
                self.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {str(e)}")

# Example task script (tasks/daily_data_cleanup.py)
"""
# Daily data cleanup task
# schedule: daily
# time: 02:00

import pandas as pd
import os
import sys
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('daily_cleanup')

def main():
    logger.info("Starting daily data cleanup")
    
    # Data folders to clean
    data_folders = ['tmp', 'cache', 'exports']
    
    # Cleanup old files (older than 7 days)
    import time
    current_time = time.time()
    cutoff_time = current_time - (7 * 24 * 60 * 60)  # 7 days ago
    
    total_cleaned = 0
    
    for folder in data_folders:
        if not os.path.exists(folder):
            logger.warning(f"Folder not found: {folder}")
            continue
        
        logger.info(f"Cleaning folder: {folder}")
        
        # Iterate through files
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            
            # Check if it's a file and older than cutoff
            if os.path.isfile(file_path):
                file_time = os.path.getmtime(file_path)
                if file_time < cutoff_time:
                    try:
                        os.remove(file_path)
                        logger.info(f"Removed old file: {file_path}")
                        total_cleaned += 1
                    except Exception as e:
                        logger.error(f"Error removing file {file_path}: {str(e)}")
    
    logger.info(f"Cleanup completed: Removed {total_cleaned} old files")
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""

# Example usage
if __name__ == "__main__":
    scheduler = DataTaskScheduler()
    
    # Option to run a specific task immediately
    if len(sys.argv) > 1 and sys.argv[1] == '--run':
        if len(sys.argv) > 2:
            task_name = sys.argv[2]
            scheduler.load_tasks()
            scheduler.run_task_now(task_name)
        else:
            print("Error: No task specified")
            print("Usage: python scheduler.py --run <task_name>")
    
    # Option to generate a report
    elif len(sys.argv) > 1 and sys.argv[1] == '--report':
        scheduler.load_tasks()
        report = scheduler.generate_schedule_report()
        
        report_file = "schedule_report.md"
        if len(sys.argv) > 2:
            report_file = sys.argv[2]
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Report generated: {report_file}")
    
    # Start the scheduler
    else:
        scheduler.start()
```

## Process Automation

Automating entire workflows with multiple steps:

```python
import pandas as pd
import numpy as np
import os
import sys
import logging
import datetime
import json
import shutil
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import subprocess
from typing import Dict, List, Optional, Any, Callable, Union, Tuple

class ProcessStep:
    """Base class for a process step in a workflow"""
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a process step
        
        Args:
            name: Name of the step
            description: Description of what the step does
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"step.{name}")
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the process step
        
        Args:
            data: Input data dictionary
            
        Returns:
            Output data dictionary
        """
        self.logger.info(f"Executing step: {self.name}")
        try:
            result = self._execute(data)
            self.logger.info(f"Step {self.name} completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error in step {self.name}: {str(e)}")
            raise
    
    def _execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation of the step's execution logic (to be overridden)
        
        Args:
            data: Input data dictionary
            
        Returns:
            Output data dictionary
        """
        # Default implementation does nothing
        return data

class DataLoadStep(ProcessStep):
    """Step to load data from a file"""
    
    def __init__(self, name: str, file_path: str, file_type: str = None, **kwargs):
        """
        Initialize a data loading step
        
        Args:
            name: Step name
            file_path: Path to the data file
            file_type: Optional file type (inferred from extension if not provided)
            **kwargs: Additional parameters to pass to the loading function
        """
        super().__init__(name, f"Load data from {file_path}")
        self.file_path = file_path
        self.file_type = file_type
        self.kwargs = kwargs
    
    def _execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load data from the file"""
        self.logger.info(f"Loading data from {self.file_path}")
        
        # Check if file exists
        if not os.path.exists(self.file_path):
            self.logger.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        # Determine file type if not provided
        file_type = self.file_type
        if not file_type:
            _, ext = os.path.splitext(self.file_path)
            file_type = ext.lstrip('.').lower()
        
        # Load data based on file type
        df = None
        
        if file_type in ['csv']:
            df = pd.read_csv(self.file_path, **self.kwargs)
        elif file_type in ['xls', 'xlsx', 'excel']:
            df = pd.read_excel(self.file_path, **self.kwargs)
        elif file_type in ['json']:
            df = pd.read_json(self.file_path, **self.kwargs)
        elif file_type in ['parquet']:
            df = pd.read_parquet(self.file_path, **self.kwargs)
        elif file_type in ['sql', 'sqlite']:
            # Assuming SQL query is in kwargs
            from sqlalchemy import create_engine
            engine = create_engine(f"sqlite:///{self.file_path}")
            query = self.kwargs.pop('query', 'SELECT * FROM data')
            df = pd.read_sql(query, engine, **self.kwargs)
        else:
            self.logger.error(f"Unsupported file type: {file_type}")
            raise ValueError(f"Unsupported file type: {file_type}")
        
        self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Update the data dictionary with the loaded dataframe
        result = data.copy()
        result[f'{self.name}_df'] = df
        
        # Also add basic statistics
        result[f'{self.name}_stats'] = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'missing_counts': df.isnull().sum().to_dict()
        }
        
        return result

class DataTransformStep(ProcessStep):
    """Step to transform data using a function"""
    
    def __init__(
        self, 
        name: str, 
        transform_fn: Callable[[pd.DataFrame], pd.DataFrame],
        input_key: str,
        output_key: Optional[str] = None,
        description: str = ""
    ):
        """
        Initialize a data transformation step
        
        Args:
            name: Step name
            transform_fn: Function that takes a DataFrame and returns a transformed DataFrame
            input_key: Key in the data dictionary for the input DataFrame
            output_key: Key in the data dictionary for the output DataFrame (defaults to input_key)
            description: Description of the transformation
        """
        desc = description or f"Transform data from {input_key}"
        super().__init__(name, desc)
        self.transform_fn = transform_fn
        self.input_key = input_key
        self.output_key = output_key or input_key
    
    def _execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transformation function to the data"""
        # Check if input dataframe exists
        if self.input_key not in data:
            self.logger.error(f"Input key '{self.input_key}' not found in data")
            raise KeyError(f"Input key '{self.input_key}' not found in data")
        
        input_df = data[self.input_key]
        
        if not isinstance(input_df, pd.DataFrame):
            self.logger.error(f"Input '{self.input_key}' is not a DataFrame")
            raise TypeError(f"Input '{self.input_key}' is not a DataFrame")
        
        # Apply transformation
        self.logger.info(f"Transforming data from {self.input_key}")
        output_df = self.transform_fn(input_df)
        
        if not isinstance(output_df, pd.DataFrame):
            self.logger.error("Transformation function did not return a DataFrame")
            raise TypeError("Transformation function did not return a DataFrame")
        
        self.logger.info(f"Transformation complete: {len(output_df)} rows")
        
        # Update the data dictionary
        result = data.copy()
        result[self.output_key] = output_df
        
        return result

class DataValidationStep(ProcessStep):
    """Step to validate data against rules"""
    
    def __init__(
        self, 
        name: str, 
        validation_rules: List[Dict[str, Any]],
        input_key: str,
        fail_on_error: bool = False,
        description: str = ""
    ):
        """
        Initialize a data validation step
        
        Args:
            name: Step name
            validation_rules: List of validation rules
            input_key: Key in the data dictionary for the input DataFrame
            fail_on_error: Whether to raise an exception if validation fails
            description: Description of the validation
        """
        desc = description or f"Validate data from {input_key}"
        super().__init__(name, desc)
        self.validation_rules = validation_rules
        self.input_key = input_key
        self.fail_on_error = fail_on_error
    
    def _execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the data against rules"""
        # Check if input dataframe exists
        if self.input_key not in data:
            self.logger.error(f"Input key '{self.input_key}' not found in data")
            raise KeyError(f"Input key '{self.input_key}' not found in data")
        
        input_df = data[self.input_key]
        
        if not isinstance(input_df, pd.DataFrame):
            self.logger.error(f"Input '{self.input_key}' is not a DataFrame")
            raise TypeError(f"Input '{self.input_key}' is not a DataFrame")
        
        # Apply validation rules
        self.logger.info(f"Validating data from {self.input_key} with {len(self.validation_rules)} rules")
        
        validation_results = []
        all_passed = True
        
        for i, rule in enumerate(self.validation_rules):
            rule_name = rule.get('name', f"Rule {i+1}")
            rule_type = rule.get('type', 'custom')
            columns = rule.get('columns', [])
            
            self.logger.info(f"Applying rule: {rule_name}")
            
            try:
                # Apply rule based on type
                if rule_type == 'not_null':
                    # Check for null values in specified columns
                    for col in columns:
                        if col not in input_df.columns:
                            result = {
                                'rule': rule_name,
                                'passed': False,
                                'column': col,
                                'message': f"Column '{col}' not found"
                            }
                            validation_results.append(result)
                            all_passed = False
                            continue
                        
                        null_count = input_df[col].isnull().sum()
                        passed = null_count == 0
                        
                        result = {
                            'rule': rule_name,
                            'passed': passed,
                            'column': col,
                            'message': f"{null_count} null values found" if not passed else "No null values"
                        }
                        
                        validation_results.append(result)
                        all_passed = all_passed and passed
                
                elif rule_type == 'unique':
                    # Check for unique values in specified columns
                    for col in columns:
                        if col not in input_df.columns:
                            result = {
                                'rule': rule_name,
                                'passed': False,
                                'column': col,
                                'message': f"Column '{col}' not found"
                            }
                            validation_results.append(result)
                            all_passed = False
                            continue
                        
                        duplicate_count = input_df[col].duplicated().sum()
                        passed = duplicate_count == 0
                        
                        result = {
                            'rule': rule_name,
                            'passed': passed,
                            'column': col,
                            'message': f"{duplicate_count} duplicate values found" if not passed else "All values unique"
                        }
                        
                        validation_results.append(result)
                        all_passed = all_passed and passed
                
                elif rule_type == 'range':
                    # Check if values are within specified range
                    min_val = rule.get('min')
                    max_val = rule.get('max')
                    
                    for col in columns:
                        if col not in input_df.columns:
                            result = {
                                'rule': rule_name,
                                'passed': False,
                                'column': col,
                                'message': f"Column '{col}' not found"
                            }
                            validation_results.append(result)
                            all_passed = False
                            continue
                        
                        # Apply range check
                        out_of_range = 0
                        
                        if min_val is not None and max_val is not None:
                            out_of_range = ((input_df[col] < min_val) | (input_df[col] > max_val)).sum()
                        elif min_val is not None:
                            out_of_range = (input_df[col] < min_val).sum()
                        elif max_val is not None:
                            out_of_range = (input_df[col] > max_val).sum()
                        
                        passed = out_of_range == 0
                        
                        result = {
                            'rule': rule_name,
                            'passed': passed,
                            'column': col,
                            'message': f"{out_of_range} values out of range" if not passed else "All values within range"
                        }
                        
                        validation_results.append(result)
                        all_passed = all_passed and passed
                
                elif rule_type == 'regex':
                    # Check if values match a regex pattern
                    pattern = rule.get('pattern')
                    
                    if not pattern:
                        self.logger.warning(f"No pattern specified for regex rule: {rule_name}")
                        continue
                    
                    for col in columns:
                        if col not in input_df.columns:
                            result = {
                                'rule': rule_name,
                                'passed': False,
                                'column': col,
                                'message': f"Column '{col}' not found"
                            }
                            validation_results.append(result)
                            all_passed = False
                            continue
                        
                        # Count values that don't match the pattern
                        import re
                        non_matching = input_df[col].astype(str).apply(
                            lambda x: not bool(re.match(pattern, x))
                        ).sum()
                        
                        passed = non_matching == 0
                        
                        result = {
                            'rule': rule_name,
                            'passed': passed,
                            'column': col,
                            'message': f"{non_matching} values don't match pattern" if not passed else "All values match pattern"
                        }
                        
                        validation_results.append(result)
                        all_passed = all_passed and passed
                
                elif rule_type == 'custom':
                    # Apply custom validation function
                    validation_fn = rule.get('function')
                    
                    if not validation_fn or not callable(validation_fn):
                        self.logger.warning(f"No function specified for custom rule: {rule_name}")
                        continue
                    
                    # Apply the function to get a boolean mask
                    mask = validation_fn(input_df)
                    
                    # Count failures
                    failure_count = (~mask).sum()
                    passed = failure_count == 0
                    
                    result = {
                        'rule': rule_name,
                        'passed': passed,
                        'message': f"{failure_count} validation failures" if not passed else "Validation passed"
                    }
                    
                    validation_results.append(result)
                    all_passed = all_passed and passed
                
                else:
                    self.logger.warning(f"Unknown rule type: {rule_type}")
            
            except Exception as e:
                self.logger.error(f"Error applying rule {rule_name}: {str(e)}")
                
                result = {
                    'rule': rule_name,
                    'passed': False,
                    'message': f"Error: {str(e)}"
                }
                
                validation_results.append(result)
                all_passed = False
        
        # Check if we should fail on validation errors
        if not all_passed and self.fail_on_error:
            self.logger.error("Validation failed")
            raise ValueError("Data validation failed")
        
        # Update the data dictionary with validation results
        result = data.copy()
        result[f'{self.name}_results'] = validation_results
        result[f'{self.name}_passed'] = all_passed
        
        self.logger.info(f"Validation {'passed' if all_passed else 'failed'}")
        
        return result

class DataExportStep(ProcessStep):
    """Step to export data to a file"""
    
    def __init__(
        self, 
        name: str, 
        output_path: str,
        input_key: str,
        file_type: str = None,
        **kwargs
    ):
        """
        Initialize a data export step
        
        Args:
            name: Step name
            output_path: Path to save the data
            input_key: Key in the data dictionary for the input DataFrame
            file_type: Optional file type (inferred from extension if not provided)
            **kwargs: Additional parameters to pass to the export function
        """
        super().__init__(name, f"Export data to {output_path}")
        self.output_path = output_path
        self.input_key = input_key
        self.file_type = file_type
        self.kwargs = kwargs
    
    def _execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Export data to the specified file"""
        # Check if input dataframe exists
        if self.input_key not in data:
            self.logger.error(f"Input key '{self.input_key}' not found in data")
            raise KeyError(f"Input key '{self.input_key}' not found in data")
        
        input_df = data[self.input_key]
        
        if not isinstance(input_df, pd.DataFrame):
            self.logger.error(f"Input '{self.input_key}' is not a DataFrame")
            raise TypeError(f"Input '{self.input_key}' is not a DataFrame")
        
        # Determine file type if not provided
        file_type = self.file_type
        if not file_type:
            _, ext = os.path.splitext(self.output_path)
            file_type = ext.lstrip('.').lower()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        
        # Export data based on file type
        self.logger.info(f"Exporting {len(input_df)} rows to {self.output_path}")
        
        if file_type in ['csv']:
            input_df.to_csv(self.output_path, **self.kwargs)
        elif file_type in ['xls', 'xlsx', 'excel']:
            input_df.to_excel(self.output_path, **self.kwargs)
        elif file_type in ['json']:
            input_df.to_json(self.output_path, **self.kwargs)
        elif file_type in ['parquet']:
            input_df.to_parquet(self.output_path, **self.kwargs)
        elif file_type in ['sql', 'sqlite']:
            # Assuming table name is in kwargs
            from sqlalchemy import create_engine
            engine = create_engine(f"sqlite:///{self.output_path}")
            table_name = self.kwargs.pop('table_name', 'data')
            input_df.to_sql(table_name, engine, **self.kwargs)
        else:
            self.logger.error(f"Unsupported file type: {file_type}")
            raise ValueError(f"Unsupported file type: {file_type}")
        
        self.logger.info(f"Data exported successfully to {self.output_path}")
        
        # Update the data dictionary
        result = data.copy()
        result[f'{self.name}_path'] = self.output_path
        
        return result

class EmailNotificationStep(ProcessStep):
    """Step to send an email notification"""
    
    def __init__(
        self, 
        name: str, 
        smtp_server: str,
        smtp_port: int,
        sender: str,
        recipients: List[str],
        subject: str,
        message_template: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        attachments: Optional[List[str]] = None
    ):
        """
        Initialize an email notification step
        
        Args:
            name: Step name
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            sender: Sender email address
            recipients: List of recipient email addresses
            subject: Email subject
            message_template: Email message template (may contain placeholders)
            username: Optional SMTP username
            password: Optional SMTP password
            attachments: Optional list of files to attach
        """
        super().__init__(name, f"Send email notification to {len(recipients)} recipients")
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender = sender
        self.recipients = recipients
        self.subject = subject
        self.message_template = message_template
        self.username = username
        self.password = password
        self.attachments = attachments or []
    
    def _execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send the email notification"""
        self.logger.info(f"Preparing email notification to {len(self.recipients)} recipients")
        
        # Format the subject and message with data values
        try:
            subject = self.subject.format(**data)
            message = self.message_template.format(**data)
        except KeyError as e:
            self.logger.error(f"Error formatting email: Missing key {e}")
try:
            subject = self.subject.format(**data)
            message = self.message_template.format(**data)
        except KeyError as e:
            self.logger.error(f"Error formatting email: Missing key {e}")
            raise
        
        # Create the email
        msg = MIMEMultipart()
        msg['From'] = self.sender
        msg['To'] = ', '.join(self.recipients)
        msg['Subject'] = subject
        
        # Attach the message body
        msg.attach(MIMEText(message, 'html'))
        
        # Attach files
        for attachment_path in self.attachments:
            # Check if attachment is a data key
            if attachment_path in data:
                attachment_path = data[attachment_path]
            
            if os.path.exists(attachment_path):
                with open(attachment_path, 'rb') as file:
                    part = MIMEApplication(file.read(), Name=os.path.basename(attachment_path))
                    part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
                    msg.attach(part)
                    self.logger.info(f"Attached file: {attachment_path}")
            else:
                self.logger.warning(f"Attachment not found: {attachment_path}")
        
        # Send the email
        self.logger.info("Connecting to SMTP server")
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                
                # Login if credentials provided
                if self.username and self.password:
                    server.login(self.username, self.password)
                
                # Send the email
                server.send_message(msg)
                self.logger.info("Email sent successfully")
        except Exception as e:
            self.logger.error(f"Error sending email: {str(e)}")
            raise
        
        # Update the data dictionary
        result = data.copy()
        result[f'{self.name}_sent'] = True
        result[f'{self.name}_time'] = datetime.datetime.now().isoformat()
        
        return result

class CommandStep(ProcessStep):
    """Step to execute a system command"""
    
    def __init__(
        self, 
        name: str, 
        command: Union[str, List[str]],
        shell: bool = False,
        capture_output: bool = True,
        timeout: Optional[int] = None,
        working_dir: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None
    ):
        """
        Initialize a command execution step
        
        Args:
            name: Step name
            command: Command to execute (string or list of arguments)
            shell: Whether to execute through the shell
            capture_output: Whether to capture command output
            timeout: Optional timeout in seconds
            working_dir: Optional working directory
            environment: Optional environment variables
        """
        cmd_str = command if isinstance(command, str) else ' '.join(command)
        super().__init__(name, f"Execute command: {cmd_str}")
        self.command = command
        self.shell = shell
        self.capture_output = capture_output
        self.timeout = timeout
        self.working_dir = working_dir
        self.environment = environment
    
    def _execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the command"""
        # Prepare the command (format with data values)
        command = self.command
        
        if isinstance(command, str):
            try:
                command = command.format(**data)
            except KeyError as e:
                self.logger.error(f"Error formatting command: Missing key {e}")
                raise
        elif isinstance(command, list):
            try:
                command = [arg.format(**data) if isinstance(arg, str) else arg for arg in command]
            except KeyError as e:
                self.logger.error(f"Error formatting command: Missing key {e}")
                raise
        
        # Prepare the environment
        env = os.environ.copy()
        if self.environment:
            env.update(self.environment)
        
        # Execute the command
        self.logger.info(f"Executing command: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=self.shell,
                capture_output=self.capture_output,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir,
                env=env
            )
            
            # Check the return code
            if result.returncode != 0:
                self.logger.error(f"Command failed with return code {result.returncode}")
                self.logger.error(f"Command output: {result.stdout}")
                self.logger.error(f"Command error: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
            
            self.logger.info(f"Command executed successfully with return code {result.returncode}")
            
            # Update the data dictionary
            output = data.copy()
            output[f'{self.name}_returncode'] = result.returncode
            output[f'{self.name}_stdout'] = result.stdout if self.capture_output else None
            output[f'{self.name}_stderr'] = result.stderr if self.capture_output else None
            
            return output
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out after {self.timeout} seconds")
            raise
        except Exception as e:
            self.logger.error(f"Error executing command: {str(e)}")
            raise

class ConditionalStep(ProcessStep):
    """Step that conditionally executes another step"""
    
    def __init__(
        self, 
        name: str, 
        condition: Callable[[Dict[str, Any]], bool],
        step: ProcessStep,
        else_step: Optional[ProcessStep] = None
    ):
        """
        Initialize a conditional step
        
        Args:
            name: Step name
            condition: Function that takes the data dictionary and returns a boolean
            step: Step to execute if the condition is True
            else_step: Optional step to execute if the condition is False
        """
        super().__init__(name, f"Conditionally execute {step.name}")
        self.condition = condition
        self.step = step
        self.else_step = else_step
    
    def _execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the step if the condition is met"""
        try:
            condition_result = self.condition(data)
            self.logger.info(f"Condition result: {condition_result}")
            
            if condition_result:
                self.logger.info(f"Executing step: {self.step.name}")
                return self.step.execute(data)
            elif self.else_step:
                self.logger.info(f"Executing else step: {self.else_step.name}")
                return self.else_step.execute(data)
            else:
                self.logger.info("Condition not met, skipping step")
                return data
                
        except Exception as e:
            self.logger.error(f"Error in conditional step: {str(e)}")
            raise

class Workflow:
    """Class to manage a series of process steps"""
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a workflow
        
        Args:
            name: Workflow name
            description: Workflow description
        """
        self.name = name
        self.description = description
        self.steps = []
        self.logger = logging.getLogger(f"workflow.{name}")
        
        # Set up logging if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def add_step(self, step: ProcessStep) -> None:
        """
        Add a step to the workflow
        
        Args:
            step: Process step to add
        """
        self.steps.append(step)
        self.logger.info(f"Added step: {step.name}")
    
    def run(self, initial_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the workflow
        
        Args:
            initial_data: Optional initial data dictionary
            
        Returns:
            Final data dictionary after all steps
        """
        self.logger.info(f"Starting workflow: {self.name}")
        
        # Initialize data dictionary
        data = initial_data or {}
        
        # Add workflow metadata
        data['workflow_name'] = self.name
        data['workflow_start_time'] = datetime.datetime.now().isoformat()
        
        # Execute each step in sequence
        for i, step in enumerate(self.steps):
            self.logger.info(f"Executing step {i+1}/{len(self.steps)}: {step.name}")
            
            try:
                # Update the data with step execution time
                data[f'{step.name}_start_time'] = datetime.datetime.now().isoformat()
                
                # Execute the step
                data = step.execute(data)
                
                # Update with completion time
                data[f'{step.name}_end_time'] = datetime.datetime.now().isoformat()
                data[f'{step.name}_status'] = 'success'
                
            except Exception as e:
                self.logger.error(f"Error in step {step.name}: {str(e)}")
                
                # Update data with error information
                data[f'{step.name}_end_time'] = datetime.datetime.now().isoformat()
                data[f'{step.name}_status'] = 'error'
                data[f'{step.name}_error'] = str(e)
                
                # Reraise the exception
                raise
        
        # Add workflow completion time
        data['workflow_end_time'] = datetime.datetime.now().isoformat()
        
        self.logger.info(f"Workflow {self.name} completed successfully")
        
        return data

# Example usage
def create_data_quality_workflow():
    """Create a sample data quality workflow"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("data_quality_workflow.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create the workflow
    workflow = Workflow(
        name="data_quality_workflow",
        description="Process customer data for quality issues"
    )
    
    # Define data transformation functions
    def clean_email(df):
        """Clean email addresses"""
        result = df.copy()
        if 'email' in result.columns:
            # Convert to lowercase and strip whitespace
            result['email'] = result['email'].str.lower().str.strip()
        return result
    
    def standardize_phone(df):
        """Standardize phone numbers"""
        result = df.copy()
        if 'phone' in result.columns:
            # Extract digits only and format
            result['phone'] = result['phone'].astype(str).apply(
                lambda x: ''.join(c for c in x if c.isdigit())
            )
            # Format as (XXX) XXX-XXXX if 10 digits
            result['phone'] = result['phone'].apply(
                lambda x: f"({x[:3]}) {x[3:6]}-{x[6:10]}" if len(x) == 10 else x
            )
        return result
    
    # Define validation rules
    validation_rules = [
        {
            'name': 'Required Fields',
            'type': 'not_null',
            'columns': ['customer_id', 'name', 'email']
        },
        {
            'name': 'Unique IDs',
            'type': 'unique',
            'columns': ['customer_id']
        },
        {
            'name': 'Valid Email Format',
            'type': 'regex',
            'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'columns': ['email']
        }
    ]
    
    # Add workflow steps
    workflow.add_step(DataLoadStep(
        name="load_data",
        file_path="customer_data.csv"
    ))
    
    workflow.add_step(DataTransformStep(
        name="clean_emails",
        transform_fn=clean_email,
        input_key="load_data_df",
        output_key="cleaned_df",
        description="Clean and standardize email addresses"
    ))
    
    workflow.add_step(DataTransformStep(
        name="standardize_phones",
        transform_fn=standardize_phone,
        input_key="cleaned_df",
        description="Standardize phone number formats"
    ))
    
    workflow.add_step(DataValidationStep(
        name="validate_data",
        validation_rules=validation_rules,
        input_key="cleaned_df",
        fail_on_error=False
    ))
    
    # Add conditional step to export failures if validation fails
    def validation_failed(data):
        return not data.get('validate_data_passed', True)
    
    workflow.add_step(ConditionalStep(
        name="export_failures",
        condition=validation_failed,
        step=DataExportStep(
            name="export_validation_failures",
            output_path="output/validation_failures.csv",
            input_key="cleaned_df"
        )
    ))
    
    # Export the processed data
    workflow.add_step(DataExportStep(
        name="export_data",
        output_path="output/processed_data.csv",
        input_key="cleaned_df"
    ))
    
    # Send notification if validation failed
    workflow.add_step(ConditionalStep(
        name="send_notification",
        condition=validation_failed,
        step=EmailNotificationStep(
            name="email_notification",
            smtp_server="smtp.example.com",
            smtp_port=587,
            sender="data-quality@example.com",
            recipients=["data-team@example.com"],
            subject="Data Quality Alert: Validation Failed",
            message_template="""
            <html>
            <body>
                <h2>Data Quality Alert</h2>
                <p>The data quality validation has failed for the latest customer data.</p>
                <p>Please review the validation results:</p>
                <ul>
                    {validate_data_results}
                </ul>
                <p>The validation failures have been exported to {export_validation_failures_path}.</p>
            </body>
            </html>
            """
        )
    ))
    
    return workflow

if __name__ == "__main__":
    # Create and run the workflow
    workflow = create_data_quality_workflow()
    
    try:
        result = workflow.run()
        print("Workflow completed successfully!")
        
        # Print summary
        print(f"Processed data exported to: {result.get('export_data_path')}")
        
        if not result.get('validate_data_passed', True):
            print(f"Validation failures exported to: {result.get('export_validation_failures_path')}")
            
    except Exception as e:
        print(f"Workflow failed: {str(e)}")
        sys.exit(1)
```

## Batch Processing

Handling large datasets in smaller chunks:

```python
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime
import time
import multiprocessing
from typing import Callable, Optional, List, Dict, Any, Tuple, Union
from tqdm import tqdm

class BatchProcessor:
    """Process large datasets in batches"""
    
    def __init__(
        self, 
        batch_size: int = 10000,
        num_workers: int = 1,
        show_progress: bool = True
    ):
        """
        Initialize the batch processor
        
        Args:
            batch_size: Number of rows to process in each batch
            num_workers: Number of parallel worker processes (1 = single process)
            show_progress: Whether to show a progress bar
        """
        self.batch_size = batch_size
        self.num_workers = min(num_workers, multiprocessing.cpu_count())
        self.show_progress = show_progress
        
        # Set up logging
        self.logger = logging.getLogger("batch_processor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def process_csv(
        self,
        input_file: str,
        process_fn: Callable[[pd.DataFrame], pd.DataFrame],
        output_file: Optional[str] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Process a large CSV file in batches
        
        Args:
            input_file: Path to input CSV file
            process_fn: Function to process each batch (takes and returns a DataFrame)
            output_file: Optional path to output file (if None, returns the combined result)
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            Combined processed DataFrame if output_file is None
        """
        if not os.path.exists(input_file):
            self.logger.error(f"Input file not found: {input_file}")
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Get file size and estimate number of rows
        file_size = os.path.getsize(input_file)
        self.logger.info(f"Processing file: {input_file} ({file_size / 1e6:.2f} MB)")
        
        # Read the file in chunks
        chunks = pd.read_csv(input_file, chunksize=self.batch_size, **kwargs)
        
        # Set up multiprocessing if needed
        if self.num_workers > 1:
            return self._process_parallel(chunks, process_fn, output_file)
        else:
            return self._process_sequential(chunks, process_fn, output_file)
    
    def _process_sequential(
        self,
        chunks: pd.io.parsers.TextFileReader,
        process_fn: Callable[[pd.DataFrame], pd.DataFrame],
        output_file: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Process chunks sequentially"""
        results = []
        total_rows = 0
        
        # Count chunks for progress bar
        if self.show_progress:
            try:
                # Try to get row count for better progress estimation
                with open(chunks.f.name, 'r') as f:
                    row_count = sum(1 for _ in f) - 1  # Subtract header row
                chunk_count = (row_count + self.batch_size - 1) // self.batch_size
            except:
                # Fall back to showing progress by chunk
                chunk_count = 100  # Arbitrary number, will update as we process
            
            pbar = tqdm(total=chunk_count, desc="Processing batches")
        
        # Process each chunk
        start_time = time.time()
        for i, chunk in enumerate(chunks):
            self.logger.debug(f"Processing batch {i+1} with {len(chunk)} rows")
            
            # Process the chunk
            processed_chunk = process_fn(chunk)
            
            # Handle the result
            if output_file:
                # Write to file (append after first chunk)
                mode = 'w' if i == 0 else 'a'
                header = i == 0
                processed_chunk.to_csv(output_file, mode=mode, header=header, index=False)
            else:
                # Store for later concatenation
                results.append(processed_chunk)
            
            # Update progress
            total_rows += len(chunk)
            if self.show_progress:
                pbar.update(1)
                if i + 1 > chunk_count:
                    pbar.total = i + 2
        
        # Close progress bar
        if self.show_progress:
            pbar.close()
        
        # Log completion
        elapsed_time = time.time() - start_time
        self.logger.info(f"Processed {total_rows} rows in {elapsed_time:.2f} seconds")
        self.logger.info(f"Processing rate: {total_rows / elapsed_time:.2f} rows/second")
        
        # Return combined result if not writing to file
        if not output_file and results:
            return pd.concat(results, ignore_index=True)
        
        return None
    
    def _process_chunk(
        self, 
        chunk_data: Tuple[int, pd.DataFrame, Callable]
    ) -> Tuple[int, pd.DataFrame]:
        """Process a single chunk (for parallel processing)"""
        chunk_id, chunk, process_fn = chunk_data
        
        try:
            # Process the chunk
            processed_chunk = process_fn(chunk)
            return (chunk_id, processed_chunk)
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
            raise
    
    def _process_parallel(
        self,
        chunks: pd.io.parsers.TextFileReader,
        process_fn: Callable[[pd.DataFrame], pd.DataFrame],
        output_file: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Process chunks in parallel"""
        # Read all chunks into memory
        all_chunks = []
        total_rows = 0
        
        self.logger.info(f"Reading chunks into memory...")
        for i, chunk in enumerate(chunks):
            all_chunks.append((i, chunk, process_fn))
            total_rows += len(chunk)
        
        self.logger.info(f"Read {len(all_chunks)} chunks with {total_rows} total rows")
        
        # Process chunks in parallel
        results = []
        start_time = time.time()
        
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            # Create progress bar if needed
            if self.show_progress:
                pbar = tqdm(total=len(all_chunks), desc="Processing batches")
                
                # Process chunks with progress updates
                for result in pool.imap_unordered(self._process_chunk, all_chunks):
                    results.append(result)
                    pbar.update(1)
                
                pbar.close()
            else:
                # Process chunks without progress bar
                results = pool.map(self._process_chunk, all_chunks)
        
        # Sort results by chunk ID
        results.sort(key=lambda x: x[0])
        processed_chunks = [r[1] for r in results]
        
        # Log completion
        elapsed_time = time.time() - start_time
        self.logger.info(f"Processed {total_rows} rows in {elapsed_time:.2f} seconds with {self.num_workers} workers")
        self.logger.info(f"Processing rate: {total_rows / elapsed_time:.2f} rows/second")
        
        # Write to file or return combined result
        if output_file:
            self.logger.info(f"Writing results to {output_file}")
            
            # Combine and write to file
            combined = pd.concat(processed_chunks, ignore_index=True)
            combined.to_csv(output_file, index=False)
            
            return None
        else:
            # Return combined result
            return pd.concat(processed_chunks, ignore_index=True)
    
    def process_database(
        self,
        query: str,
        connection_string: str,
        process_fn: Callable[[pd.DataFrame], pd.DataFrame],
        output_table: Optional[str] = None,
        id_column: str = 'id',
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Process data from a database in batches
        
        Args:
            query: SQL query to fetch data
            connection_string: Database connection string
            process_fn: Function to process each batch
            output_table: Optional table to write results to
            id_column: Column to use for batch segmentation
            **kwargs: Additional arguments to pass to pd.read_sql
            
        Returns:
            Combined processed DataFrame if output_table is None
        """
        from sqlalchemy import create_engine, text
        
        # Create engine
        engine = create_engine(connection_string)
        
        # Get total row count and range of IDs
        with engine.connect() as conn:
            # Extract table name from query (simplistic approach)
            table_name = query.split('FROM')[1].split('WHERE')[0].strip()
            
            # Get min and max IDs
            min_id_query = f"SELECT MIN({id_column}) FROM {table_name}"
            max_id_query = f"SELECT MAX({id_column}) FROM {table_name}"
            
            min_id = conn.execute(text(min_id_query)).scalar()
            max_id = conn.execute(text(max_id_query)).scalar()
            
            # Get total row count
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            total_rows = conn.execute(text(count_query)).scalar()
        
        self.logger.info(f"Processing {total_rows} rows from database")
        self.logger.info(f"ID range: {min_id} to {max_id}")
        
        # Process in batches based on ID ranges
        results = []
        batch_count = (total_rows + self.batch_size - 1) // self.batch_size
        
        if self.show_progress:
            pbar = tqdm(total=batch_count, desc="Processing batches")
        
        start_time = time.time()
        
        for batch_start in range(min_id, max_id + 1, self.batch_size):
            batch_end = min(batch_start + self.batch_size - 1, max_id)
            
            # Modify query to get only this batch
            batch_query = f"{query} WHERE {id_column} BETWEEN {batch_start} AND {batch_end}"
            
            # Fetch batch
            batch = pd.read_sql(batch_query, engine, **kwargs)
            
            if len(batch) == 0:
                continue
            
            # Process batch
            processed_batch = process_fn(batch)
            
            # Handle result
            if output_table:
                # Write to database
                if_exists = 'replace' if batch_start == min_id else 'append'
                processed_batch.to_sql(output_table, engine, if_exists=if_exists, index=False)
            else:
                # Store for later combination
                results.append(processed_batch)
            
            # Update progress
            if self.show_progress:
                pbar.update(1)
        
        # Close progress bar
        if self.show_progress:
            pbar.close()
        
        # Log completion
        elapsed_time = time.time() - start_time
        self.logger.info(f"Processed {total_rows} rows in {elapsed_time:.2f} seconds")
        
        # Return combined result if not writing to database
        if not output_table and results:
            return pd.concat(results, ignore_index=True)
        
        return None
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        process_fn: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Process a large DataFrame in batches
        
        Args:
            df: Input DataFrame
            process_fn: Function to process each batch
            
        Returns:
            Processed DataFrame
        """
        self.logger.info(f"Processing DataFrame with {len(df)} rows")
        
        # Split into batches
        batch_count = (len(df) + self.batch_size - 1) // self.batch_size
        batches = []
        
        for i in range(batch_count):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(df))
            batches.append((i, df.iloc[start_idx:end_idx], process_fn))
        
        # Process batches
        if self.num_workers > 1:
            # Parallel processing
            results = []
            start_time = time.time()
            
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                # Create progress bar if needed
                if self.show_progress:
                    pbar = tqdm(total=batch_count, desc="Processing batches")
                    
                    # Process batches with progress updates
                    for result in pool.imap_unordered(self._process_chunk, batches):
                        results.append(result)
                        pbar.update(1)
                    
                    pbar.close()
                else:
                    # Process batches without progress bar
                    results = pool.map(self._process_chunk, batches)
            
            # Sort results by batch ID
            results.sort(key=lambda x: x[0])
            processed_batches = [r[1] for r in results]
            
        else:
            # Sequential processing
            processed_batches = []
            start_time = time.time()
            
            if self.show_progress:
                pbar = tqdm(total=batch_count, desc="Processing batches")
            
            for i, batch_df, batch_fn in batches:
                processed_batch = batch_fn(batch_df)
                processed_batches.append(processed_batch)
                
                if self.show_progress:
                    pbar.update(1)
            
            if self.show_progress:
                pbar.close()
        
        # Log completion
        elapsed_time = time.time() - start_time
        self.logger.info(f"Processed {len(df)} rows in {elapsed_time:.2f} seconds")
        
        # Combine and return
        return pd.concat(processed_batches, ignore_index=True)

# Example usage
def demo_batch_processing():
    """Demonstrate batch processing with a simple example"""
    # Create a large sample DataFrame
    rows = 1_000_000
    print(f"Creating sample data with {rows} rows...")
    
    df = pd.DataFrame({
        'id': range(1, rows + 1),
        'value': np.random.randint(1, 100, size=rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], size=rows),
        'text': np.random.choice(['foo', 'bar', 'baz', 'qux'], size=rows)
    })
    
    # Define a process function
    def process_batch(batch_df):
        # Simple transformations
        result = batch_df.copy()
        result['value_squared'] = result['value'] ** 2
        result['category_code'] = result['category'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4})
        
        # Simulate some processing time
        time.sleep(0.01)
        
        return result
    
    # Create a batch processor
    processor = BatchProcessor(
        batch_size=50000,
        num_workers=4,  # Use 4 worker processes
        show_progress=True
    )
    
    # Process the DataFrame
    print("Processing DataFrame in batches...")
    result_
# Example usage
def demo_batch_processing():
    """Demonstrate batch processing with a simple example"""
    # Create a large sample DataFrame
    rows = 1_000_000
    print(f"Creating sample data with {rows} rows...")
    
    df = pd.DataFrame({
        'id': range(1, rows + 1),
        'value': np.random.randint(1, 100, size=rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], size=rows),
        'text': np.random.choice(['foo', 'bar', 'baz', 'qux'], size=rows)
    })
    
    # Define a process function
    def process_batch(batch_df):
        # Simple transformations
        result = batch_df.copy()
        result['value_squared'] = result['value'] ** 2
        result['category_code'] = result['category'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4})
        
        # Simulate some processing time
        time.sleep(0.01)
        
        return result
    
    # Create a batch processor
    processor = BatchProcessor(
        batch_size=50000,
        num_workers=4,  # Use 4 worker processes
        show_progress=True
    )
    
    # Process the DataFrame
    print("Processing DataFrame in batches...")
    result_df = processor.process_dataframe(df, process_batch)
    
    print(f"Processing complete. Result has {len(result_df)} rows and {len(result_df.columns)} columns")
    print(f"Sample of results:\n{result_df.head()}")
    
    # Save to CSV to demonstrate file processing
    csv_path = "sample_data.csv"
    print(f"Saving sample data to {csv_path}...")
    df.to_csv(csv_path, index=False)
    
    # Process the CSV file
    print("Processing CSV file in batches...")
    output_path = "processed_data.csv"
    processor.process_csv(
        csv_path,
        process_batch,
        output_path
    )
    
    print(f"CSV processing complete. Results saved to {output_path}")
    
    # Clean up
    os.remove(csv_path)
    os.remove(output_path)
    
    print("Batch processing demo completed successfully!")

# Advanced batch processing with error handling
def batch_process_with_error_handling(
    input_file: str,
    output_file: str,
    batch_size: int = 10000,
    max_retries: int = 3,
    error_log: str = "errors.json"
):
    """
    Process a large file with error handling and retries
    
    Args:
        input_file: Input CSV file path
        output_file: Output CSV file path
        batch_size: Number of rows per batch
        max_retries: Maximum number of retry attempts for failed batches
        error_log: Path to save error information
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("batch_processing.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("batch_processor")
    
    # Track errors
    errors = []
    
    # Function to process a batch with error handling
    def process_batch_safe(batch_df):
        """Process a batch with error handling and data validation"""
        try:
            # Data validation
            if batch_df.empty:
                logger.warning("Empty batch received")
                return batch_df
            
            # Check for required columns
            required_columns = ['id', 'value', 'category']
            missing_columns = [col for col in required_columns if col not in batch_df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Check for invalid values
            if batch_df['value'].isnull().any():
                # Fill missing values with mean
                batch_df['value'] = batch_df['value'].fillna(batch_df['value'].mean())
                logger.warning("Filled missing values in 'value' column")
            
            # Process the data
            result = batch_df.copy()
            
            # Apply transformations
            result['value_squared'] = result['value'] ** 2
            result['category_code'] = result['category'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4})
            
            # Handle unmapped categories
            if result['category_code'].isnull().any():
                result['category_code'] = result['category_code'].fillna(-1)
                logger.warning("Found unmapped categories")
            
            # Add processing timestamp
            result['processed_at'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            # Record the error
            error_info = {
                'batch_start_id': batch_df['id'].min() if 'id' in batch_df.columns else None,
                'batch_end_id': batch_df['id'].max() if 'id' in batch_df.columns else None,
                'row_count': len(batch_df),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            errors.append(error_info)
            
            logger.error(f"Error processing batch: {str(e)}")
            
            # Return the original batch to allow retries
            return None
    
    # Initialize processor
    processor = BatchProcessor(batch_size=batch_size, show_progress=True)
    
    # Process the file with retries
    logger.info(f"Processing {input_file} with batch size {batch_size}")
    
    try:
        # First attempt
        failed_batches = []
        
        # Custom chunk handler that collects failed batches for retry
        def chunk_handler(chunk):
            result = process_batch_safe(chunk)
            if result is None:
                failed_batches.append(chunk)
                # Return empty DataFrame to skip this batch
                return pd.DataFrame(columns=chunk.columns)
            return result
        
        # Process the file
        processor.process_csv(input_file, chunk_handler, output_file)
        
        # Retry logic for failed batches
        retry_count = 0
        while failed_batches and retry_count < max_retries:
            retry_count += 1
            logger.info(f"Retry attempt {retry_count} for {len(failed_batches)} failed batches")
            
            # Save current failed batches and reset for next retry
            current_failed = failed_batches
            failed_batches = []
            
            # Combine failed batches into a single DataFrame
            retry_df = pd.concat(current_failed, ignore_index=True)
            
            # Process the retry batches
            retry_output = f"retry_{retry_count}.csv"
            retry_results = processor.process_dataframe(retry_df, process_batch_safe)
            
            # Identify successful retries
            if not retry_results.empty:
                # Append successful retries to output
                retry_results.to_csv(output_file, mode='a', header=False, index=False)
                logger.info(f"Successfully processed {len(retry_results)} rows in retry {retry_count}")
        
        # Report final status
        if failed_batches:
            logger.warning(f"Failed to process {sum(len(batch) for batch in failed_batches)} rows after {max_retries} retries")
        
        # Save error log
        if errors:
            with open(error_log, 'w') as f:
                json.dump(errors, f, indent=2)
            logger.info(f"Saved error information to {error_log}")
            
        logger.info(f"Processing completed. Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Fatal error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    # Run batch processing demo
    demo_batch_processing()
## Logging and Monitoring

Setting up proper logging is essential for automated scripts:

```python
import logging
import os
from datetime import datetime
import json
from typing import Dict, Any, Optional, List

class StructuredLogger:
    """A structured logging class that provides consistent, searchable logs"""
    
    def __init__(
        self, 
        app_name: str,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        log_to_console: bool = True,
        include_timestamp: bool = True,
        include_context: bool = True,
        json_format: bool = False
    ):
        """
        Initialize the structured logger
        
        Args:
            app_name: Name of the application
            log_level: Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional path to log file
            log_to_console: Whether to also log to console
            include_timestamp: Whether to include ISO timestamp in records
            include_context: Whether to include context details in records
            json_format: Whether to output logs in JSON format
        """
        self.app_name = app_name
        self.include_timestamp = include_timestamp
        self.include_context = include_context
        self.json_format = json_format
        
        # Map string level to logging level
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        log_level_value = level_map.get(log_level.upper(), logging.INFO)
        
        # Create logger
        self.logger = logging.getLogger(app_name)
        self.logger.setLevel(log_level_value)
        self.logger.handlers = []  # Remove any existing handlers
        
        # Create formatter
        if json_format:
            formatter = logging.Formatter('%(message)s')
        else:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Add file handler if log_file specified
        if log_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level_value)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level_value)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def _format_record(self, level: str, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format a log record based on configuration"""
        if self.json_format:
            record = {
                "level": level,
                "message": message,
                "app": self.app_name
            }
            
            if self.include_timestamp:
                record["timestamp"] = datetime.now().isoformat()
            
            if context and self.include_context:
                record["context"] = context
            
            return json.dumps(record)
        else:
            # For non-JSON format, we'll add context as key-value pairs to the message
            if context and self.include_context:
                context_str = ", ".join(f"{k}={v}" for k, v in context.items())
                return f"{message} [{context_str}]"
            else:
                return message
    
    def debug(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log a debug message"""
        self.logger.debug(self._format_record("DEBUG", message, context))
    
    def info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an info message"""
        self.logger.info(self._format_record("INFO", message, context))
    
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message"""
        self.logger.warning(self._format_record("WARNING", message, context))
    
    def error(self, message: str, context: Optional[Dict[str, Any]] = None, exc_info: bool = False) -> None:
        """Log an error message"""
        self.logger.error(self._format_record("ERROR", message, context), exc_info=exc_info)
    
    def critical(self, message: str, context: Optional[Dict[str, Any]] = None, exc_info: bool = False) -> None:
        """Log a critical message"""
        self.logger.critical(self._format_record("CRITICAL", message, context), exc_info=exc_info)
    
    def set_level(self, level: str) -> None:
        """Change the logging level"""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        log_level = level_map.get(level.upper(), logging.INFO)
        
        self.logger.setLevel(log_level)
        for handler in self.logger.handlers:
            handler.setLevel(log_level)
    
    def add_context_to_all_logs(self, global_context: Dict[str, Any]) -> None:
        """Add context values to all subsequent log messages"""
        # Create a filter to add context to all log records
        class ContextFilter(logging.Filter):
            def __init__(self, context):
                super().__init__()
                self.context = context
            
            def filter(self, record):
                # Add context fields to the record
                for key, value in self.context.items():
                    setattr(record, key, value)
                return True
        
        # Apply the filter to all handlers
        context_filter = ContextFilter(global_context)
        for handler in self.logger.handlers:
            handler.addFilter(context_filter)


class LogAnalyzer:
    """Analyze log files to identify patterns and issues"""
    
    def __init__(self, log_files: List[str], json_format: bool = False):
        """
        Initialize the log analyzer
        
        Args:
            log_files: List of log file paths to analyze
            json_format: Whether logs are in JSON format
        """
        self.log_files = log_files
        self.json_format = json_format
    
    def count_by_level(self) -> Dict[str, int]:
        """Count log entries by level"""
        counts = {"DEBUG": 0, "INFO": 0, "WARNING": 0, "ERROR": 0, "CRITICAL": 0}
        
        for log_file in self.log_files:
            if not os.path.exists(log_file):
                print(f"Log file not found: {log_file}")
                continue
            
            with open(log_file, 'r') as f:
                for line in f:
                    if self.json_format:
                        try:
                            record = json.loads(line)
                            level = record.get("level", "").upper()
                            if level in counts:
                                counts[level] += 1
                        except json.JSONDecodeError:
                            continue
                    else:
                        # Simple parsing for standard log format
                        if " - DEBUG - " in line:
                            counts["DEBUG"] += 1
                        elif " - INFO - " in line:
                            counts["INFO"] += 1
                        elif " - WARNING - " in line:
                            counts["WARNING"] += 1
                        elif " - ERROR - " in line:
                            counts["ERROR"] += 1
                        elif " - CRITICAL - " in line:
                            counts["CRITICAL"] += 1
        
        return counts
    
    def error_analysis(self) -> List[Dict[str, Any]]:
        """Extract and analyze error entries"""
        errors = []
        
        for log_file in self.log_files:
            if not os.path.exists(log_file):
                print(f"Log file not found: {log_file}")
                continue
            
            with open(log_file, 'r') as f:
                for line in f:
                    if self.json_format:
                        try:
                            record = json.loads(line)
                            level = record.get("level", "").upper()
                            if level in ["ERROR", "CRITICAL"]:
                                errors.append(record)
                        except json.JSONDecodeError:
                            continue
                    else:
                        # Simple parsing for standard log format
                        if " - ERROR - " in line or " - CRITICAL - " in line:
                            # Extract timestamp and message
                            parts = line.split(" - ", 3)
                            if len(parts) >= 4:
                                timestamp = parts[0]
                                app = parts[1]
                                level = parts[2]
                                message = parts[3].strip()
                                
                                errors.append({
                                    "timestamp": timestamp,
                                    "app": app,
                                    "level": level,
                                    "message": message
                                })
        
        return errors
    
    def find_patterns(self) -> Dict[str, int]:
        """Find common patterns or messages in the logs"""
        message_counts = {}
        
        for log_file in self.log_files:
            if not os.path.exists(log_file):
                print(f"Log file not found: {log_file}")
                continue
            
            with open(log_file, 'r') as f:
                for line in f:
                    if self.json_format:
                        try:
                            record = json.loads(line)
                            message = record.get("message", "")
                            # Simplify the message by removing specific details
                            pattern = self._extract_pattern(message)
                            message_counts[pattern] = message_counts.get(pattern, 0) + 1
                        except json.JSONDecodeError:
                            continue
                    else:
                        # Extract message part from standard log format
                        parts = line.split(" - ", 3)
                        if len(parts) >= 4:
                            message = parts[3].strip()
                            pattern = self._extract_pattern(message)
                            message_counts[pattern] = message_counts.get(pattern, 0) + 1
        
        # Sort by frequency
        return dict(sorted(message_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _extract_pattern(self, message: str) -> str:
        """
        Extract a pattern from a message by removing specific details
        
        This is a simplified implementation and could be enhanced with regex
        """
        # Replace numbers with #
        pattern = ''.join('#' if c.isdigit() else c for c in message)
        
        # Replace common variable parts
        replacements = [
            (r'id=\S+', 'id=#'),
            (r'path=\S+', 'path=#'),
            (r'file=\S+', 'file=#'),
            (r'user=\S+', 'user=#'),
            (r'ip=\S+', 'ip=#'),
            (r'\d{4}-\d{2}-\d{2}', 'DATE'),
            (r'\d{2}:\d{2}:\d{2}', 'TIME')
        ]
        
        import re
        for pattern_re, replacement in replacements:
            pattern = re.sub(pattern_re, replacement, pattern)
        
        return pattern
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive analysis report"""
        report = {
            "analysis_time": datetime.now().isoformat(),
            "log_files": self.log_files,
            "entry_counts": self.count_by_level(),
            "top_patterns": dict(list(self.find_patterns().items())[:10]),
            "recent_errors": self.error_analysis()[-10:] if self.error_analysis() else []
        }
        
        # Calculate summary statistics
        total_entries = sum(report["entry_counts"].values())
        error_rate = (report["entry_counts"]["ERROR"] + report["entry_counts"]["CRITICAL"]) / total_entries if total_entries > 0 else 0
        
        report["summary"] = {
            "total_entries": total_entries,
            "error_rate": error_rate,
            "error_count": report["entry_counts"]["ERROR"] + report["entry_counts"]["CRITICAL"]
        }
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report


# Example usage
def demo_structured_logging():
    """Demonstrate structured logging with context"""
    # Create a logger for a data processing application
    logger = StructuredLogger(
        app_name="data_processor",
        log_level="INFO",
        log_file="logs/data_processor.log",
        log_to_console=True,
        json_format=True
    )
    
    # Log application startup
    logger.info("Application started", {
        "version": "1.0.0",
        "env": "development"
    })
    
    # Log some processing events
    try:
        # Simulate loading a file
        logger.info("Loading data file", {
            "file": "customers.csv",
            "size_mb": 24.5
        })
        
        # Simulate processing
        logger.info("Processing data", {
            "rows": 10000,
            "columns": 15
        })
        
        # Simulate a warning
        logger.warning("Found missing values", {
            "column": "email",
            "missing_count": 42
        })
        
        # Simulate an error
        x = 1 / 0  # This will raise a ZeroDivisionError
    except Exception as e:
        logger.error(
            f"Processing error: {str(e)}", 
            {
                "error_type": type(e).__name__,
                "stage": "data_transformation"
            },
            exc_info=True
        )
    
    # Log completion
    logger.info("Application completed", {
        "duration_seconds": 5.2,
        "status": "partial_success"
    })
    
    # Analyze the logs
    analyzer = LogAnalyzer(["logs/data_processor.log"], json_format=True)
    report = analyzer.generate_report("logs/analysis_report.json")
    
    print("Log Analysis Report:")
    print(f"Total entries: {report['summary']['total_entries']}")
    print(f"Error rate: {report['summary']['error_rate']:.2%}")
    print("Entry counts by level:")
    for level, count in report["entry_counts"].items():
        print(f"  {level}: {count}")
    
    print("Recent errors:")
    for error in report["recent_errors"]:
        print(f"  {error.get('timestamp', 'N/A')}: {error.get('message', 'N/A')}")

if __name__ == "__main__":
    # Set up log directory
    os.makedirs("logs", exist_ok=True)
    
    # Run the demo
    demo_structured_logging()
```

## Configuration Management

Managing configuration across different environments and systems:

```python
import os
import json
import yaml
import configparser
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging

class ConfigManager:
    """Manage configuration with environment-specific settings"""
    
    def __init__(
        self, 
        app_name: str,
        config_dir: str = "config",
        env_var_prefix: Optional[str] = None
    ):
        """
        Initialize the configuration manager
        
        Args:
            app_name: Name of the application
            config_dir: Directory containing configuration files
            env_var_prefix: Prefix for environment variables
        """
        self.app_name = app_name
        self.config_dir = Path(config_dir)
        self.env_var_prefix = env_var_prefix or app_name.upper()
        
        # Set up logging
        self.logger = logging.getLogger(f"{app_name}.config")
        
        # Initialize configuration
        self.config = {}
        
        # Environment setting - default to development
        self.environment = os.environ.get(f"{self.env_var_prefix}_ENV", "development")
        
        # Create config directory if it doesn't exist
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True)
            self.logger.info(f"Created config directory: {self.config_dir}")
    
    def load_configuration(self) -> Dict[str, Any]:
        """
        Load configuration from various sources in order of precedence:
        1. Default configuration file
        2. Environment-specific configuration file
        3. Local configuration file (not in version control)
        4. Environment variables
        
        Returns:
            The loaded configuration dictionary
        """
        self.logger.info(f"Loading configuration for environment: {self.environment}")
        
        # Start with empty configuration
        self.config = {}
        
        # 1. Load default configuration
        default_config = self._load_config_file("default")
        if default_config:
            self.config.update(default_config)
            self.logger.info("Loaded default configuration")
        
        # 2. Load environment-specific configuration
        env_config = self._load_config_file(self.environment)
        if env_config:
            self._deep_update(self.config, env_config)
            self.logger.info(f"Loaded {self.environment} configuration")
        
        # 3. Load local configuration (not in version control)
        local_config = self._load_config_file("local")
        if local_config:
            self._deep_update(self.config, local_config)
            self.logger.info("Loaded local configuration")
        
        # 4. Override with environment variables
        self._load_from_env_vars()
        
        self.logger.info("Configuration loaded successfully")
        return self.config
    
    def _load_config_file(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a configuration file by name
        
        Args:
            config_name: Name of the configuration (without extension)
            
        Returns:
            Configuration dictionary or None if file not found
        """
        # Try different file extensions in order of preference
        for ext in ["yaml", "yml", "json", "ini"]:
            config_file = self.config_dir / f"{config_name}.{ext}"
            
            if not config_file.exists():
                continue
            
            try:
                if ext in ["yaml", "yml"]:
                    with open(config_file, 'r') as f:
                        return yaml.safe_load(f)
                
                elif ext == "json":
                    with open(config_file, 'r') as f:
                        return json.load(f)
                
                elif ext == "ini":
                    parser = configparser.ConfigParser()
                    parser.read(config_file)
                    
                    # Convert to dictionary
                    result = {}
                    for section in parser.sections():
                        result[section] = dict(parser[section])
                    
                    return result
                
            except Exception as e:
                self.logger.error(f"Error loading {config_file}: {str(e)}")
                return None
        
        return None
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively update a dictionary with another dictionary
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recurse into nested dictionaries
                self._deep_update(target[key], value)
            else:
                # Update or add value
                target[key] = value
    
    def _load_from_env_vars(self) -> None:
        """Load configuration from environment variables"""
        prefix = f"{self.env_var_prefix}_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                
                # Split by double underscore to represent nesting
                parts = config_key.split('__')
                
                # Navigate to the right level in the config
                current = self.config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set the value
                current[parts[-1]] = self._convert_value(value)
    
    def _convert_value(self, value: str) -> Any:
        """
        Convert string values to appropriate Python types
        
        Args:
            value: String value from environment variable
            
        Returns:
            Converted value
        """
        # Try to convert to appropriate type
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        elif value.lower() == 'null' or value.lower() == 'none':
            return None
        
        # Try to convert to number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            # Keep as string
            return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value
        
        Args:
            key: Configuration key, can use dot notation for nested keys
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        parts = key.split('.')
        
        # Navigate through the config
        current = self.config
        for part in parts:
            if part not in current:
                return default
            current = current[part]
        
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value
        
        Args:
            key: Configuration key, can use dot notation for nested keys
            value: Value to set
        """
        parts = key.split('.')
        
        # Navigate through the config
        current = self.config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set
# Navigate through the config
        current = self.config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the value
        current[parts[-1]] = value
    
    def save(self, config_name: Optional[str] = None) -> None:
        """
        Save the current configuration to a file
        
        Args:
            config_name: Name of the configuration file (without extension)
        """
        if config_name is None:
            config_name = self.environment
        
        # Save as YAML (preferred format)
        config_file = self.config_dir / f"{config_name}.yaml"
        
        try:
            with open(config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            self.logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
    
    def validate(self, schema: Dict[str, Any]) -> List[str]:
        """
        Validate the configuration against a schema
        
        Args:
            schema: Schema dictionary with required keys and types
            
        Returns:
            List of validation errors
        """
        errors = []
        
        def validate_item(path, schema_item, config_item):
            # Check type
            if "type" in schema_item:
                expected_type = schema_item["type"]
                
                # Map schema types to Python types
                type_map = {
                    "string": str,
                    "integer": int,
                    "number": (int, float),
                    "boolean": bool,
                    "object": dict,
                    "array": list
                }
                
                python_type = type_map.get(expected_type)
                
                if python_type and not isinstance(config_item, python_type):
                    errors.append(f"{path}: Expected type {expected_type}, got {type(config_item).__name__}")
            
            # Check required keys for objects
            if "properties" in schema_item and isinstance(config_item, dict):
                for key, prop in schema_item["properties"].items():
                    if "required" in prop and prop["required"] and key not in config_item:
                        errors.append(f"{path}: Missing required property '{key}'")
                    
                    if key in config_item:
                        validate_item(f"{path}.{key}", prop, config_item[key])
            
            # Check constraints
            if "minLength" in schema_item and isinstance(config_item, str):
                if len(config_item) < schema_item["minLength"]:
                    errors.append(f"{path}: String length {len(config_item)} is less than minimum {schema_item['minLength']}")
            
            if "maxLength" in schema_item and isinstance(config_item, str):
                if len(config_item) > schema_item["maxLength"]:
                    errors.append(f"{path}: String length {len(config_item)} is greater than maximum {schema_item['maxLength']}")
            
            if "minimum" in schema_item and isinstance(config_item, (int, float)):
                if config_item < schema_item["minimum"]:
                    errors.append(f"{path}: Value {config_item} is less than minimum {schema_item['minimum']}")
            
            if "maximum" in schema_item and isinstance(config_item, (int, float)):
                if config_item > schema_item["maximum"]:
                    errors.append(f"{path}: Value {config_item} is greater than maximum {schema_item['maximum']}")
            
            if "enum" in schema_item and config_item not in schema_item["enum"]:
                errors.append(f"{path}: Value {config_item} is not one of {schema_item['enum']}")
        
        # Start validation at root
        validate_item("root", schema, self.config)
        
        return errors

# Example usage
def demo_config_management():
    """Demonstrate configuration management"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration manager
    config_manager = ConfigManager(
        app_name="data_processor",
        config_dir="config",
        env_var_prefix="DATA_PROC"
    )
    
    # Create example configuration files
    os.makedirs("config", exist_ok=True)
    
    # Default configuration
    default_config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "data_processor",
            "user": "postgres",
            "password": "password"
        },
        "logging": {
            "level": "info",
            "file": "logs/data_processor.log",
            "max_size_mb": 10,
            "backup_count": 3
        },
        "processing": {
            "batch_size": 1000,
            "threads": 4,
            "retry_count": 3
        }
    }
    
    with open("config/default.yaml", 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    # Production configuration
    production_config = {
        "database": {
            "host": "db.example.com",
            "password": "PLACEHOLDER"  # Will be set by environment variable
        },
        "logging": {
            "level": "warning"
        },
        "processing": {
            "batch_size": 5000,
            "threads": 8
        }
    }
    
    with open("config/production.yaml", 'w') as f:
        yaml.dump(production_config, f, default_flow_style=False)
    
    # Load configuration
    config = config_manager.load_configuration()
    
    print("Loaded Configuration:")
    print(f"Environment: {config_manager.environment}")
    print(f"Database host: {config_manager.get('database.host')}")
    print(f"Logging level: {config_manager.get('logging.level')}")
    print(f"Batch size: {config_manager.get('processing.batch_size')}")
    
    # Validate configuration
    schema = {
        "type": "object",
        "properties": {
            "database": {
                "type": "object",
                "required": True,
                "properties": {
                    "host": {"type": "string", "required": True},
                    "port": {"type": "integer", "required": True},
                    "name": {"type": "string", "required": True},
                    "user": {"type": "string", "required": True},
                    "password": {"type": "string", "required": True}
                }
            },
            "logging": {
                "type": "object",
                "required": True,
                "properties": {
                    "level": {
                        "type": "string",
                        "required": True,
                        "enum": ["debug", "info", "warning", "error", "critical"]
                    }
                }
            }
        }
    }
    
    validation_errors = config_manager.validate(schema)
    
    if validation_errors:
        print("\nConfiguration validation errors:")
        for error in validation_errors:
            print(f"- {error}")
    else:
        print("\nConfiguration is valid!")
    
    # Clean up
    import shutil
    shutil.rmtree("config")

if __name__ == "__main__":
    demo_config_management()
## Email and Notification Integration

Adding notifications to your automated scripts:

```python
import smtplib
import os
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import logging
from jinja2 import Template

class NotificationManager:
    """Manage notifications across different channels"""
    
    def __init__(self, app_name: str, config: Dict[str, Any] = None):
        """
        Initialize the notification manager
        
        Args:
            app_name: Name of the application
            config: Configuration dictionary
        """
        self.app_name = app_name
        self.config = config or {}
        
        # Set up logging
        self.logger = logging.getLogger(f"{app_name}.notifications")
        
        # Initialize notification channels
        self.email_sender = EmailSender(self.config.get("email", {}))
        self.slack_sender = SlackSender(self.config.get("slack", {}))
        self.teams_sender = TeamsSender(self.config.get("teams", {}))
    
    def send_notification(
        self,
        subject: str,
        message: str,
        level: str = "info",
        channels: List[str] = None,
        attachments: List[str] = None,
        template: Optional[str] = None,
        template_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """
        Send a notification across specified channels
        
        Args:
            subject: Notification subject/title
            message: Notification message (can be HTML for email)
            level: Notification level (info, warning, error, critical)
            channels: List of channels to use (email, slack, teams)
            attachments: Optional list of file paths to attach
            template: Optional template name to use
            template_data: Optional data to render the template with
            
        Returns:
            Dictionary of channels and their success status
        """
        # Default to all configured channels if none specified
        if channels is None:
            channels = []
            if self.email_sender.is_configured():
                channels.append("email")
            if self.slack_sender.is_configured():
                channels.append("slack")
            if self.teams_sender.is_configured():
                channels.append("teams")
        
        # Check level thresholds for each channel
        allowed_channels = []
        
        for channel in channels:
            channel_config = self.config.get(channel, {})
            threshold = channel_config.get("level_threshold", "info").lower()
            
            # Check if level meets threshold
            if self._level_meets_threshold(level, threshold):
                allowed_channels.append(channel)
        
        # Log notification
        self.logger.info(f"Sending {level} notification: {subject}")
        
        # If using a template, render it
        if template:
            message = self._render_template(template, template_data, message)
        
        # Send to each channel
        results = {}
        
        for channel in allowed_channels:
            try:
                if channel == "email":
                    results["email"] = self.email_sender.send(
                        subject, message, level, attachments
                    )
                elif channel == "slack":
                    results["slack"] = self.slack_sender.send(
                        subject, message, level
                    )
                elif channel == "teams":
                    results["teams"] = self.teams_sender.send(
                        subject, message, level
                    )
                else:
                    self.logger.warning(f"Unknown notification channel: {channel}")
                    results[channel] = False
            except Exception as e:
                self.logger.error(f"Error sending notification to {channel}: {str(e)}")
                results[channel] = False
        
        return results
    
    def _level_meets_threshold(self, level: str, threshold: str) -> bool:
        """Check if a notification level meets a threshold"""
        levels = ["debug", "info", "warning", "error", "critical"]
        
        try:
            level_idx = levels.index(level.lower())
            threshold_idx = levels.index(threshold.lower())
            
            return level_idx >= threshold_idx
        except ValueError:
            # Default to allowing the notification if level is unrecognized
            return True
    
    def _render_template(
        self,
        template_name: str,
        template_data: Dict[str, Any],
        default_message: str
    ) -> str:
        """Render a notification template"""
        try:
            # Look for templates in a templates directory
            template_dir = self.config.get("template_dir", "templates")
            template_path = os.path.join(template_dir, f"{template_name}.html")
            
            if not os.path.exists(template_path):
                self.logger.warning(f"Template not found: {template_path}")
                return default_message
            
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            # Render the template
            template = Template(template_content)
            
            # Add standard context variables
            context = {
                "app_name": self.app_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "level": template_data.get("level", "info")
            }
            context.update(template_data or {})
            
            return template.render(**context)
        
        except Exception as e:
            self.logger.error(f"Error rendering template: {str(e)}")
            return default_message

class EmailSender:
    """Send email notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the email sender
        
        Args:
            config: Email configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("notifications.email")
    
    def is_configured(self) -> bool:
        """Check if email is configured"""
        return bool(
            self.config and
            self.config.get("smtp_server") and
            self.config.get("sender")
        )
    
    def send(
        self,
        subject: str,
        message: str,
        level: str = "info",
        attachments: List[str] = None
    ) -> bool:
        """
        Send an email notification
        
        Args:
            subject: Email subject
            message: Email message (can be HTML)
            level: Notification level
            attachments: Optional list of file paths to attach
            
        Returns:
            True if the email was sent successfully
        """
        if not self.is_configured():
            self.logger.error("Email is not configured")
            return False
        
        try:
            # Get configuration
            smtp_server = self.config["smtp_server"]
            smtp_port = int(self.config.get("smtp_port", 587))
            sender = self.config["sender"]
            recipients = self.config.get("recipients", [])
            username = self.config.get("username")
            password = self.config.get("password")
            use_tls = self.config.get("use_tls", True)
            
            if not recipients:
                self.logger.error("No recipients specified")
                return False
            
            # Create the email
            msg = MIMEMultipart()
            msg["From"] = sender
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = f"[{level.upper()}] {subject}"
            
            # Attach the message
            msg.attach(MIMEText(message, "html"))
            
            # Attach files if provided
            if attachments:
                for attachment_path in attachments:
                    if os.path.exists(attachment_path):
                        with open(attachment_path, "rb") as f:
                            attachment = MIMEApplication(f.read())
                            attachment.add_header(
                                "Content-Disposition",
                                f"attachment; filename={os.path.basename(attachment_path)}"
                            )
                            msg.attach(attachment)
                    else:
                        self.logger.warning(f"Attachment not found: {attachment_path}")
            
            # Connect to the SMTP server
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if use_tls:
                    server.starttls()
                
                # Login if credentials provided
                if username and password:
                    server.login(username, password)
                
                # Send the email
                server.send_message(msg)
            
            self.logger.info(f"Email sent to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email: {str(e)}")
            return False

class SlackSender:
    """Send Slack notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Slack sender
        
        Args:
            config: Slack configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("notifications.slack")
    
    def is_configured(self) -> bool:
        """Check if Slack is configured"""
        return bool(
            self.config and
            (self.config.get("webhook_url") or self.config.get("token"))
        )
    
    def send(
        self,
        subject: str,
        message: str,
        level: str = "info"
    ) -> bool:
        """
        Send a Slack notification
        
        Args:
            subject: Message title
            message: Message content
            level: Notification level
            
        Returns:
            True if the message was sent successfully
        """
        if not self.is_configured():
            self.logger.error("Slack is not configured")
            return False
        
        try:
            # Get configuration
            webhook_url = self.config.get("webhook_url")
            token = self.config.get("token")
            channel = self.config.get("channel")
            username = self.config.get("username", "Notification Bot")
            
            # Set color based on level
            color_map = {
                "info": "#2196F3",      # Blue
                "warning": "#FF9800",   # Orange
                "error": "#F44336",     # Red
                "critical": "#B71C1C"   # Dark Red
            }
            color = color_map.get(level.lower(), "#2196F3")
            
            if webhook_url:
                # Use webhook (simpler)
                payload = {
                    "username": username,
                    "attachments": [
                        {
                            "fallback": subject,
                            "color": color,
                            "title": subject,
                            "text": message,
                            "ts": int(datetime.now().timestamp())
                        }
                    ]
                }
                
                if channel:
                    payload["channel"] = channel
                
                response = requests.post(webhook_url, json=payload)
                response.raise_for_status()
                
            elif token:
                # Use Slack API
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "attachments": [
                        {
                            "fallback": subject,
                            "color": color,
                            "title": subject,
                            "text": message,
                            "ts": int(datetime.now().timestamp())
                        }
                    ]
                }
                
                if channel:
                    payload["channel"] = channel
                
                response = requests.post(
                    "https://slack.com/api/chat.postMessage",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                # Check for Slack API errors
                response_data = response.json()
                if not response_data.get("ok"):
                    error = response_data.get("error", "Unknown error")
                    self.logger.error(f"Slack API error: {error}")
                    return False
            
            self.logger.info("Slack notification sent")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending Slack notification: {str(e)}")
            return False

class TeamsSender:
    """Send Microsoft Teams notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Teams sender
        
        Args:
            config: Teams configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("notifications.teams")
    
    def is_configured(self) -> bool:
        """Check if Teams is configured"""
        return bool(
            self.config and
            self.config.get("webhook_url")
        )
    
    def send(
        self,
        subject: str,
        message: str,
        level: str = "info"
    ) -> bool:
        """
        Send a Teams notification
        
        Args:
            subject: Message title
            message: Message content
            level: Notification level
            
        Returns:
            True if the message was sent successfully
        """
        if not self.is_configured():
            self.logger.error("Teams is not configured")
            return False
        
        try:
            # Get configuration
            webhook_url = self.config["webhook_url"]
            
            # Set color based on level
            color_map = {
                "info": "#2196F3",      # Blue
                "warning": "#FF9800",   # Orange
                "error": "#F44336",     # Red
                "critical": "#B71C1C"   # Dark Red
            }
            theme_color = color_map.get(level.lower(), "#2196F3")
            
            # Create Teams message card
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": theme_color.lstrip('#'),
                "summary": subject,
                "sections": [
                    {
                        "activityTitle": subject,
                        "activitySubtitle": f"Level: {level.upper()}",
                        "text": message
                    }
                ]
            }
            
            # Send the message
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            self.logger.info("Teams notification sent")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending Teams notification: {str(e)}")
            return False

# Example usage
def demo_notifications():
    """Demonstrate notification system"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = {
        "email": {
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "sender": "notifications@example.com",
            "recipients": ["admin@example.com", "manager@example.com"],
            "username": "notifications@example.com",
            "password": "YOUR_PASSWORD",
            "use_tls": True,
            "level_threshold": "warning"  # Only send warnings and above
        },
        "slack": {
            "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
            "channel": "#alerts",
            "username": "Data Monitor",
            "level_threshold": "info"  # Send all notifications
        },
        "teams": {
            "webhook_url": "https://outlook.office.com/webhook/YOUR_WEBHOOK_URL",
            "level_threshold": "error"  # Only send errors and critical alerts
        },
        "template_dir": "templates"
    }
    
    # Create notification manager
    notifier = NotificationManager("data_processor", config)
    
    # Create a simple HTML email template
    os.makedirs("templates", exist_ok=True)
    
    with open("templates/alert.html", "w") as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; }
                .header { background-color: #f0f0f0; padding: 10px; }
                .content { padding: 20px; }
                .footer { font-size: 12px; color: #666; padding: 10px; }
                .{{ level }} { color: {% if level == 'error' or level == 'critical' %}#cc0000{% elif level == 'warning' %}#cc7700{% else %}#007700{% endif %}; }
            </style>
        </head>
        <body>
            <div class="header">
                <h2>{{ app_name }} Alert</h2>
            </div>
            <div class="content">
                <p class="{{ level }}"><strong>{{ level|upper }}</strong>: {{ message }}</p>
                <p>Details:</p>
                <ul>
                    {% for key, value in details.items() %}
                    <li><strong>{{ key }}:</strong> {{ value }}</li>
                    {% endfor %}
                </ul>
            </div>
            <div class="footer">
                Generated at {{ timestamp }}
            </div>
        </body>
        </html>
        """)
    
    # Send notifications
    
    # Info notification
    notifier.send_notification(
        subject="Data Processing Started",
        message="The data processing job has started.",
        level="info",
        channels=["slack"]
    )
    
    # Warning notification
    notifier.send_notification(
        subject="Missing Values Detected",
        message="Some records have missing values in the 'email' column.",
        level="warning",
        channels=["email", "slack"],
        template="alert",
        template_data={
            "message": "Missing values detected in data",
            "details": {
                "column": "email",
                "missing_count": 42,
                "total_records": 1500,
                "percent_missing": "2.8%"
            }
        }
    )
    
    # Error notification (all channels)
    notifier.send_notification(
        subject="Processing Failed",
        message="The data processing job failed due to a database error.",
        level="error",
        template="alert",
        template_data={
            "message": "Data processing job failed",
            "details": {
                "error": "Database connection timeout",
                "job_id": "DP-12345",
                "attempts": 3,
                "affected_tables": "customers, orders"
            }
        }
    )
    
    # Clean up
    import shutil
    shutil.rmtree("templates")

if __name__ == "__main__":
    demo_notifications()
```

## Mini-Project: Data Governance Automation Suite

Putting it all together to build a complete data governance automation tool:

```python
import os
import sys
import json
import yaml
import logging
import argparse
import pandas as pd
import schedule
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path

# Import components from previous sections
# For a real project, these would be in separate modules
from config_manager import ConfigManager
from batch_processor import BatchProcessor
from structured_logger import StructuredLogger
from notification_manager import NotificationManager

class DataGovernanceTool:
    """Integrated tool for data governance automation"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data governance tool
        
        Args:
            config_path: Optional path to configuration file
        """
        # Get environment
        self.env = os.environ.get("DATA_GOV_ENV", "development")
        
        # Set up configuration
        self.config_manager = ConfigManager(
            app_name="data_governance",
            config_dir="config",
            env_var_prefix="DATA_GOV"
        )
        
        # Load configuration
        if config_path:
            self.config_manager.load_from_file(config_path)
        else:
            self.config_manager.load_configuration()
        
        self.config = self.config_manager.config
        
        # Set up logging
        log_config = self.config.get("logging", {})
        self.logger = StructuredLogger(
            app_name="data_governance",
            log_level=log_config.get("level", "INFO"),
            log_file=log_config.get("file"),
            log_to_console=log_config.get("console", True),
            json_format=log_config.get("json_format", False)
        )
        
        # Set up batch processor
        proc_config = self.config.get("processing", {})
        self.batch_processor = BatchProcessor(
            batch_size=proc_config.get("batch_size", 10000),
            num_workers=proc_config.get("num_workers", 1),
            show_progress=proc_config.get("show_progress", True)
        )
        
        # Set up notification manager
        self.notifier = NotificationManager(
            app_name="data_governance",
            config=self.config.get("notifications", {})
        )
        
        # Initialize rule registry
        self.rule_registry = self._load_rules()
        
        self.logger.info(f"Data Governance Tool initialized (env: {self.env})")
    
    def _load_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load data quality rules from configuration"""
        rules_path = self.config.get("rules_path", "rules")
        
        if not os.path.exists(rules_path):
            os.makedirs(rules_path)
            self.logger.info(f"Created rules directory: {rules_path}")
        
        rules = {}
        
        # Load rule files
        for file_name in os.listdir(rules_path):
            if file_name.endswith((".json", ".yaml", ".yml")):
                file_path = os.path.join(rules_path, file_name)
                
                try:
                    if file_name.endswith(".json"):
                        with open(file_path, 'r') as f:
                            rule_set = json.load(f)
                    else:
                        with open(file_path, 'r') as f:
                            rule_set = yaml.safe_load(f)
                    
                    # Validate rule set
                    if "name" not in rule_set:
                        self.logger.warning(f"Missing name in rule set: {file_path}")
                        continue
                    
                    rules[rule_set["name"]] = rule_set
                    self.logger.info(f"Loaded rule set: {rule_set['name']}")
                    
                except Exception as e:
                    self.logger.error(f"Error loading rule file {file_path}: {str(e)}")
        
        return rules
    
    def validate_data(
        self,
        data_source: str,
        rule_sets: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        notify: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate data against quality rules
        
        Args:
            data_source: Path to data file or database connection string
            rule_sets: Optional list of rule sets to apply
            output_
"""
        Validate data against quality rules
        
        Args:
            data_source: Path to data file or database connection string
            rule_sets: Optional list of rule sets to apply
            output_path: Optional path to save validation results
            notify: Whether to send notifications for validation failures
            
        Returns:
            Tuple of (passed, results_dict)
        """
        self.logger.info(f"Validating data from {data_source}")
        
        # Load data
        try:
            if data_source.endswith((".csv", ".xlsx", ".xls")):
                # File source
                if data_source.endswith(".csv"):
                    df = pd.read_csv(data_source)
                else:
                    df = pd.read_excel(data_source)
                
                source_type = "file"
                source_name = os.path.basename(data_source)
            elif "://" in data_source:
                # Database source
                import sqlalchemy
                
                # Extract table name from connection string
                # This is a simplified approach - real implementation would be more robust
                if "table=" in data_source:
                    table = data_source.split("table=")[1].split("&")[0]
                else:
                    table = "data"  # Default table name
                
                engine = sqlalchemy.create_engine(data_source)
                df = pd.read_sql(f"SELECT * FROM {table}", engine)
                
                source_type = "database"
                source_name = table
            else:
                self.logger.error(f"Unsupported data source: {data_source}")
                return False, {"error": "Unsupported data source"}
            
            self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            
            if notify:
                self.notifier.send_notification(
                    subject=f"Data Validation Failed - {source_name}",
                    message=f"Failed to load data from {data_source}: {str(e)}",
                    level="error"
                )
            
            return False, {"error": str(e)}
        
        # Determine which rule sets to apply
        if rule_sets:
            active_rule_sets = [rs for rs in rule_sets if rs in self.rule_registry]
            if len(active_rule_sets) < len(rule_sets):
                missing = set(rule_sets) - set(active_rule_sets)
                self.logger.warning(f"Some rule sets not found: {missing}")
        else:
            # Apply all rule sets
            active_rule_sets = list(self.rule_registry.keys())
        
        if not active_rule_sets:
            self.logger.error("No valid rule sets specified")
            return False, {"error": "No valid rule sets specified"}
        
        # Apply rules
        validation_results = {
            "source": source_name,
            "source_type": source_type,
            "timestamp": datetime.now().isoformat(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "rule_sets": {},
            "passed": True
        }
        
        for rule_set_name in active_rule_sets:
            rule_set = self.rule_registry[rule_set_name]
            
            self.logger.info(f"Applying rule set: {rule_set_name}")
            
            # Process rules in batches if dataset is large
            if len(df) > self.batch_processor.batch_size:
                rule_results = self._apply_rule_set_in_batches(df, rule_set)
            else:
                rule_results = self._apply_rule_set(df, rule_set)
            
            # Add results to validation results
            validation_results["rule_sets"][rule_set_name] = rule_results
            
            # Update overall pass/fail status
            if not rule_results["passed"]:
                validation_results["passed"] = False
        
        # Save results if output path provided
        if output_path:
            try:
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                with open(output_path, 'w') as f:
                    json.dump(validation_results, f, indent=2)
                
                self.logger.info(f"Validation results saved to {output_path}")
                
            except Exception as e:
                self.logger.error(f"Error saving validation results: {str(e)}")
        
        # Send notification if validation failed
        if notify and not validation_results["passed"]:
            # Calculate summary statistics
            rule_count = sum(len(rs["rules"]) for rs in validation_results["rule_sets"].values())
            failed_rule_count = sum(
                sum(1 for r in rs["rules"].values() if not r["passed"])
                for rs in validation_results["rule_sets"].values()
            )
            
            # Create notification message
            subject = f"Data Validation Failed - {source_name}"
            message = f"""
            <h2>Data Validation Failures</h2>
            <p>The data source <strong>{source_name}</strong> has failed validation.</p>
            <p>
                <strong>Summary:</strong><br>
                - {failed_rule_count} of {rule_count} rules failed<br>
                - {len(active_rule_sets)} rule sets applied<br>
                - {len(df)} rows evaluated
            </p>
            """
            
            # Add details for each rule set
            message += "<h3>Rule Set Details:</h3>"
            
            for rule_set_name, rule_results in validation_results["rule_sets"].items():
                if not rule_results["passed"]:
                    failed_rules = {k: v for k, v in rule_results["rules"].items() if not v["passed"]}
                    
                    message += f"""
                    <h4>{rule_set_name}</h4>
                    <ul>
                    """
                    
                    for rule_name, rule_result in failed_rules.items():
                        message += f"""
                        <li>
                            <strong>{rule_name}:</strong> {rule_result.get("message", "Rule failed")}
                            {f"<br><em>Found {rule_result.get('failure_count', 0)} violations</em>" if "failure_count" in rule_result else ""}
                        </li>
                        """
                    
                    message += "</ul>"
            
            # Send the notification
            self.notifier.send_notification(
                subject=subject,
                message=message,
                level="error"
            )
        
        return validation_results["passed"], validation_results
    
    def _apply_rule_set(self, df: pd.DataFrame, rule_set: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a rule set to a DataFrame"""
        results = {
            "name": rule_set["name"],
            "description": rule_set.get("description", ""),
            "rules": {},
            "passed": True
        }
        
        # Apply each rule
        rules = rule_set.get("rules", {})
        
        for rule_name, rule in rules.items():
            rule_type = rule.get("type", "custom")
            columns = rule.get("columns", [])
            
            # Check if required columns exist
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                results["rules"][rule_name] = {
                    "passed": False,
                    "message": f"Missing columns: {missing_columns}",
                    "columns": columns
                }
                results["passed"] = False
                continue
            
            # Apply rule based on type
            if rule_type == "not_null":
                # Check for null values in columns
                rule_results = {
                    "passed": True,
                    "columns": columns,
                    "null_counts": {}
                }
                
                for col in columns:
                    null_count = df[col].isnull().sum()
                    rule_results["null_counts"][col] = null_count
                    
                    if null_count > 0:
                        rule_results["passed"] = False
                
                if not rule_results["passed"]:
                    rule_results["message"] = "Found null values in one or more columns"
                    rule_results["failure_count"] = sum(rule_results["null_counts"].values())
                
                results["rules"][rule_name] = rule_results
                if not rule_results["passed"]:
                    results["passed"] = False
            
            elif rule_type == "unique":
                # Check for duplicates in columns
                rule_results = {
                    "passed": True,
                    "columns": columns,
                    "duplicate_counts": {}
                }
                
                for col in columns:
                    duplicate_count = df[col].duplicated().sum()
                    rule_results["duplicate_counts"][col] = duplicate_count
                    
                    if duplicate_count > 0:
                        rule_results["passed"] = False
                
                if not rule_results["passed"]:
                    rule_results["message"] = "Found duplicate values in one or more columns"
                    rule_results["failure_count"] = sum(rule_results["duplicate_counts"].values())
                
                results["rules"][rule_name] = rule_results
                if not rule_results["passed"]:
                    results["passed"] = False
            
            elif rule_type == "range":
                # Check if values are in specified range
                min_val = rule.get("min")
                max_val = rule.get("max")
                
                rule_results = {
                    "passed": True,
                    "columns": columns,
                    "range": {"min": min_val, "max": max_val},
                    "out_of_range_counts": {}
                }
                
                for col in columns:
                    # Skip non-numeric columns
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        self.logger.warning(f"Column {col} is not numeric, skipping range check")
                        continue
                    
                    # Count values outside range
                    if min_val is not None and max_val is not None:
                        out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                    elif min_val is not None:
                        out_of_range = (df[col] < min_val).sum()
                    elif max_val is not None:
                        out_of_range = (df[col] > max_val).sum()
                    else:
                        out_of_range = 0
                    
                    rule_results["out_of_range_counts"][col] = out_of_range
                    
                    if out_of_range > 0:
                        rule_results["passed"] = False
                
                if not rule_results["passed"]:
                    rule_results["message"] = "Found values outside specified range"
                    rule_results["failure_count"] = sum(rule_results["out_of_range_counts"].values())
                
                results["rules"][rule_name] = rule_results
                if not rule_results["passed"]:
                    results["passed"] = False
            
            elif rule_type == "regex":
                # Check if values match regex pattern
                pattern = rule.get("pattern")
                
                if not pattern:
                    results["rules"][rule_name] = {
                        "passed": False,
                        "message": "No pattern specified for regex rule",
                        "columns": columns
                    }
                    results["passed"] = False
                    continue
                
                rule_results = {
                    "passed": True,
                    "columns": columns,
                    "pattern": pattern,
                    "non_matching_counts": {}
                }
                
                import re
                regex = re.compile(pattern)
                
                for col in columns:
                    # Count non-matching values
                    non_matching = df[col].astype(str).apply(
                        lambda x: regex.match(x) is None
                    ).sum()
                    
                    rule_results["non_matching_counts"][col] = non_matching
                    
                    if non_matching > 0:
                        rule_results["passed"] = False
                
                if not rule_results["passed"]:
                    rule_results["message"] = "Found values not matching pattern"
                    rule_results["failure_count"] = sum(rule_results["non_matching_counts"].values())
                
                results["rules"][rule_name] = rule_results
                if not rule_results["passed"]:
                    results["passed"] = False
            
            elif rule_type == "custom":
                # Apply custom validation function
                expression = rule.get("expression")
                
                if not expression:
                    results["rules"][rule_name] = {
                        "passed": False,
                        "message": "No expression specified for custom rule",
                        "columns": columns
                    }
                    results["passed"] = False
                    continue
                
                try:
                    # Create a safe subset DataFrame with required columns
                    if columns:
                        subset = df[columns].copy()
                    else:
                        subset = df.copy()
                    
                    # Evaluate the expression
                    mask = eval(expression, {"df": subset, "pd": pd, "np": pd.np})
                    
                    if isinstance(mask, pd.Series) and mask.dtype == bool:
                        # Count failures
                        failure_count = (~mask).sum()
                        
                        rule_results = {
                            "passed": failure_count == 0,
                            "columns": columns,
                            "expression": expression,
                            "failure_count": failure_count
                        }
                        
                        if failure_count > 0:
                            rule_results["message"] = f"Custom rule failed with {failure_count} violations"
                        
                        results["rules"][rule_name] = rule_results
                        if not rule_results["passed"]:
                            results["passed"] = False
                    
                    else:
                        # Handle case where expression doesn't produce a boolean mask
                        results["rules"][rule_name] = {
                            "passed": False,
                            "message": "Custom rule expression did not produce a boolean Series",
                            "columns": columns,
                            "expression": expression
                        }
                        results["passed"] = False
                
                except Exception as e:
                    results["rules"][rule_name] = {
                        "passed": False,
                        "message": f"Error evaluating custom rule: {str(e)}",
                        "columns": columns,
                        "expression": expression
                    }
                    results["passed"] = False
            
            else:
                # Unknown rule type
                results["rules"][rule_name] = {
                    "passed": False,
                    "message": f"Unknown rule type: {rule_type}",
                    "columns": columns
                }
                results["passed"] = False
        
        return results
    
    def _apply_rule_set_in_batches(self, df: pd.DataFrame, rule_set: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a rule set to a DataFrame in batches"""
        self.logger.info(f"Processing rule set {rule_set['name']} in batches")
        
        # Initialize results
        results = {
            "name": rule_set["name"],
            "description": rule_set.get("description", ""),
            "rules": {},
            "passed": True,
            "processed_in_batches": True,
            "batch_count": (len(df) + self.batch_processor.batch_size - 1) // self.batch_processor.batch_size
        }
        
        # Initialize rule results
        rules = rule_set.get("rules", {})
        for rule_name in rules:
            results["rules"][rule_name] = {
                "passed": True,
                "processed_in_batches": True
            }
        
        # Define batch processing function
        def process_batch(batch_df):
            # Apply rule set to the batch
            batch_results = self._apply_rule_set(batch_df, rule_set)
            return batch_results
        
        # Process in batches
        all_batch_results = self.batch_processor.process_dataframe(df, process_batch)
        
        # Combine results
        for batch_result in all_batch_results:
            # Update overall pass/fail status
            if not batch_result["passed"]:
                results["passed"] = False
            
            # Combine rule results
            for rule_name, rule_result in batch_result["rules"].items():
                if rule_name not in results["rules"]:
                    results["rules"][rule_name] = rule_result
                else:
                    # Update pass/fail status
                    if not rule_result["passed"]:
                        results["rules"][rule_name]["passed"] = False
                    
                    # Combine counts and messages
                    for key in rule_result:
                        if key in ["null_counts", "duplicate_counts", "out_of_range_counts", "non_matching_counts"]:
                            if key not in results["rules"][rule_name]:
                                results["rules"][rule_name][key] = rule_result[key]
                            else:
                                for col, count in rule_result[key].items():
                                    if col in results["rules"][rule_name][key]:
                                        results["rules"][rule_name][key][col] += count
                                    else:
                                        results["rules"][rule_name][key][col] = count
                    
                    # Update failure count
                    if "failure_count" in rule_result:
                        if "failure_count" not in results["rules"][rule_name]:
                            results["rules"][rule_name]["failure_count"] = rule_result["failure_count"]
                        else:
                            results["rules"][rule_name]["failure_count"] += rule_result["failure_count"]
        
        return results
    
    def schedule_validations(self) -> None:
        """Set up scheduled data validations based on configuration"""
        schedules = self.config.get("schedules", [])
        
        if not schedules:
            self.logger.warning("No validation schedules configured")
            return
        
        # Clear existing schedules
        schedule.clear()
        
        for sched in schedules:
            # Get schedule parameters
            name = sched.get("name", "unnamed")
            data_source = sched.get("data_source")
            rule_sets = sched.get("rule_sets")
            output_dir = sched.get("output_dir", "validation_results")
            schedule_type = sched.get("type", "daily")
            time_str = sched.get("time", "00:00")
            days = sched.get("days")  # For weekly schedules
            notify = sched.get("notify", True)
            
            if not data_source:
                self.logger.error(f"Missing data source in schedule {name}")
                continue
            
            # Create job function
            def run_validation(name=name, data_source=data_source, rule_sets=rule_sets, output_dir=output_dir, notify=notify):
                self.logger.info(f"Running scheduled validation: {name}")
                
                # Generate output path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(output_dir, f"{name}_{timestamp}.json")
                
                # Run validation
                try:
                    passed, results = self.validate_data(
                        data_source=data_source,
                        rule_sets=rule_sets,
                        output_path=output_path,
                        notify=notify
                    )
                    
                    self.logger.info(f"Scheduled validation {name} completed: {'Passed' if passed else 'Failed'}")
                    return passed
                except Exception as e:
                    self.logger.error(f"Error in scheduled validation {name}: {str(e)}")
                    
                    if notify:
                        self.notifier.send_notification(
                            subject=f"Scheduled Validation Error - {name}",
                            message=f"An error occurred during scheduled validation {name}: {str(e)}",
                            level="error"
                        )
                    
                    return False
            
            # Schedule based on type
            if schedule_type == "daily":
                schedule.every().day.at(time_str).do(run_validation)
                self.logger.info(f"Scheduled validation {name} to run daily at {time_str}")
            
            elif schedule_type == "weekly":
                if not days:
                    self.logger.error(f"Missing days in weekly schedule {name}")
                    continue
                
                for day in days:
                    day = day.lower()
                    if day == "monday":
                        schedule.every().monday.at(time_str).do(run_validation)
                    elif day == "tuesday":
                        schedule.every().tuesday.at(time_str).do(run_validation)
                    elif day == "wednesday":
                        schedule.every().wednesday.at(time_str).do(run_validation)
                    elif day == "thursday":
                        schedule.every().thursday.at(time_str).do(run_validation)
                    elif day == "friday":
                        schedule.every().friday.at(time_str).do(run_validation)
                    elif day == "saturday":
                        schedule.every().saturday.at(time_str).do(run_validation)
                    elif day == "sunday":
                        schedule.every().sunday.at(time_str).do(run_validation)
                
                self.logger.info(f"Scheduled validation {name} to run weekly on {', '.join(days)} at {time_str}")
            
            elif schedule_type == "hourly":
                interval = int(sched.get("interval", 1))
                schedule.every(interval).hours.do(run_validation)
                self.logger.info(f"Scheduled validation {name} to run every {interval} hours")
            
            else:
                self.logger.error(f"Unknown schedule type: {schedule_type}")
    
    def run_scheduler(self) -> None:
        """Run the scheduler main loop"""
        self.logger.info("Starting scheduler")
        
        # Set up schedules
        self.schedule_validations()
        
        # Run pending at startup
        schedule.run_pending()
        
        try:
            # Main loop
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"Scheduler error: {str(e)}")

# Command-line interface
def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(
        description="Data Governance Automation Tool",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Setup parser arguments
    parser.add_argument("--config", "-c", help="Path to configuration file")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate data")
    validate_parser.add_argument("data_source", help="Path to data file or database connection string")
    validate_parser.add_argument("--rule-sets", "-r", nargs="+", help="Rule sets to apply")
    validate_parser.add_argument("--output", "-o", help="Path to save validation results")
    validate_parser.add_argument("--no-notify", action="store_true", help="Disable notifications")
    
    # List rules command
    list_parser = subparsers.add_parser("list-rules", help="List available rule sets")
    
    # Run scheduler command
    scheduler_parser = subparsers.add_parser("scheduler", help="Run the validation scheduler")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create tool instance
    tool = DataGovernanceTool(args.config)
    
    # Execute command
    if args.command == "validate":
        passed, results = tool.validate_data(
            data_source=args.data_source,
            rule_sets=args.rule_sets,
            output_path=args.output,
            notify=not args.no_notify
        )
        
        # Print summary
        print(f"Validation {'passed' if passed else 'failed'}")
        
        if not passed:
            failed_rules = {}
            
            for rule_set_name, rule_set_results in results["rule_sets"].items():
                for rule_name, rule_result in rule_set_results["rules"].items():
                    if not rule_result["passed"]:
                        if rule_set_name not in failed_rules:
                            failed_rules[rule_set_name] = []
                        
                        failed_rules[rule_set_name].append({
                            "name": rule_name,
                            "message": rule_result.get("message", "Rule failed")
                        })
            
            for rule_set_name, rules in failed_rules.items():
                print(f"\nRule set: {rule_set_name}")
                for rule in rules:
                    print(f"  - {rule['name']}: {rule['message']}")
        
        # Return exit code based on validation result
        return 0 if passed else 1
    
    elif args.command == "list-rules":
        # List available rule sets
        print("Available rule sets:")
        
        for name, rule_set in tool.rule_registry.items():
            description = rule_set.get("description", "No description")
            rule_count = len(rule_set.get("rules", {}))
            
            print(f"\n{name} ({rule_count} rules)")
            print(f"  {description}")
            
            # Print rule details
            print("  Rules:")
            for rule_name, rule in rule_set.get("rules", {}).items():
                rule_type = rule.get("type", "custom")
                columns = ", ".join(rule.get("columns", []))
                
                print(f"    - {rule_name} ({rule_type})")
                if columns:
                    print(f"      Columns: {columns}")
        
        return 0
    
    elif args.command == "scheduler":
        # Run the scheduler
        tool.run_scheduler()
        return 0
    
    else:
        # Show help if no command specified
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

Example configuration file (`config/default.yaml`):

```yaml
# Data Governance Tool Configuration

# Logging configuration
logging:
  level: info
  file: logs/data_governance.log
  console: true
  json_format: false

# Processing configuration
processing:
  batch_size: 10000
  num_workers: 2
  show_progress: true

# Rules configuration
rules_path: rules

# Notification configuration
notifications:
  email:
    smtp_server: smtp.example.com
    smtp_port: 587
    sender: notifications@example.com
    recipients:
      - data-team@example.com
      - manager@example.com
    username: notifications@example.com
    password: ${EMAIL_PASSWORD}  # Set by environment variable
    use_tls: true
    level_threshold: warning
  
  slack:
    webhook_url: ${SLACK_WEBHOOK_URL}  # Set by environment variable
    channel: "#data-quality"
    username: "Data Governor"
    level_threshold: info

# Scheduled validations
schedules:
  - name: customer_daily
    data_source: data/customers.csv
    rule_sets:
      - customer_rules
    output_dir: validation_results
    type: daily
    time: "08:00"
    notify: true
  
  - name: sales_weekly
    data_source: postgresql://user:pass@localhost:5432/sales?table=transactions
    rule_sets:
      - transaction_rules
      - financial_rules
    output_dir: validation_results
    type: weekly
    days:
      - Monday
      - Thursday
    time: "06:00"
    notify: true
```

Example rule set file (`rules/customer_rules.yaml`):

```yaml
name: customer_rules
description: Validation rules for customer data

rules:
  required_fields:
    type: not_null
    columns:
      - customer_id
      - email
      - name
      - registration_date
  
  unique_identifiers:
    type: unique
    columns:
      - customer_id
      - email
  
  valid_email_format:
    type: regex
    pattern: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$
    columns:
      - email
  
  age_range:
    type: range
    min: 18
    max: 120
    columns:
      - age
  
  complete_profile:
    type: custom
    columns:
      - name
      - email
      - phone
      - address
    expression: df[['name', 'email']].notna().all(axis=1) & (df[['phone', 'address']].notna().sum(axis=1) >= 1)
```
