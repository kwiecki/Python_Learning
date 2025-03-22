# Python_Learning Fundamentals for Data Analysts

My structured approach to learning Python for data analysis (created with AI)
Welcome to the Core Python Fundamentals module! This guide is designed specifically for data analysts and data governance professionals who want to add Python to their technical toolkit. Rather than teaching Python in isolation, we'll focus on concepts and examples directly relevant to data analysis work.

## Why Python for Data Analysts?

Python has become an essential tool for data professionals because it allows you to:
- Automate repetitive data tasks
- Clean and transform data more efficiently than spreadsheet tools
- Connect to databases, APIs, and other data sources
- Perform advanced data analysis beyond what's possible in Excel
- Create reproducible data workflows
- Build data quality monitoring tools

## Module Overview

This module covers fundamental Python concepts with practical applications for data work:

1. [Setting Up Your Environment](#setting-up-your-environment)
2. [Python Basics](#python-basics)
3. [Control Flow](#control-flow)
4. [Functions](#functions)
5. [Modules and Packages](#modules-and-packages)
6. [File Operations](#file-operations)
7. [Error Handling](#error-handling)
8. [Mini-Project: Data Validation Tool](#mini-project-data-validation-tool)

## Setting Up Your Environment

### Installing Python

Download and install Python from [python.org](https://www.python.org/downloads/). Be sure to check "Add Python to PATH" during installation.

### Setting Up VS Code

1. Download and install [Visual Studio Code](https://code.visualstudio.com/)
2. Install the Python extension from the marketplace
3. Configure VS Code for Python development:
   - Select your Python interpreter
   - Enable linting for code quality

### Creating a Virtual Environment

Virtual environments allow you to manage dependencies for different projects:

```bash
# Create a virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate

# Install required packages
pip install pandas numpy matplotlib
```

## Python Basics

### Variables and Data Types

Python has several built-in data types particularly useful for data analysis:

```python
# Numeric types
revenue = 15000.50  # float
customer_count = 250  # integer

# Strings for text data
customer_name = "Acme Corporation"
status = 'Active'

# Booleans for flags
is_valid = True
has_completed = False

# None for missing values
contact_date = None
```

#### Practice Exercise: Variable Types
Create variables for different types of data you might encounter in a dataset (customer information, financial data, etc.)

### Data Structures

#### Lists
Lists store ordered collections of items:

```python
# Customer IDs
customer_ids = [1001, 1002, 1003, 1004]

# Access by index (0-based)
first_customer = customer_ids[0]  # 1001

# Add items
customer_ids.append(1005)  # [1001, 1002, 1003, 1004, 1005]

# Slicing
recent_customers = customer_ids[2:5]  # [1003, 1004, 1005]
```

#### Dictionaries
Dictionaries store key-value pairs (similar to lookup tables):

```python
# Customer record
customer = {
    "id": 1001,
    "name": "Acme Corporation",
    "industry": "Manufacturing",
    "active": True,
    "annual_revenue": 1500000
}

# Accessing values
company_name = customer["name"]

# Adding new key-value pairs
customer["contact_email"] = "contact@acme.com"
```

#### Practice Exercise: Customer Records
Create a list of dictionaries representing customer records with multiple fields.

## Control Flow

### Conditional Statements

```python
# Data validation example
def validate_customer_record(customer):
    if "id" not in customer:
        return "ERROR: Missing customer ID"
    
    if customer["annual_revenue"] < 0:
        return "ERROR: Revenue cannot be negative"
        
    if not isinstance(customer["name"], str):
        return "ERROR: Name must be text"
    
    return "Valid"
```

### Loops

```python
# Processing multiple records
customers = [
    {"id": 1001, "name": "Acme Corp", "annual_revenue": 1500000},
    {"id": 1002, "name": "TechSolutions", "annual_revenue": 2750000},
    {"id": 1003, "name": "Global Services", "annual_revenue": -5000}  # Error!
]

# For loop to validate all customers
for customer in customers:
    result = validate_customer_record(customer)
    if result != "Valid":
        print(f"Customer {customer['id']}: {result}")
```

#### Practice Exercise: Data Validation Loop
Write a loop that checks for various data quality issues in a list of records.

## Functions

Functions make code reusable and organized:

```python
def calculate_data_quality_score(dataset, required_fields):
    """
    Calculate a simple data quality score based on completeness.
    
    Args:
        dataset: List of dictionaries containing records
        required_fields: List of field names that should be present and non-empty
        
    Returns:
        Float between 0-100 representing quality percentage
    """
    total_fields = len(dataset) * len(required_fields)
    missing_fields = 0
    
    for record in dataset:
        for field in required_fields:
            if field not in record or record[field] == "" or record[field] is None:
                missing_fields += 1
    
    if total_fields == 0:
        return 0
    
    completeness = (total_fields - missing_fields) / total_fields
    return round(completeness * 100, 2)
```

#### Practice Exercise: Custom Functions
Write a function that calculates the percentage of duplicate values in a dataset.

## Modules and Packages

Python's power comes from its ecosystem of libraries:

```python
# Standard library modules
import csv
import datetime
import os

# Third-party packages
import pandas as pd
import numpy as np

# Using modules
today = datetime.datetime.now()
file_exists = os.path.exists("customer_data.csv")
```

Important packages for data analysts:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Data visualization
- **requests**: API calls
- **sqlalchemy**: Database connections

## File Operations

Working with CSV files (common in data analysis):

```python
import csv

# Reading a CSV file
def read_customer_data(filename):
    customers = []
    try:
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                customers.append(row)
        return customers
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return []

# Writing to a CSV file
def write_validated_data(valid_customers, output_filename):
    if not valid_customers:
        return
        
    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = valid_customers[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for customer in valid_customers:
            writer.writerow(customer)
```

#### Practice Exercise: File Processing
Create a script that reads a CSV file, performs simple data cleansing, and writes the results to a new file.

## Error Handling

Proper error handling is crucial for robust data pipelines:

```python
def process_data_file(filename):
    try:
        with open(filename, 'r') as file:
            data = file.read()
            # Process data...
            return data
    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist")
        return None
    except PermissionError:
        print(f"Error: No permission to read {filename}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
```

#### Practice Exercise: Robust Data Processing
Enhance a data loading function with appropriate error handling for different scenarios.

## Mini-Project: Data Validation Tool

Let's combine everything we've learned to create a simple data validation tool:

```python
import csv
import os
import datetime

def validate_customer_data(input_file, output_file, error_log):
    """
    Validate customer data from a CSV file and output valid records
    to a new file while logging errors.
    """
    valid_records = []
    error_records = []
    
    try:
        # Read input file
        with open(input_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row_num, row in enumerate(reader, start=2):  # Start at 2 to account for header
                errors = []
                
                # Check required fields
                if not row.get('customer_id'):
                    errors.append("Missing customer ID")
                
                if not row.get('company_name'):
                    errors.append("Missing company name")
                
                # Data type validation
                try:
                    if row.get('annual_revenue'):
                        revenue = float(row['annual_revenue'])
                        if revenue < 0:
                            errors.append("Revenue cannot be negative")
                except ValueError:
                    errors.append("Revenue must be a number")
                
                # Date format validation
                if row.get('last_contact_date'):
                    try:
                        datetime.datetime.strptime(row['last_contact_date'], '%Y-%m-%d')
                    except ValueError:
                        errors.append("Invalid date format (must be YYYY-MM-DD)")
                
                # Record status
                if errors:
                    error_records.append({
                        'row': row_num,
                        'customer_id': row.get('customer_id', 'MISSING'),
                        'errors': '; '.join(errors)
                    })
                else:
                    valid_records.append(row)
        
        # Write valid records to output file
        if valid_records:
            with open(output_file, 'w', newline='') as csvfile:
                fieldnames = valid_records[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in valid_records:
                    writer.writerow(record)
        
        # Write error log
        with open(error_log, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['row', 'customer_id', 'errors'])
            writer.writeheader()
            for error in error_records:
                writer.writerow(error)
        
        return len(valid_records), len(error_records)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0, 0

# Example usage
if __name__ == "__main__":
    valid_count, error_count = validate_customer_data(
        'customer_data.csv',
        'valid_customers.csv',
        'data_errors.csv'
    )
    
    print(f"Processed customer data:")
    print(f"- Valid records: {valid_count}")
    print(f"- Records with errors: {error_count}")
```

## Next Steps

Once you've mastered these fundamentals, you'll be ready to move on to more advanced topics:

1. Data analysis with pandas
2. Working with SQL databases in Python
3. Data visualization
4. Building automated data quality reports

## Resources

- [Python Documentation](https://docs.python.org/3/)
- [Real Python Tutorials](https://realpython.com/)
- [Python for Data Analysis (Book by Wes McKinney)](https://wesmckinney.com/book/)

## Exercises and Projects

For additional practice, try these exercises:

1. Create a script that detects duplicate records in a CSV file
2. Build a command-line tool to validate a specific data format
3. Write a program that generates a data quality report for a CSV file
4. Create a simple ETL pipeline that extracts data from one format, transforms it, and loads it into another

## Contributing

If you've found this guide helpful, consider contributing:
- Add new examples relevant to data governance and quality
- Share additional exercises
- Suggest improvements or corrections

Happy coding!
