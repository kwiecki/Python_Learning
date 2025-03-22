# Example 1: pyproject.toml for modern Python packaging

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "datautils"
version = "0.1.0"
description = "Utilities for data processing and analysis"
readme = "README.md"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
license = {text = "MIT"}
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Data Science",
]
keywords = ["data", "processing", "analysis", "utilities"]
dependencies = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "pyyaml>=6.0",
    "click>=8.0.0",
]
requires-python = ">=3.8"

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "black>=22.3.0",
    "isort>=5.10.0",
    "flake8>=4.0.0",
    "mypy>=0.950",
    "sphinx>=4.5.0",
    "sphinx-rtd-theme>=1.0.0",
]
viz = [
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "plotly>=5.7.0",
]

[project.urls]
"Homepage" = "https://github.com/yourusername/datautils"
"Bug Tracker" = "https://github.com/yourusername/datautils/issues"
"Documentation" = "https://datautils.readthedocs.io/"

[project.scripts]
datautils = "datautils.cli:main"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 88
target-version = ["py38"]
include = '\.pyi?$'

[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "--cov=src --cov-report=term-missing"
```

# Example 2: setup.py for backward compatibility

```python
from setuptools import setup

if __name__ == "__main__":
    setup()
```

# Example 3: .gitignore for Python projects

```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# VS Code
.vscode/

# PyCharm
.idea/

# Data files
*.csv
*.xlsx
*.parquet
*.json
*.yaml
*.yml
!tests/fixtures/*.csv
!tests/fixtures/*.json
!tests/fixtures/*.yaml
!tests/fixtures/*.yml

# Logs
logs/
*.log

# Environment variables
.env
.env.*
```

# Example 4: GitHub Actions workflow for CI/CD

```yaml
name: Python Package

on:
  push:
    branches: [ main ]
    tags:
      - 'v*'
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python -m pip install ".[dev]"
    
    - name: Check code formatting with Black
      run: |
        black --check src tests
    
    - name: Sort imports with isort
      run: |
        isort --check-only --profile black src tests
    
    - name: Lint with flake8
      run: |
        flake8 src tests
    
    - name: Type check with mypy
      run: |
        mypy src
    
    - name: Test with pytest
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  publish:
    needs: test
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: |
        python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: ${{ secrets.PYPI_USERNAME }}
        TWINE_PASSWORD: ${{ secrets.PYPI_PASSWORD }}
      run: |
        twine upload dist/*
```

# Example 5: Dockerfile for containerized deployment

```dockerfile
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# Install dependencies
COPY pyproject.toml setup.py ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir build && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -e .

# Final image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy wheels from builder stage
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy the actual application
COPY src/ /app/src/
COPY README.md /app/

# Create non-root user
RUN useradd -m appuser
USER appuser

# Run the application
ENTRYPOINT ["datautils"]
CMD ["--help"]
```

# Example 6: pre-commit configuration for automated checks

```yaml
# .pre-commit-config.yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
    -   id: debug-statements

-   repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
    -   id: black

-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    -   id: isort
        args: ["--profile", "black"]

-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
        additional_dependencies: [flake8-docstrings]

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
    -   id: mypy
        additional_dependencies: [types-all]
```

# Example 7: Sphinx documentation configuration

```python
# docs/conf.py
import os
import sys
from datetime import datetime

# Add package to path
sys.path.insert(0, os.path.abspath('../src'))

# Project information
project = 'DataUtils'
author = 'Your Name'
copyright = f'{datetime.now().year}, {author}'
release = '0.1.0'

# Add extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
]

# Configure autodoc
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
}

# HTML theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
}

# Output options
html_static_path = ['_static']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
```

# Example 8: flake8 configuration

```ini
# .flake8
[flake8]
max-line-length = 88
extend-ignore = E203
exclude = .git,__pycache__,build,dist
per-file-ignores =
    # Allow imported but unused in __init__.py files
    __init__.py: F401
    # Allow missing docstrings in tests
    tests/*: D100,D101,D102,D103
max-complexity = 10
```

# Example 9: requirements.txt alternatives (dev vs. prod)

```
# requirements.txt - For main dependencies
numpy>=1.20.0
pandas>=1.3.0
pyyaml>=6.0
click>=8.0.0
```

```
# requirements-dev.txt - For development dependencies
-r requirements.txt
pytest>=7.0.0
pytest-cov>=3.0.0
black>=22.3.0
isort>=5.10.0
flake8>=4.0.0
mypy>=0.950
sphinx>=4.5.0
sphinx-rtd-theme>=1.0.0
```

# Example 10: Makefile for common project tasks

```makefile
.PHONY: clean clean-test clean-pyc clean-build help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@echo "clean - remove all build, test, coverage and Python artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "lint - check style with flake8"
	@echo "type - check types with mypy"
	@echo "test - run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "servedocs - compile the docs watching for changes"
	@echo "release - package and upload a release"
	@echo "dist - package"
	@echo "install - install the package to the active Python's site-packages"
	@echo "format - format code with black and isort"

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint:
	flake8 src tests

type:
	mypy src

test:
	pytest

test-all:
	tox

coverage:
	pytest --cov=src
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs:
	rm -f docs/api/datautils*.rst
	rm -f docs/api/modules.rst
	sphinx-apidoc -o docs/api src/datautils
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist
	twine upload dist/*

dist: clean
	python -m build
	ls -l dist

install: clean
	pip install -e .

format:
	isort src tests
	black src tests
```

# Example 11: __main__.py for making a package runnable

```python
"""
Command-line entry point for datautils package.

This module allows the package to be run directly:
$ python -m datautils
"""

from datautils.cli import main

if __name__ == "__main__":
    main()
```

# Example 12: setup.cfg for editable install configuration

```ini
[metadata]
name = datautils
version = attr: datautils.__version__
description = Utilities for data processing and analysis
long_description = file: README.md
long_description_content_type = text/markdown
author = Your Name
author_email = your.email@example.com
license = MIT
license_file = LICENSE
classifiers =
    Development Status :: 3 - Alpha
    Intended Audience :: Developers
    License :: OSI Approved :: MIT License
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.8
    Programming Language :: Python :: 3.9
    Programming Language :: Python :: 3.10
    Topic :: Scientific/Engineering :: Data Science
keywords = data, processing, analysis, utilities
project_urls =
    Bug Tracker = https://github.com/yourusername/datautils/issues
    Documentation = https://datautils.readthedocs.io/
    Source Code = https://github.com/yourusername/datautils

[options]
package_dir =
    = src
packages = find:
python_requires = >=3.8
install_requires =
    numpy>=1.20.0
    pandas>=1.3.0
    pyyaml>=6.0
    click>=8.0.0

[options.packages.find]
where = src

[options.extras_require]
dev =
    pytest>=7.0.0
    pytest-cov>=3.0.0
    black>=22.3.0
    isort>=5.10.0
    flake8>=4.0.0
    mypy>=0.950
    sphinx>=4.5.0
    sphinx-rtd-theme>=1.0.0
viz =
    matplotlib>=3.5.
matplotlib>=3.5.0
    seaborn>=0.11.0
    plotly>=5.7.0

[options.entry_points]
console_scripts =
    datautils = datautils.cli:main
```

# Example 13: README.md for your package

```markdown
# DataUtils

[![PyPI version](https://badge.fury.io/py/datautils.svg)](https://badge.fury.io/py/datautils)
[![Python Version](https://img.shields.io/pypi/pyversions/datautils.svg)](https://pypi.org/project/datautils/)
[![Tests](https://github.com/yourusername/datautils/workflows/Tests/badge.svg)](https://github.com/yourusername/datautils/actions?query=workflow%3ATests)
[![Documentation Status](https://readthedocs.org/projects/datautils/badge/?version=latest)](https://datautils.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/yourusername/datautils/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/datautils)

A collection of utilities for data processing and analysis in Python.

## Features

- Data loading from various sources (CSV, Excel, JSON, databases)
- Data cleaning and transformation tools
- Common data validation functions
- Statistical analysis helpers
- Command-line interface for data operations

## Installation

Install the latest release:

```bash
pip install datautils
```

For development installation:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from datautils import cleaning, validation

# Load and clean data
df = pd.read_csv('data.csv')
df = cleaning.clean_column_names(df)
df = cleaning.remove_duplicates(df)

# Validate data
validation_results = validation.validate_data_types(df, {
    'age': 'numeric',
    'name': 'string',
    'email': 'email'
})

if validation_results['valid']:
    print("Data is valid!")
else:
    print("Validation errors:", validation_results['errors'])
```

## Command-line Usage

```bash
# Clean a CSV file
datautils clean data.csv --output clean_data.csv

# Validate a CSV file
datautils validate data.csv --schema schema.json

# Generate statistics
datautils stats data.csv --output stats.json
```

## Documentation

For full documentation, visit [datautils.readthedocs.io](https://datautils.readthedocs.io/).

## Development

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/datautils.git
   cd datautils
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=src
```

### Code Style

This project uses black, isort, and flake8 for code formatting and linting:

```bash
# Format code
black src tests
isort src tests

# Check style
flake8 src tests
```

## License

MIT - See the [LICENSE](LICENSE) file for details.
```

# Example 14: Sample module structure with proper import organization

```python
# src/datautils/__init__.py
"""Data processing utilities for Python."""

from datautils.version import __version__

# Import key components to make them available at package level
from datautils.cleaning import clean_column_names, remove_duplicates
from datautils.validation import validate_data_types, validate_values

__all__ = [
    "__version__",
    "clean_column_names",
    "remove_duplicates",
    "validate_data_types",
    "validate_values",
]
```

```python
# src/datautils/version.py
"""Version information."""

__version__ = "0.1.0"
```

```python
# src/datautils/cleaning.py
"""Data cleaning utilities."""

from typing import Dict, List, Optional, Union

import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by converting to lowercase, replacing spaces with underscores,
    and removing special characters.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned column names
    
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'Column Name': [1, 2], 'Another-Column': [3, 4]})
        >>> clean_df = clean_column_names(df)
        >>> clean_df.columns.tolist()
        ['column_name', 'another_column']
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Clean column names
    result.columns = [
        str(col).lower()
        .replace(' ', '_')
        .replace('-', '_')
        .replace('/', '_')
        .replace('\\', '_')
        .replace('.', '_')
        .replace('(', '')
        .replace(')', '')
        for col in result.columns
    ]
    
    return result


def remove_duplicates(
    df: pd.DataFrame, subset: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Optional list of columns to consider when identifying duplicates
        
    Returns:
        DataFrame with duplicates removed
    
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'a': [1, 2, 1, 3],
        ...     'b': [4, 5, 4, 6]
        ... })
        >>> remove_duplicates(df).shape
        (3, 2)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Remove duplicates
    result = result.drop_duplicates(subset=subset)
    
    return result
```

```python
# src/datautils/validation.py
"""Data validation utilities."""

import re
from typing import Any, Dict, List, Optional, Union

import pandas as pd


def validate_data_types(
    df: pd.DataFrame, type_schema: Dict[str, str]
) -> Dict[str, Any]:
    """
    Validate that columns in the DataFrame match the expected data types.
    
    Args:
        df: Input DataFrame
        type_schema: Dictionary mapping column names to expected types
                    (numeric, string, date, email, etc.)
        
    Returns:
        Dictionary with validation results
    
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35],
        ...     'name': ['Alice', 'Bob', 'Charlie']
        ... })
        >>> validate_data_types(df, {'age': 'numeric', 'name': 'string'})
        {'valid': True, 'errors': []}
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    results = {"valid": True, "errors": []}
    
    for column, expected_type in type_schema.items():
        if column not in df.columns:
            results["valid"] = False
            results["errors"].append(f"Column '{column}' not found in DataFrame")
            continue
        
        # Check data type
        if expected_type == "numeric":
            if not pd.api.types.is_numeric_dtype(df[column]):
                results["valid"] = False
                results["errors"].append(
                    f"Column '{column}' should be numeric but is {df[column].dtype}"
                )
        
        elif expected_type == "string":
            if not pd.api.types.is_string_dtype(df[column]):
                results["valid"] = False
                results["errors"].append(
                    f"Column '{column}' should be string but is {df[column].dtype}"
                )
        
        elif expected_type == "date":
            try:
                pd.to_datetime(df[column])
            except:
                results["valid"] = False
                results["errors"].append(
                    f"Column '{column}' contains values that cannot be converted to dates"
                )
        
        elif expected_type == "email":
            if not pd.api.types.is_string_dtype(df[column]):
                results["valid"] = False
                results["errors"].append(
                    f"Column '{column}' should contain email strings but is {df[column].dtype}"
                )
            else:
                # Check if all non-null values match email pattern
                email_pattern = r"[^@]+@[^@]+\.[^@]+"
                invalid_emails = df[column].dropna().apply(
                    lambda x: not bool(re.match(email_pattern, str(x)))
                )
                if invalid_emails.any():
                    results["valid"] = False
                    invalid_count = invalid_emails.sum()
                    results["errors"].append(
                        f"Column '{column}' contains {invalid_count} invalid email addresses"
                    )
    
    return results


def validate_values(
    df: pd.DataFrame, validation_rules: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate values in DataFrame columns based on specified rules.
    
    Args:
        df: Input DataFrame
        validation_rules: Dictionary mapping column names to validation rules
                         (min, max, allowed_values, etc.)
        
    Returns:
        Dictionary with validation results
    
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35],
        ...     'category': ['A', 'B', 'C']
        ... })
        >>> rules = {
        ...     'age': {'min': 0, 'max': 120},
        ...     'category': {'allowed_values': ['A', 'B', 'C']}
        ... }
        >>> validate_values(df, rules)
        {'valid': True, 'errors': []}
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    results = {"valid": True, "errors": []}
    
    for column, rules in validation_rules.items():
        if column not in df.columns:
            results["valid"] = False
            results["errors"].append(f"Column '{column}' not found in DataFrame")
            continue
        
        # Apply validation rules
        if "min" in rules and pd.api.types.is_numeric_dtype(df[column]):
            min_value = rules["min"]
            below_min = df[column] < min_value
            if below_min.any():
                results["valid"] = False
                count = below_min.sum()
                results["errors"].append(
                    f"Column '{column}' has {count} values below minimum {min_value}"
                )
        
        if "max" in rules and pd.api.types.is_numeric_dtype(df[column]):
            max_value = rules["max"]
            above_max = df[column] > max_value
            if above_max.any():
                results["valid"] = False
                count = above_max.sum()
                results["errors"].append(
                    f"Column '{column}' has {count} values above maximum {max_value}"
                )
        
        if "allowed_values" in rules:
            allowed = rules["allowed_values"]
            not_allowed = ~df[column].isin(allowed)
            if not_allowed.any():
                results["valid"] = False
                count = not_allowed.sum()
                results["errors"].append(
                    f"Column '{column}' has {count} values not in allowed set {allowed}"
                )
    
    return results
```

```python
# src/datautils/cli.py
"""Command-line interface for datautils."""

import json
import sys
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from datautils import cleaning, validation


@click.group()
@click.version_option()
def main():
    """DataUtils - Utilities for data processing and analysis."""
    pass


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path. If not provided, output is written to stdout.",
)
def clean(input_file, output):
    """Clean a data file by standardizing column names and removing duplicates."""
    try:
        # Determine file type from extension
        input_path = Path(input_file)
        if input_path.suffix.lower() == ".csv":
            df = pd.read_csv(input_file)
        elif input_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(input_file)
        else:
            click.echo(f"Unsupported file format: {input_path.suffix}", err=True)
            sys.exit(1)
        
        # Clean the data
        df = cleaning.clean_column_names(df)
        df = cleaning.remove_duplicates(df)
        
        # Output the results
        if output:
            output_path = Path(output)
            if output_path.suffix.lower() == ".csv":
                df.to_csv(output, index=False)
            elif output_path.suffix.lower() in [".xlsx", ".xls"]:
                df.to_excel(output, index=False)
            else:
                click.echo(f"Unsupported output format: {output_path.suffix}", err=True)
                sys.exit(1)
            
            click.echo(f"Cleaned data written to {output}")
        else:
            # Write to stdout as CSV
            click.echo(df.to_csv(index=False))
    
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--schema",
    "-s",
    type=click.Path(exists=True),
    help="JSON schema file for validation.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path for validation results. If not provided, results are written to stdout.",
)
def validate(input_file, schema, output):
    """Validate a data file against a schema."""
    try:
        # Determine file type from extension
        input_path = Path(input_file)
        if input_path.suffix.lower() == ".csv":
            df = pd.read_csv(input_file)
        elif input_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(input_file)
        else:
            click.echo(f"Unsupported file format: {input_path.suffix}", err=True)
            sys.exit(1)
        
        # Load schema
        if schema:
            with open(schema, "r") as f:
                validation_schema = json.load(f)
        else:
            # Use a simple schema based on DataFrame dtypes
            validation_schema = {}
            for column in df.columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    validation_schema[column] = "numeric"
                elif pd.api.types.is_string_dtype(df[column]):
                    validation_schema[column] = "string"
                elif pd.api.types.is_datetime64_dtype(df[column]):
                    validation_schema[column] = "date"
        
        # Validate the data
        results = validation.validate_data_types(df, validation_schema)
        
        # Output the results
        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            click.echo(f"Validation results written to {output}")
        else:
            # Write to stdout as JSON
            click.echo(json.dumps(results, indent=2))
        
        # Exit with error code if validation failed
        if not results["valid"]:
            sys.exit(1)
    
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

# Example 15: tox.ini for multi-environment testing

```ini
[tox]
envlist = py38, py39, py310, lint, docs
isolated_build = True

[testenv]
deps =
    pytest>=7.0.0
    pytest-cov>=3.0.0
commands =
    pytest {posargs:tests} --cov=src

[testenv:lint]
deps =
    black>=22.3.0
    isort>=5.10.0
    flake8>=4.0.0
    mypy>=0.950
commands =
    black --check src tests
    isort --check-only --profile black src tests
    flake8 src tests
    mypy src

[testenv:docs]
deps =
    sphinx>=4.5.0
    sphinx-rtd-theme>=1.0.0
commands =
    sphinx-build -b html docs docs/_build/html

[testenv:build]
deps =
    build
    twine
commands =
    python -m build
    twine check dist/*
```

# Example 16: pytest.ini configuration

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = --cov=src --cov-report=term-missing

markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
```

# Example 17: Type hints with mypy

```python
# src/datautils/statistics.py
"""Statistical utilities with proper type hints."""

from typing import Dict, List, Optional, Union, overload

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


@overload
def calculate_summary_stats(data: pd.Series) -> Dict[str, float]: ...


@overload
def calculate_summary_stats(data: List[float]) -> Dict[str, float]: ...


@overload
def calculate_summary_stats(data: np.ndarray) -> Dict[str, float]: ...


def calculate_summary_stats(
    data: Union[pd.Series, List[float], np.ndarray]
) -> Dict[str, float]:
    """
    Calculate basic summary statistics for the given data.
    
    Args:
        data: Input data as Pandas Series, list of floats, or NumPy array
        
    Returns:
        Dictionary with calculated statistics
        
    Examples:
        >>> calculate_summary_stats([1, 2, 3, 4, 5])
        {'count': 5, 'mean': 3.0, 'std': 1.5811388300841898, 'min': 1, 'q1': 2.0, 'median': 3.0, 'q3': 4.0, 'max': 5}
    """
    # Convert input to numpy array for consistent processing
    if isinstance(data, pd.Series):
        arr = data.dropna().to_numpy()
    elif isinstance(data, list):
        arr = np.array(data)
    else:
        arr = data
    
    # Handle empty arrays
    if len(arr) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q1": np.nan,
            "median": np.nan,
            "q3": np.nan,
            "max": np.nan,
        }
    
    # Calculate statistics
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(np.max(arr)),
    }


def detect_outliers(
    data: ArrayLike, method: str = "iqr", threshold: float = 1.5
) -> np.ndarray:
    """
    Detect outliers in the data using the specified method.
    
    Args:
        data: Input data
        method: Method to use for outlier detection ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean mask indicating outliers
        
    Examples:
        >>> detect_outliers([1, 2, 3, 4, 100])
        array([False, False, False, False,  True])
    """
    arr = np.asarray(data)
    
    if method == "iqr":
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (arr < lower_bound) | (arr > upper_bound)
    
    elif method == "zscore":
        mean = np.mean(arr)
        std = np.std(arr)
        return np.abs((arr - mean) / std) > threshold
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")
```

# Example 18: Test file with fixtures and parametrized tests

```python
# tests/test_cleaning.py
"""Tests for the cleaning module."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from datautils.cleaning import clean_column_names, remove_duplicates


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "Column A": [1, 2, 3],
            "Column-B": [4, 5, 6],
            "Column.C": [7, 8, 9],
        }
    )


@pytest.fixture
def duplicate_df():
    """Create a DataFrame with duplicates for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 1, 4],
            "value": [10, 20, 30, 10, 40],
        }
    )


class TestCleanColumnNames:
    """Tests for the clean_column_names function."""
    
    def test_clean_column_names(self, sample_df):
        """Test that column names are cleaned correctly."""
        # Act
        result = clean_column_names(sample_df)
        
        # Assert
        expected_columns = ["column_a", "column_b", "column_c"]
        assert list(result.columns) == expected_columns
    
    def test_original_unchanged(self, sample_df):
        """Test that the original DataFrame is not modified."""
        # Arrange
        original_columns = list(sample_df.columns)
        
        # Act
        _ = clean_column_names(sample_df)
        
        # Assert
        assert list(sample_df.columns) == original_columns
    
    def test_invalid_input(self):
        """Test that TypeError is raised for invalid input."""
        # Act & Assert
        with pytest.raises(TypeError):
            clean_column_names("not a dataframe")


class TestRemoveDuplicates:
    """Tests for the remove_duplicates function."""
    
    def test_remove_all_duplicates(self, duplicate_df):
        """Test removing all duplicates."""
        # Act
        result = remove_duplicates(duplicate_df)
        
        # Assert
        assert len(result) == 4  # 5 rows - 1 duplicate
        assert result.duplicated().sum() == 0
    
    def test_remove_duplicates_subset(self, duplicate_df):
        """Test removing duplicates based on a subset of columns."""
        # Act
        result = remove_duplicates(duplicate_df, subset=["id"])
        
        # Assert
        assert len(result) == 4
        assert result.duplicated(subset=["id"]).sum() == 0
    
    @pytest.mark.parametrize(
        "input_data,expected_rows",
        [
            (pd.DataFrame({"a": [1, 2, 3]}), 3),  # No duplicates
            (pd.DataFrame({"a": [1, 1, 1]}), 1),  # All duplicates
            (pd.DataFrame({}), 0),  # Empty DataFrame
        ],
    )
    def test_various_inputs(self, input_data, expected_rows):
        """Test with various input data."""
        # Act
        result = remove_duplicates(input_data)
        
        # Assert
        assert len(result) == expected_rows
```

# Example 19: Documentation examples (RST format)

```rst
.. datautils documentation master file

Welcome to DataUtils
===================

DataUtils is a Python package providing utilities for data processing and analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   usage
   api/modules
   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

```rst
.. _installation:

Installation
===========

Basic Installation
-----------------

You can install the latest release of DataUtils from PyPI::

    pip install datautils

This will install the core package with basic dependencies.

Development Installation
----------------------

For development, you can install the package with development dependencies::

    pip install -e ".[dev]"

This will install additional packages needed for development, testing, and documentation.

Optional Dependencies
-------------------

DataUtils has optional dependencies for visualization capabilities::

    pip install "datautils[viz]"

Requirements
-----------

DataUtils requires:

* Python 3.8 or newer
* NumPy 1.20.0 or newer
* pandas 1.3.0 or newer
```

# Example 20: Docker Compose for development environment

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - .:/app
      - ./data:/app/data
    command: bash
    environment:
      - PYTHONPATH=/app/src
    ports:
      - "8888:8888"  # For Jupyter Notebook

  tests:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - .:/app
    command: pytest
    environment:
      - PYTHONPATH=/app/src

  docs:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - .:/app
    ports:
      - "8000:8000"
    command: sphinx-autobuild docs docs/_build/html --host 0.0.0.0 --port 8000
```
