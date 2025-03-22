# Data Quality & Validation for Data Professionals

Welcome to the Data Quality & Validation module! This guide focuses on implementing robust data quality checks and validation processes - core skills for data governance and analytics professionals.

## Why Data Quality & Validation Matter

Data quality is the foundation of effective data governance because:
- Poor quality data leads to incorrect analysis and flawed decisions
- Data validation ensures business rules and standards are enforced
- Quality issues detected early save significant time and resources
- Systematic validation creates trustworthy data assets
- Well-documented quality processes support compliance requirements
- Automated quality checks enable scalable data governance

## Module Overview

This module covers key data quality and validation techniques:

1. [Data Profiling Techniques](#data-profiling-techniques)
2. [Validation Frameworks](#validation-frameworks)
3. [Rule-based Cleansing](#rule-based-cleansing)
4. [Standardization Methods](#standardization-methods)
5. [Data Type Validation](#data-type-validation)
6. [Business Rule Implementation](#business-rule-implementation)
7. [Duplicate Detection](#duplicate-detection)
8. [Data Quality Metrics and Reporting](#data-quality-metrics-and-reporting)
9. [Mini-Project: Data Quality Pipeline](#mini-project-data-quality-pipeline)

## Data Profiling Techniques

Understanding your data through statistical analysis:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def profile_dataset(df, output_file=None):
    """
    Generate a comprehensive profile of a dataset
    
    Args:
        df: Pandas DataFrame to profile
        output_file: Optional file path to save the report as HTML
    
    Returns:
        A profile report dictionary
    """
    profile = {}
    
    # Basic DataFrame information
    profile['shape'] = df.shape
    profile['columns'] = list(df.columns)
    profile['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Missing values analysis
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df) * 100).round(2)
    profile['missing_values'] = {
        'counts': missing_values.to_dict(),
        'percentages': missing_percentage.to_dict()
    }
    
    # Duplicate rows
    duplicate_count = df.duplicated().sum()
    profile['duplicates'] = {
        'count': int(duplicate_count),
        'percentage': round((duplicate_count / len(df) * 100), 2)
    }
    
    # Statistical summary for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    profile['numeric_stats'] = {}
    
    for col in numeric_cols:
        profile['numeric_stats'][col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'std': float(df[col].std()),
            'skew': float(df[col].skew()),
            'kurtosis': float(stats.kurtosis(df[col].dropna()))
        }
        
        # Detect outliers using IQR method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
        
        profile['numeric_stats'][col]['outliers'] = {
            'count': int(outliers),
            'percentage': round((outliers / df[col].count() * 100), 2)
        }
    
    # Categorical columns analysis
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    profile['categorical_stats'] = {}
    
    for col in cat_cols:
        value_counts = df[col].value_counts().head(10).to_dict()  # Top 10 values
        unique_count = df[col].nunique()
        
        profile['categorical_stats'][col] = {
            'unique_count': unique_count,
            'unique_percentage': round((unique_count / len(df) * 100), 2),
            'top_values': value_counts,
            'sample_values': df[col].dropna().sample(min(5, df[col].count())).tolist()
        }
    
    # Date columns analysis
    date_cols = []
    for col in df.columns:
        # Try to convert to datetime
        try:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                pd.to_datetime(df[col], errors='raise')
            date_cols.append(col)
        except:
            continue
    
    profile['date_stats'] = {}
    for col in date_cols:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            date_series = pd.to_datetime(df[col])
        else:
            date_series = df[col]
        
        profile['date_stats'][col] = {
            'min_date': date_series.min().strftime('%Y-%m-%d'),
            'max_date': date_series.max().strftime('%Y-%m-%d'),
            'range_days': (date_series.max() - date_series.min()).days,
            'unique_count': date_series.nunique(),
            'weekday_distribution': date_series.dt.day_name().value_counts().to_dict()
        }
    
    # Generate HTML report if output file specified
    if output_file:
        generate_html_profile(df, profile, output_file)
    
    return profile

def generate_html_profile(df, profile, output_file):
    """Create an HTML profile report"""
    import base64
    from io import BytesIO
    
    html = f"""
    <html>
    <head>
        <title>Data Profile Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .missing-high {{ background-color: #ffcdd2; }}
            .missing-medium {{ background-color: #fff9c4; }}
            .missing-low {{ background-color: #c8e6c9; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .chart {{ margin: 10px; max-width: 400px; }}
        </style>
    </head>
    <body>
        <h1>Data Profile Report</h1>
        <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Overview</h2>
        <p>Rows: {profile['shape'][0]}, Columns: {profile['shape'][1]}</p>
        
        <h3>Data Types</h3>
        <table>
            <tr>
                <th>Column</th>
                <th>Data Type</th>
            </tr>
    """
    
    for col, dtype in profile['dtypes'].items():
        html += f"""
            <tr>
                <td>{col}</td>
                <td>{dtype}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h3>Missing Values</h3>
        <table>
            <tr>
                <th>Column</th>
                <th>Missing Count</th>
                <th>Missing Percentage</th>
            </tr>
    """
    
    for col in profile['columns']:
        count = profile['missing_values']['counts'].get(col, 0)
        percentage = profile['missing_values']['percentages'].get(col, 0)
        
        if percentage > 20:
            css_class = "missing-high"
        elif percentage > 5:
            css_class = "missing-medium"
        else:
            css_class = "missing-low"
            
        html += f"""
            <tr class="{css_class}">
                <td>{col}</td>
                <td>{count}</td>
                <td>{percentage}%</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h3>Duplicates</h3>
        <p>Duplicate rows: {count} ({percentage}%)</p>
        
        <h2>Numeric Columns</h2>
    """.format(count=profile['duplicates']['count'], 
               percentage=profile['duplicates']['percentage'])
    
    # Add numeric statistics
    for col, stats in profile['numeric_stats'].items():
        # Create a histogram
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        
        # Convert plot to base64 for embedding
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        html += f"""
            <h3>{col}</h3>
            <div class="container">
                <div>
                    <table>
                        <tr><th>Statistic</th><th>Value</th></tr>
                        <tr><td>Min</td><td>{stats['min']}</td></tr>
                        <tr><td>Max</td><td>{stats['max']}</td></tr>
                        <tr><td>Mean</td><td>{stats['mean']}</td></tr>
                        <tr><td>Median</td><td>{stats['median']}</td></tr>
                        <tr><td>Standard Deviation</td><td>{stats['std']}</td></tr>
                        <tr><td>Skewness</td><td>{stats['skew']}</td></tr>
                        <tr><td>Kurtosis</td><td>{stats['kurtosis']}</td></tr>
                        <tr><td>Outliers</td><td>{stats['outliers']['count']} ({stats['outliers']['percentage']}%)</td></tr>
                    </table>
                </div>
                <div class="chart">
                    <img src="data:image/png;base64,{img_str}" width="400">
                </div>
            </div>
        """
    
    # Add categorical statistics
    html += """
        <h2>Categorical Columns</h2>
    """
    
    for col, stats in profile['categorical_stats'].items():
        # Create a bar chart for top values
        plt.figure(figsize=(8, 4))
        values = list(stats['top_values'].keys())
        counts = list(stats['top_values'].values())
        
        if len(values) > 0:
            plt.barh(values, counts)
            plt.title(f'Top Values in {col}')
            plt.tight_layout()
            
            # Convert plot to base64 for embedding
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            html += f"""
                <h3>{col}</h3>
                <div class="container">
                    <div>
                        <table>
                            <tr><th>Statistic</th><th>Value</th></tr>
                            <tr><td>Unique Values</td><td>{stats['unique_count']}</td></tr>
                            <tr><td>Unique Percentage</td><td>{stats['unique_percentage']}%</td></tr>
                            <tr><td>Sample Values</td><td>{', '.join(map(str, stats['sample_values']))}</td></tr>
                        </table>
                    </div>
                    <div class="chart">
                        <img src="data:image/png;base64,{img_str}" width="400">
                    </div>
                </div>
            """
    
    # Add date statistics
    if profile['date_stats']:
        html += """
            <h2>Date Columns</h2>
        """
        
        for col, stats in profile['date_stats'].items():
            html += f"""
                <h3>{col}</h3>
                <table>
                    <tr><th>Statistic</th><th>Value</th></tr>
                    <tr><td>Minimum Date</td><td>{stats['min_date']}</td></tr>
                    <tr><td>Maximum Date</td><td>{stats['max_date']}</td></tr>
                    <tr><td>Range (days)</td><td>{stats['range_days']}</td></tr>
                    <tr><td>Unique Dates</td><td>{stats['unique_count']}</td></tr>
                </table>
                
                <h4>Weekday Distribution</h4>
                <table>
                    <tr><th>Weekday</th><th>Count</th></tr>
            """
            
            for day, count in stats['weekday_distribution'].items():
                html += f"""
                    <tr><td>{day}</td><td>{count}</td></tr>
                """
            
            html += """
                </table>
            """
    
    html += """
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Profile report saved to {output_file}")

# Function to identify potential data quality issues
def identify_quality_issues(df):
    """
    Identify potential data quality issues in a DataFrame
    
    Args:
        df: Pandas DataFrame to analyze
        
    Returns:
        A dictionary of identified issues
    """
    issues = {
        'completeness': [],
        'validity': [],
        'consistency': [],
        'uniqueness': [],
        'accuracy': []
    }
    
    # Check completeness (missing values)
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    for col, pct in missing_pct.items():
        if pct > 0:
            severity = 'low' if pct < 5 else ('medium' if pct < 20 else 'high')
            issues['completeness'].append({
                'column': col,
                'issue': f"{pct}% missing values",
                'severity': severity
            })
    
    # Check validity (data types and ranges)
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        # Check for extreme values
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (3 * IQR)  # More extreme than typical outlier bound
        upper_bound = Q3 + (3 * IQR)
        
        extreme_values = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
        if extreme_values > 0:
            pct = round((extreme_values / df[col].count() * 100), 2)
            severity = 'low' if pct < 1 else ('medium' if pct < 5 else 'high')
            issues['validity'].append({
                'column': col,
                'issue': f"{extreme_values} extreme values ({pct}%)",
                'severity': severity
            })
        
        # Check for invalid values based on column name
        if 'age' in col.lower():
            invalid_ages = df[(df[col] < 0) | (df[col] > 120)][col].count()
            if invalid_ages > 0:
                pct = round((invalid_ages / df[col].count() * 100), 2)
                issues['validity'].append({
                    'column': col,
                    'issue': f"{invalid_ages} invalid ages ({pct}%)",
                    'severity': 'high'
                })
        
        elif any(term in col.lower() for term in ['price', 'cost', 'revenue', 'sales']):
            negative_values = df[df[col] < 0][col].count()
            if negative_values > 0:
                pct = round((negative_values / df[col].count() * 100), 2)
                issues['validity'].append({
                    'column': col,
                    'issue': f"{negative_values} negative values ({pct}%)",
                    'severity': 'medium'
                })
    
    # Check consistency (data patterns)
    # Example: Date columns should be in a consistent range
    date_cols = []
    for col in df.columns:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                pd.to_datetime(df[col], errors='raise')
            date_cols.append(col)
        except:
            continue
    
    if len(date_cols) > 1:
        date_ranges = {}
        for col in date_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                date_series = pd.to_datetime(df[col])
            else:
                date_series = df[col]
            date_ranges[col] = (date_series.min(), date_series.max())
        
        # Check for inconsistent date ranges
        range_years = {col: (dates[1].year - dates[0].year) for col, dates in date_ranges.items()}
        avg_years = sum(range_years.values()) / len(range_years)
        
        for col, years in range_years.items():
            if abs(years - avg_years) > 5:  # More than 5 years different from average
                issues['consistency'].append({
                    'column': col,
                    'issue': f"Date range ({years} years) differs significantly from other date columns",
                    'severity': 'medium'
                })
    
    # Check uniqueness (duplicate values where unexpected)
    id_cols = [col for col in df.columns if col.lower().endswith('id') or 'code' in col.lower()]
    for col in id_cols:
        duplicate_count = df[col].duplicated().sum()
        if duplicate_count > 0:
            pct = round((duplicate_count / df[col].count() * 100), 2)
            issues['uniqueness'].append({
                'column': col,
                'issue': f"{duplicate_count} duplicate values ({pct}%)",
                'severity': 'high'
            })
    
    # Check duplicate rows
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        pct = round((duplicate_rows / len(df) * 100), 2)
        severity = 'low' if pct < 1 else ('medium' if pct < 5 else 'high')
        issues['uniqueness'].append({
            'column': 'entire_row',
            'issue': f"{duplicate_rows} duplicate rows ({pct}%)",
            'severity': severity
        })
    
    # Check accuracy (patterns that suggest data entry errors)
    # Example: Check for common typo patterns in text columns
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        # Check for multiple consecutive spaces (likely error)
        multi_space = df[df[col].str.contains('  ', na=False)][col].count()
        if multi_space > 0:
            pct = round((multi_space / df[col].count() * 100), 2)
            issues['accuracy'].append({
                'column': col,
                'issue': f"{multi_space} values with multiple spaces ({pct}%)",
                'severity': 'low'
            })
        
        # Check for mixed case when most values are not
        if df[col].str.isupper().mean() > 0.8:  # Mostly uppercase
            mixed_case = df[~df[col].str.isupper() & ~df[col].isnull()][col].count()
            if mixed_case > 0:
                pct = round((mixed_case / df[col].count() * 100), 2)
                issues['accuracy'].append({
                    'column': col,
                    'issue': f"{mixed_case} values not in uppercase ({pct}%)",
                    'severity': 'low'
                })
        elif df[col].str.islower().mean() > 0.8:  # Mostly lowercase
            mixed_case = df[~df[col].str.islower() & ~df[col].isnull()][col].count()
            if mixed_case > 0:
                pct = round((mixed_case / df[col].count() * 100), 2)
                issues['accuracy'].append({
                    'column': col,
                    'issue': f"{mixed_case} values not in lowercase ({pct}%)",
                    'severity': 'low'
                })
    
    return issues

# Create correlation heatmap to detect relationships
def plot_correlation_heatmap(df, figsize=(12, 10)):
    """
    Create a correlation heatmap for numeric columns
    
    Args:
        df: Pandas DataFrame
        figsize: Figure size as a tuple of (width, height)
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Compute correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Plot the heatmap
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        cmap=cmap,
        vmax=1.0, 
        vmin=-1.0,
        center=0,
        square=True, 
        linewidths=.5,
        cbar_kws={"shrink": .5},
        annot=True
    )
    
    plt.title('Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.show()
```

## Validation Frameworks

Building reusable validation frameworks:

```python
import pandas as pd
import re
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Union, Pattern

@dataclass
class ValidationRule:
    """A rule for validating data"""
    name: str
    description: str
    column: str
    validation_fn: Callable
    severity: str = 'medium'  # 'low', 'medium', 'high'
    active: bool = True
    dependencies: List[str] = field(default_factory=list)

@dataclass
class ValidationResult:
    """Result of a validation check"""
    rule_name: str
    column: str
    is_valid: bool
    invalid_count: int
    invalid_percentage: float
    invalid_examples: List[Any]
    message: str
    severity: str
    details: Dict[str, Any] = field(default_factory=dict)

class DataValidator:
    """Data validation framework for pandas DataFrames"""
    
    def __init__(self):
        self.rules = {}
        self.last_results = {}
        self._register_common_rules()
    
    def _register_common_rules(self):
        """Register commonly used validation rules"""
        
        # Completeness rules
        self.add_rule(
            ValidationRule(
                name="no_missing_values",
                description="Check for missing values",
                column="",  # Will be specified when applied
                validation_fn=lambda df, col: df[col].notnull(),
                severity="medium"
            )
        )
        
        # Format rules
        self.add_rule(
            ValidationRule(
                name="is_numeric",
                description="Check if values are numeric",
                column="",
                validation_fn=lambda df, col: pd.to_numeric(df[col], errors='coerce').notnull(),
                severity="high"
            )
        )
        
        self.add_rule(
            ValidationRule(
                name="is_date",
                description="Check if values are valid dates",
                column="",
                validation_fn=lambda df, col: pd.to_datetime(df[col], errors='coerce').notnull(),
                severity="high"
            )
        )
        
        self.add_rule(
            ValidationRule(
                name="is_email",
                description="Check if values are valid email addresses",
                column="",
                validation_fn=lambda df, col: df[col].astype(str).str.match(
                    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                ),
                severity="medium"
            )
        )
        
        # Range rules
        self.add_rule(
            ValidationRule(
                name="in_range",
                description="Check if values are within specified range",
                column="",
                # Additional parameters will be passed when applied
                validation_fn=lambda df, col, min_val=None, max_val=None: (
                    (df[col] >= min_val if min_val is not None else True) &
                    (df[col] <= max_val if max_val is not None else True)
                ),
                severity="medium"
            )
        )
        
        # Uniqueness rules
        self.add_rule(
            ValidationRule(
                name="is_unique",
                description="Check if values are unique",
                column="",
                validation_fn=lambda df, col: ~df[col].duplicated(),
                severity="high"
            )
        )
        
        # Pattern rules
        self.add_rule(
            ValidationRule(
                name="matches_pattern",
                description="Check if values match a regex pattern",
                column="",
                # Pattern will be passed when applied
                validation_fn=lambda df, col, pattern: df[col].astype(str).str.match(pattern),
                severity="medium"
            )
        )
        
        # Categorical rules
        self.add_rule(
            ValidationRule(
                name="in_allowed_values",
                description="Check if values are in a list of allowed values",
                column="",
                # Allowed values will be passed when applied
                validation_fn=lambda df, col, allowed_values: df[col].isin(allowed_values),
                severity="high"
            )
        )
    
    def add_rule(self, rule):
        """Add a validation rule to the registry"""
        self.rules[rule.name] = rule
    
    def apply_rule(self, df, rule_name, column, **kwargs):
        """
        Apply a single validation rule to a DataFrame column
        
        Args:
            df: Pandas DataFrame
            rule_name: Name of the rule to apply
            column: Column to validate
            **kwargs: Additional parameters for the rule's validation function
            
        Returns:
            ValidationResult object
        """
        if rule_name not in self.rules:
            raise ValueError(f"Rule '{rule_name}' not found")
        
        rule = self.rules[rule_name]
        rule_with_column = ValidationRule(
            name=rule.name,
            description=rule.description,
            column=column,
            validation_fn=rule.validation_fn,
            severity=rule.severity,
            active=rule.active,
            dependencies=rule.dependencies
        )
        
        # Check if column exists
        if column not in df.columns:
            return ValidationResult(
                rule_name=rule_name,
                column=column,
                is_valid=False,
                invalid_count=len(df),
                invalid_percentage=100.0,
                invalid_examples=[],
                message=f"Column '{column}' not found in DataFrame",
                severity="high"
            )
        
        # Apply the validation function
        try:
            valid_mask = rule.validation_fn(df, column, **kwargs)
            
            # Count invalid rows
            invalid_mask = ~valid_mask
            invalid_count = invalid_mask.sum()
            invalid_percentage = (invalid_count / len(df) * 100) if len(df) > 0 else 0.0
            
            # Get examples of invalid values
            invalid_df = df[invalid_mask]
            invalid_examples = invalid_df[column].head(5).tolist()
            
            is_valid = invalid_count == 0
            
            # Generate appropriate message
            if is_valid:
                message = f"Validation '{rule_name}' passed for column '{column}'"
            else:
                message = (f"Validation '{rule_name}' failed for column '{column}': "
                           f"{invalid_count} invalid values ({invalid_percentage:.2f}%)")
            
            result = ValidationResult(
                rule_name=rule_name,
                column=column,
                is_valid=is_valid,
                invalid_count=invalid_count,
                invalid_percentage=invalid_percentage,
                invalid_examples=invalid_examples,
                message=message,
                severity=rule.severity,
                details={
                    'params': kwargs,
                    'total_rows': len(df)
                }
            )
            
            # Store the result
            self.last_results[(rule_name, column)] = result
            
            return result
            
        except Exception as e:
            # Handle errors in validation function
            return ValidationResult(
                rule_name=rule_name,
                column=column,
                is_valid=False,
                invalid_count=len(df),
                invalid_percentage=100.0,
                invalid_examples=[],
                message=f"Error applying rule '{rule_name}': {str(e)}",
                severity="high"
            )
    
    def validate_dataframe(self, df, validation_plan):
        """
        Validate a DataFrame with multiple rules
        
        Args:
            df: Pandas DataFrame to validate
            validation_plan: List of dictionaries specifying validation rules
                Each dict should have keys: rule_name, column, and optional parameters
                
        Returns:
            Dictionary with validation results and summary
        """
        results = []
        
        for rule_config in validation_plan:
            rule_name = rule_config.get('rule_name')
            column = rule_config.get('column')
            
            # Extract other parameters
            params = {k: v for k, v in rule_config.items() 
                     if k not in ['rule_name', 'column']}
            
            # Apply the rule
            result = self.apply_rule(df, rule_name, column, **params)
            results.append(result)
        
        # Generate summary
        passed_count = sum(1 for r in results if r.is_valid)
        total_count = len(results)
        pass_percentage = (passed_count / total_count * 100) if
# Generate summary
        passed_count = sum(1 for r in results if r.is_valid)
        total_count = len(results)
        pass_percentage = (passed_count / total_count * 100) if total_count > 0 else 0.0
        
        high_severity_failures = [r for r in results if not r.is_valid and r.severity == 'high']
        medium_severity_failures = [r for r in results if not r.is_valid and r.severity == 'medium']
        low_severity_failures = [r for r in results if not r.is_valid and r.severity == 'low']
        
        summary = {
            'passed_rules': passed_count,
            'total_rules': total_count,
            'pass_percentage': pass_percentage,
            'high_severity_failures': len(high_severity_failures),
            'medium_severity_failures': len(medium_severity_failures),
            'low_severity_failures': len(low_severity_failures),
            'validation_time': datetime.datetime.now().isoformat()
        }
        
        return {
            'results': results,
            'summary': summary
        }
    
    def generate_validation_report(self, validation_results, output_file=None):
        """
        Generate an HTML validation report
        
        Args:
            validation_results: Results from validate_dataframe
            output_file: Optional file path to save the report as HTML
        """
        summary = validation_results['summary']
        results = validation_results['results']
        
        # Start building HTML
        html = f"""
        <html>
        <head>
            <title>Data Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .valid {{ background-color: #c8e6c9; }}
                .invalid {{ background-color: #ffcdd2; }}
                .warning {{ background-color: #fff9c4; }}
                .severity-high {{ border-left: 5px solid #f44336; }}
                .severity-medium {{ border-left: 5px solid #ff9800; }}
                .severity-low {{ border-left: 5px solid #4caf50; }}
                .summary {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
                .summary-box {{ padding: 15px; border-radius: 5px; width: 30%; text-align: center; }}
                .pass-rate {{ background-color: #e3f2fd; }}
                .failures {{ background-color: #ffebee; }}
                .info {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>Data Validation Report</h1>
            <p>Generated: {summary['validation_time']}</p>
            
            <div class="summary">
                <div class="summary-box pass-rate">
                    <h2>Pass Rate</h2>
                    <p style="font-size: 24px;">{summary['pass_percentage']:.1f}%</p>
                    <p>{summary['passed_rules']} of {summary['total_rules']} rules passed</p>
                </div>
                
                <div class="summary-box failures">
                    <h2>Failures by Severity</h2>
                    <p><strong>High:</strong> {summary['high_severity_failures']}</p>
                    <p><strong>Medium:</strong> {summary['medium_severity_failures']}</p>
                    <p><strong>Low:</strong> {summary['low_severity_failures']}</p>
                </div>
                
                <div class="summary-box info">
                    <h2>Overall Status</h2>
                    <p style="font-size: 18px; font-weight: bold;">
        """
        
        # Determine overall status
        if summary['high_severity_failures'] > 0:
            html += "<span style='color: #f44336;'>FAILED</span>"
        elif summary['medium_severity_failures'] > 0:
            html += "<span style='color: #ff9800;'>WARNING</span>"
        else:
            html += "<span style='color: #4caf50;'>PASSED</span>"
        
        html += """
                    </p>
                </div>
            </div>
            
            <h2>Validation Results</h2>
            <table>
                <tr>
                    <th>Rule</th>
                    <th>Column</th>
                    <th>Status</th>
                    <th>Invalid Count</th>
                    <th>Invalid %</th>
                    <th>Examples</th>
                    <th>Message</th>
                </tr>
        """
        
        # Add a row for each validation result
        for result in sorted(results, key=lambda r: (not r.is_valid, r.severity, r.column)):
            status_class = "valid" if result.is_valid else "invalid"
            severity_class = f"severity-{result.severity}"
            
            html += f"""
                <tr class="{status_class} {severity_class}">
                    <td>{result.rule_name}</td>
                    <td>{result.column}</td>
                    <td>{"PASS" if result.is_valid else "FAIL"}</td>
                    <td>{result.invalid_count}</td>
                    <td>{result.invalid_percentage:.2f}%</td>
                    <td>{", ".join(map(str, result.invalid_examples))}</td>
                    <td>{result.message}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html)
            print(f"Validation report saved to {output_file}")
        
        return html
```

## Rule-based Cleansing

Implementing rule-based data cleansing:

```python
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Union, Callable, Optional, Any
from dataclasses import dataclass

@dataclass
class CleansingRule:
    """A rule for cleansing data"""
    name: str
    description: str
    condition_fn: Callable  # Function that identifies rows/values to cleanse
    transform_fn: Callable  # Function that applies the cleansing
    columns: List[str]
    active: bool = True

class DataCleanser:
    """Rule-based data cleansing framework"""
    
    def __init__(self):
        self.rules = {}
        self.cleansing_log = []
        self._register_common_rules()
    
    def _register_common_rules(self):
        """Register commonly used cleansing rules"""
        
        # Remove leading/trailing whitespace
        self.add_rule(
            CleansingRule(
                name="trim_whitespace",
                description="Remove leading and trailing whitespace",
                columns=[],  # Will be specified when applied
                condition_fn=lambda df, col: df[col].astype(str).str.match(r'^\s.*|.*\s$'),
                transform_fn=lambda series: series.astype(str).str.strip()
            )
        )
        
        # Convert to upper case
        self.add_rule(
            CleansingRule(
                name="to_uppercase",
                description="Convert text to upper case",
                columns=[],
                condition_fn=lambda df, col: ~df[col].astype(str).str.isupper() & (df[col].notnull()),
                transform_fn=lambda series: series.astype(str).str.upper()
            )
        )
        
        # Convert to lower case
        self.add_rule(
            CleansingRule(
                name="to_lowercase",
                description="Convert text to lower case",
                columns=[],
                condition_fn=lambda df, col: ~df[col].astype(str).str.islower() & (df[col].notnull()),
                transform_fn=lambda series: series.astype(str).str.lower()
            )
        )
        
        # Convert to title case
        self.add_rule(
            CleansingRule(
                name="to_titlecase",
                description="Convert text to title case",
                columns=[],
                condition_fn=lambda df, col: df[col].notnull(),
                transform_fn=lambda series: series.astype(str).str.title()
            )
        )
        
        # Replace multiple spaces with single space
        self.add_rule(
            CleansingRule(
                name="normalize_spaces",
                description="Replace multiple spaces with a single space",
                columns=[],
                condition_fn=lambda df, col: df[col].astype(str).str.contains('  ', na=False),
                transform_fn=lambda series: series.astype(str).str.replace(r'\s+', ' ', regex=True)
            )
        )
        
        # Fill missing values with a default
        self.add_rule(
            CleansingRule(
                name="fill_missing",
                description="Fill missing values with a default value",
                columns=[],
                condition_fn=lambda df, col: df[col].isnull(),
                # Default value will be specified when applied
                transform_fn=lambda series, default_value: series.fillna(default_value)
            )
        )
        
        # Remove special characters
        self.add_rule(
            CleansingRule(
                name="remove_special_chars",
                description="Remove special characters",
                columns=[],
                condition_fn=lambda df, col: df[col].astype(str).str.contains(r'[^\w\s]', na=False),
                # Pattern will be specified when applied
                transform_fn=lambda series, pattern=r'[^\w\s]': 
                    series.astype(str).str.replace(pattern, '', regex=True)
            )
        )
        
        # Standardize phone numbers
        self.add_rule(
            CleansingRule(
                name="standardize_phone",
                description="Standardize phone number format",
                columns=[],
                condition_fn=lambda df, col: df[col].notnull(),
                transform_fn=lambda series: series.apply(self._standardize_phone)
            )
        )
        
        # Fix date format
        self.add_rule(
            CleansingRule(
                name="standardize_date",
                description="Standardize date format",
                columns=[],
                condition_fn=lambda df, col: df[col].notnull(),
                # Format will be specified when applied
                transform_fn=lambda series, output_format='%Y-%m-%d': 
                    pd.to_datetime(series, errors='coerce').dt.strftime(output_format)
            )
        )
        
        # Round numeric values
        self.add_rule(
            CleansingRule(
                name="round_numbers",
                description="Round numeric values to specified decimals",
                columns=[],
                condition_fn=lambda df, col: df[col].notnull(),
                # Decimals will be specified when applied
                transform_fn=lambda series, decimals=2: series.round(decimals)
            )
        )
        
        # Replace values using mapping
        self.add_rule(
            CleansingRule(
                name="map_values",
                description="Replace values using a mapping dictionary",
                columns=[],
                condition_fn=lambda df, col, mapping: df[col].isin(mapping.keys()),
                # Mapping will be specified when applied
                transform_fn=lambda series, mapping: series.map(mapping).fillna(series)
            )
        )
    
    def _standardize_phone(self, phone):
        """Helper function to standardize phone numbers"""
        if not phone or pd.isna(phone):
            return phone
        
        # Convert to string if not already
        phone_str = str(phone)
        
        # Extract digits only
        digits = ''.join(c for c in phone_str if c.isdigit())
        
        # Format based on number of digits
        if len(digits) == 10:
            return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"({digits[1:4]}) {digits[4:7]}-{digits[7:11]}"
        else:
            return phone_str  # Return original if can't standardize
    
    def add_rule(self, rule):
        """Add a cleansing rule to the registry"""
        self.rules[rule.name] = rule
    
    def apply_rule(self, df, rule_name, columns, **kwargs):
        """
        Apply a single cleansing rule to a DataFrame
        
        Args:
            df: Pandas DataFrame to cleanse
            rule_name: Name of the rule to apply
            columns: Columns to apply the rule to
            **kwargs: Additional parameters for rule condition and transform functions
            
        Returns:
            Cleansed DataFrame and log entry
        """
        if rule_name not in self.rules:
            raise ValueError(f"Rule '{rule_name}' not found")
        
        rule = self.rules[rule_name]
        
        # Make a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Ensure columns is a list
        if isinstance(columns, str):
            columns = [columns]
        
        # Track modifications
        modifications = {}
        
        # Apply rule to each column
        for col in columns:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found, skipping")
                continue
            
            # Identify rows to cleanse
            try:
                condition = rule.condition_fn(df, col, **kwargs)
                rows_to_cleanse = df[condition]
                
                if len(rows_to_cleanse) == 0:
                    # No rows match the condition, skip this column
                    continue
                
                # Apply transformation to those rows
                original_values = rows_to_cleanse[col].copy()
                
                # Apply the transform function
                transformed_values = rule.transform_fn(original_values, **kwargs)
                
                # Update the result DataFrame
                result_df.loc[condition, col] = transformed_values
                
                # Track modifications
                modifications[col] = {
                    'rows_changed': len(rows_to_cleanse),
                    'percentage_changed': (len(rows_to_cleanse) / len(df) * 100),
                    'sample_changes': []
                }
                
                # Store some sample changes
                for i, (idx, row) in enumerate(rows_to_cleanse.iterrows()):
                    if i >= 5:  # Limit to 5 examples
                        break
                    
                    modifications[col]['sample_changes'].append({
                        'row_index': idx,
                        'before': str(row[col]),
                        'after': str(transformed_values.iloc[i])
                    })
                
            except Exception as e:
                print(f"Error applying rule '{rule_name}' to column '{col}': {str(e)}")
                continue
        
        # Create log entry
        log_entry = {
            'rule_name': rule_name,
            'columns': columns,
            'parameters': kwargs,
            'modifications': modifications,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        self.cleansing_log.append(log_entry)
        
        return result_df, log_entry
    
    def cleanse_dataframe(self, df, cleansing_plan):
        """
        Cleanse a DataFrame with multiple rules
        
        Args:
            df: Pandas DataFrame to cleanse
            cleansing_plan: List of dictionaries specifying cleansing rules
                Each dict should have keys: rule_name, columns, and optional parameters
                
        Returns:
            Cleansed DataFrame and full log
        """
        result_df = df.copy()
        
        for rule_config in cleansing_plan:
            rule_name = rule_config.get('rule_name')
            columns = rule_config.get('columns')
            
            # Extract other parameters
            params = {k: v for k, v in rule_config.items() 
                     if k not in ['rule_name', 'columns']}
            
            # Apply the rule
            result_df, _ = self.apply_rule(result_df, rule_name, columns, **params)
        
        return result_df, self.cleansing_log
    
    def generate_cleansing_report(self, cleansing_log, output_file=None):
        """
        Generate an HTML report of data cleansing activities
        
        Args:
            cleansing_log: Log of cleansing operations (from cleanse_dataframe)
            output_file: Optional file path to save the report as HTML
        """
        # Calculate total changes
        total_rows_changed = 0
        total_changes = 0
        affected_columns = set()
        
        for entry in cleansing_log:
            for col, details in entry['modifications'].items():
                total_rows_changed += details['rows_changed']
                total_changes += 1
                affected_columns.add(col)
        
        # Build HTML report
        html = f"""
        <html>
        <head>
            <title>Data Cleansing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
                .summary-box {{ padding: 15px; border-radius: 5px; width: 30%; text-align: center; }}
                .rules-applied {{ background-color: #e3f2fd; }}
                .rows-changed {{ background-color: #f5f5f5; }}
                .columns-affected {{ background-color: #e8f5e9; }}
                details {{ margin-bottom: 10px; }}
                summary {{ cursor: pointer; padding: 10px; background-color: #f2f2f2; }}
                .changes {{ background-color: #f9f9f9; padding: 10px; }}
            </style>
        </head>
        <body>
            <h1>Data Cleansing Report</h1>
            <p>Generated: {datetime.datetime.now().isoformat()}</p>
            
            <div class="summary">
                <div class="summary-box rules-applied">
                    <h2>Rules Applied</h2>
                    <p style="font-size: 24px;">{len(cleansing_log)}</p>
                </div>
                
                <div class="summary-box rows-changed">
                    <h2>Cell Changes</h2>
                    <p style="font-size: 24px;">{total_rows_changed}</p>
                </div>
                
                <div class="summary-box columns-affected">
                    <h2>Columns Affected</h2>
                    <p style="font-size: 24px;">{len(affected_columns)}</p>
                </div>
            </div>
            
            <h2>Cleansing Operations</h2>
        """
        
        # Add details for each cleansing operation
        for entry in cleansing_log:
            rule_name = entry['rule_name']
            columns = ", ".join(entry['columns'])
            params = ", ".join(f"{k}={v}" for k, v in entry['parameters'].items())
            
            html += f"""
            <details>
                <summary>
                    <strong>{rule_name}</strong> - Applied to {columns} {f"with parameters: {params}" if params else ""}
                </summary>
                <div class="changes">
                    <table>
                        <tr>
                            <th>Column</th>
                            <th>Rows Changed</th>
                            <th>% Changed</th>
                            <th>Sample Changes (Before → After)</th>
                        </tr>
            """
            
            # Add details for each modified column
            for col, details in entry['modifications'].items():
                html += f"""
                        <tr>
                            <td>{col}</td>
                            <td>{details['rows_changed']}</td>
                            <td>{details['percentage_changed']:.2f}%</td>
                            <td>
                """
                
                # Add sample changes
                for sample in details['sample_changes']:
                    html += f"""
                                <div>"{sample['before']}" → "{sample['after']}"</div>
                    """
                
                html += """
                            </td>
                        </tr>
                """
            
            html += """
                    </table>
                </div>
            </details>
            """
        
        html += """
        </body>
        </html>
        """
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html)
            print(f"Cleansing report saved to {output_file}")
        
        return html
```

## Standardization Methods

Methods for standardizing and normalizing data:

```python
def standardize_names(df, name_columns, method='split'):
    """
    Standardize name formats across columns
    
    Args:
        df: Pandas DataFrame
        name_columns: List of column names containing names to standardize
        method: Method to use - 'split', 'combined', or 'extract'
        
    Returns:
        DataFrame with standardized names
    """
    result_df = df.copy()
    
    if method == 'split':
        # Split full names into first and last name columns
        for col in name_columns:
            if col not in df.columns:
                continue
                
            # Create new columns for first and last name
            first_col = f"{col}_first"
            last_col = f"{col}_last"
            
            # Split the names
            result_df[first_col] = df[col].str.split(r'\s+', expand=True)[0]
            result_df[last_col] = df[col].str.split(r'\s+', expand=True)[1]
            
            # Handle middle names/initials - capture everything after first name and before last
            full_parts = df[col].str.split(r'\s+', expand=True)
            if full_parts.shape[1] > 2:  # Has middle name
                middle_col = f"{col}_middle"
                result_df[middle_col] = full_parts[1]
    
    elif method == 'combined':
        # Ensure first and last names are combined consistently
        for col in name_columns:
            if col.endswith('_first') or col.endswith('_fname'):
                # Find corresponding last name column
                base_col = col.replace('_first', '').replace('_fname', '')
                last_col = next((c for c in df.columns if 
                                c == f"{base_col}_last" or 
                                c == f"{base_col}_lname"), None)
                
                if last_col:
                    # Create combined name column
                    full_col = f"{base_col}_full"
                    result_df[full_col] = df[col] + ' ' + df[last_col]
    
    elif method == 'extract':
        # Extract first and last names from inconsistent formats
        for col in name_columns:
            if col not in df.columns:
                continue
                
            # Handle "Last, First" format
            mask_comma = df[col].str.contains(',', na=False)
            
            # Create new columns
            first_col = f"{col}_first"
            last_col = f"{col}_last"
            
            # Initialize columns
            result_df[first_col] = None
            result_df[last_col] = None
            
            # Extract from "Last, First" format
            if mask_comma.any():
                comma_parts = df.loc[mask_comma, col].str.split(',', expand=True)
                result_df.loc[mask_comma, last_col] = comma_parts[0].str.strip()
                result_df.loc[mask_comma, first_col] = comma_parts[1].str.strip()
            
            # Extract from "First Last" format
            mask_space = ~mask_comma & df[col].str.contains(r'\s+', na=False)
            if mask_space.any():
                space_parts = df.loc[mask_space, col].str.split(r'\s+', expand=True)
                result_df.loc[mask_space, first_col] = space_parts[0]
                result_df.loc[mask_space, last_col] = space_parts[1]
    
    # Standardize case format
    for col in result_df.columns:
        if any(suffix in col for suffix in 
               ['_first', '_last', '_middle', '_full', '_fname', '_lname']):
            # Apply title case to names
            result_df[col] = result_df[col].str.title()
    
    return result_df

def standardize_addresses(df, address_columns):
    """
    Standardize address formats
    
    Args:
        df: Pandas DataFrame
        address_columns: Dictionary mapping column names to address components
            e.g., {'address': 'street', 'city': 'city', 'state': 'state', 'zip': 'postal_code'}
        
    Returns:
        DataFrame with standardized addresses
    """
    result_df = df.copy()
    
    # Standardize street addresses
    if 'address' in address_columns:
        col = address_columns['address']
        if col in df.columns:
            # Convert to title case but keep certain words lowercase
            result_df[col] = df[col].str.title()
            
            # Standardize abbreviations
            abbrev_map = {
                r'\bSt\b': 'Street',
                r'\bRd\b': 'Road',
                r'\bAve\b': 'Avenue',
                r'\bBlvd\b': 'Boulevard',
                r'\bLn\b': 'Lane',
                r'\bCt\b': 'Court',
                r'\bDr\b': 'Drive',
                r'\bHwy\b': 'Highway',
                r'\bPl\b': 'Place',
                r'\bSte\b': 'Suite',
                r'\bApt\b': 'Apartment',
                r'\bN\b': 'North',
                r'\bS\b': 'South',
                r'\bE\b': 'East',
                r'\bW\b': 'West',
                r'\bNE\b': 'Northeast',
                r'\bNW\b': 'Northwest',
                r'\bSE\b': 'Southeast',
                r'\bSW\b': 'Southwest',
            }
            
            for pattern, replacement in abbrev_map.items():
                result_df[col] = result_df[col].str.replace(pattern, replacement, regex=True)
    
    # Standardize state codes
    if 'state' in address_columns:
        col = address_columns['state']
        if col in df.columns:
            # Map state names to standardized two-letter codes
            state_map = {
                'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
                'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
                'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
                'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
                'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
                'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
                'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
                'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY',
                'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
                'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
                'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
                'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV',
                'WISCONSIN': 'WI', 'WYOMING': 'WY', 'DISTRICT OF COLUMBIA': 'DC'
            }
            
            # Normalize to uppercase for matching
            uppercase_states = df[col].str.upper()
            
            # Find full state names and replace with codes
            for state_name, code in state_map.items():
                mask = uppercase_states == state_name
                result_df.loc[mask, col] = code
            
            # Ensure all state codes are uppercase
            mask_two_letter = df[col].str.len() == 2
            result_df.loc[mask_two_letter, col] = df.loc[mask_two_letter, col].str.upper()
    
    # Standardize ZIP/postal codes
    if 'zip' in address_columns:
        col = address_columns['zip']
        if col in df.columns:
            # Convert to string
            result_df[col] = df[col].astype(str)
            
            # Fill missing leading zeros
            five_digit_mask = result_df[col].str.len() == 5
            result_df.loc[~five_digit_mask, col] = result_df.loc[~five_digit_mask, col].str.zfill(5)
            
            # Handle 9-digit ZIP codes (ZIP+4)
            zip4_mask = result_df[col].str.contains('-', na=False)
            
            # Ensure proper format for ZIP+4
            for idx in result_df[zip4_mask].index:
                parts = result_df.loc[idx, col].split
# Handle 9-digit ZIP codes (ZIP+4)
            zip4_mask = result_df[col].str.contains('-', na=False)
            
            # Ensure proper format for ZIP+4
            for idx in result_df[zip4_mask].index:
                parts = result_df.loc[idx, col].split('-')
                if len(parts) == 2:
                    main_zip = parts[0].zfill(5)
                    plus4 = parts[1].zfill(4)
                    result_df.loc[idx, col] = f"{main_zip}-{plus4}"
    
    # Standardize city names
    if 'city' in address_columns:
        col = address_columns['city']
        if col in df.columns:
            # Apply title case
            result_df[col] = df[col].str.title()
            
            # Standardize common abbreviations
            city_abbrev_map = {
                r'St\.': 'Saint',
                r'Mt\.': 'Mount',
                r'Ft\.': 'Fort',
                r'N\.': 'North',
                r'S\.': 'South',
                r'E\.': 'East',
                r'W\.': 'West',
            }
            
            for pattern, replacement in city_abbrev_map.items():
                result_df[col] = result_df[col].str.replace(pattern, replacement, regex=True)
    
    return result_df

def standardize_company_names(df, company_column):
    """
    Standardize company name formats
    
    Args:
        df: Pandas DataFrame
        company_column: Column name containing company names
        
    Returns:
        DataFrame with standardized company names
    """
    result_df = df.copy()
    
    if company_column not in df.columns:
        return result_df
    
    # Convert to string, strip whitespace, and convert to title case
    result_df[company_column] = (df[company_column]
                               .astype(str)
                               .str.strip()
                               .str.title())
    
    # Standardize legal suffixes
    suffix_map = {
        r'\bInc$': 'Inc.',
        r'\bInc\.$': 'Inc.',
        r'\bInc\b': 'Inc.',
        r'\bCorp$': 'Corp.',
        r'\bCorp\.$': 'Corp.',
        r'\bCorp\b': 'Corp.',
        r'\bCorporation$': 'Corp.',
        r'\bCorporation\.$': 'Corp.',
        r'\bCorporation\b': 'Corp.',
        r'\bLlc$': 'LLC',
        r'\bLlc\.$': 'LLC',
        r'\bLlc\b': 'LLC',
        r'\bLimited$': 'Ltd.',
        r'\bLimited\.$': 'Ltd.',
        r'\bLimited\b': 'Ltd.',
        r'\bLtd$': 'Ltd.',
        r'\bLtd\.$': 'Ltd.',
        r'\bLtd\b': 'Ltd.',
        r'\bL\.L\.C\.$': 'LLC',
        r'\bL\.L\.C$': 'LLC',
        r'\bL\.P\.$': 'LP',
        r'\bLlp$': 'LLP',
        r'\bLlp\.$': 'LLP',
        r'\bL\.L\.P\.$': 'LLP',
    }
    
    for pattern, replacement in suffix_map.items():
        result_df[company_column] = result_df[company_column].str.replace(pattern, replacement, regex=True)
    
    # Remove common prefixes that might be inconsistently applied
    prefix_map = {
        r'^The ': '',
    }
    
    for pattern, replacement in prefix_map.items():
        result_df[company_column] = result_df[company_column].str.replace(pattern, replacement, regex=True)
    
    # Fix specific abbreviations
    abbrev_map = {
        r'\bIntl\b': 'International',
        r'\bIntl\.\b': 'International',
        r'\bSys\b': 'Systems',
        r'\bSys\.\b': 'Systems',
        r'\bTech\b': 'Technologies',
        r'\bTech\.\b': 'Technologies',
        r'\bMfg\b': 'Manufacturing',
        r'\bMfg\.\b': 'Manufacturing',
        r'\bSvcs\b': 'Services',
        r'\bSvcs\.\b': 'Services',
        r'\bAssoc\b': 'Associates',
        r'\bAssoc\.\b': 'Associates',
        r'\bCo\b': 'Company',
        r'\bCo\.\b': 'Company',
    }
    
    for pattern, replacement in abbrev_map.items():
        result_df[company_column] = result_df[company_column].str.replace(pattern, replacement, regex=True)
    
    # Normalize ampersands
    result_df[company_column] = result_df[company_column].str.replace(' And ', ' & ', regex=False)
    
    return result_df

def standardize_phone_numbers(df, phone_column, format_pattern='(XXX) XXX-XXXX'):
    """
    Standardize phone number formats
    
    Args:
        df: Pandas DataFrame
        phone_column: Column name containing phone numbers
        format_pattern: Desired format pattern, where X represents a digit
            Options include: '(XXX) XXX-XXXX', 'XXX-XXX-XXXX', 'XXXXXXXXXX'
        
    Returns:
        DataFrame with standardized phone numbers
    """
    result_df = df.copy()
    
    if phone_column not in df.columns:
        return result_df
    
    # Extract digits only
    result_df[phone_column] = df[phone_column].astype(str).apply(
        lambda x: ''.join(c for c in x if c.isdigit())
    )
    
    # Apply desired format
    def format_phone(phone_digits):
        if not phone_digits or len(phone_digits) < 10:
            return phone_digits  # Return original if invalid
        
        # Handle country code if present
        country_code = ''
        local_number = phone_digits
        
        if len(phone_digits) > 10:
            country_code = phone_digits[:-10]
            local_number = phone_digits[-10:]
        
        # Apply formatting
        if format_pattern == '(XXX) XXX-XXXX':
            formatted = f"({local_number[:3]}) {local_number[3:6]}-{local_number[6:10]}"
        elif format_pattern == 'XXX-XXX-XXXX':
            formatted = f"{local_number[:3]}-{local_number[3:6]}-{local_number[6:10]}"
        else:  # Plain format
            formatted = local_number
        
        # Add country code if present
        if country_code:
            formatted = f"+{country_code} {formatted}"
        
        return formatted
    
    result_df[phone_column] = result_df[phone_column].apply(format_phone)
    
    return result_df

def standardize_dates(df, date_columns, output_format='%Y-%m-%d'):
    """
    Standardize date formats
    
    Args:
        df: Pandas DataFrame
        date_columns: List of column names containing dates
        output_format: Desired date format string
        
    Returns:
        DataFrame with standardized dates
    """
    result_df = df.copy()
    
    for col in date_columns:
        if col not in df.columns:
            continue
        
        # Try to convert to datetime
        try:
            # Handle various input formats
            result_df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Convert to desired output format
            result_df[col] = result_df[col].dt.strftime(output_format)
        except Exception as e:
            print(f"Error standardizing dates in column '{col}': {str(e)}")
    
    return result_df

def standardize_categorical_values(df, categorical_mappings):
    """
    Standardize categorical values using mapping dictionaries
    
    Args:
        df: Pandas DataFrame
        categorical_mappings: Dictionary mapping column names to value mappings
            e.g., {'gender': {'M': 'Male', 'F': 'Female'}, 'status': {'A': 'Active', 'I': 'Inactive'}}
        
    Returns:
        DataFrame with standardized categorical values
    """
    result_df = df.copy()
    
    for col, mapping in categorical_mappings.items():
        if col not in df.columns:
            continue
        
        # Apply mapping to standardize values
        result_df[col] = df[col].map(mapping).fillna(df[col])
    
    return result_df

def standardize_currency(df, currency_columns, decimal_places=2):
    """
    Standardize currency values
    
    Args:
        df: Pandas DataFrame
        currency_columns: List of column names containing currency values
        decimal_places: Number of decimal places to round to
        
    Returns:
        DataFrame with standardized currency values
    """
    result_df = df.copy()
    
    for col in currency_columns:
        if col not in df.columns:
            continue
        
        # Convert to numeric, coercing errors
        result_df[col] = pd.to_numeric(
            # Remove currency symbols and commas
            df[col].astype(str)
                 .str.replace(r'[$€£¥]', '', regex=True)
                 .str.replace(r',', '', regex=True),
            errors='coerce'
        )
        
        # Round to specified decimal places
        result_df[col] = result_df[col].round(decimal_places)
    
    return result_df
```

## Data Type Validation

Ensuring data types match expected formats:

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Callable, Optional
import re
import datetime

@dataclass
class TypeSpec:
    """Specification for a column data type"""
    name: str
    validation_fn: Callable
    coercion_fn: Optional[Callable] = None
    parameters: Dict[str, Any] = None

class DataTypeValidator:
    """Validate and enforce data types in pandas DataFrames"""
    
    def __init__(self):
        self.type_specs = {}
        self._register_standard_types()
    
    def _register_standard_types(self):
        """Register standard data types"""
        
        # Integer
        self.register_type(
            TypeSpec(
                name="integer",
                validation_fn=lambda s: pd.to_numeric(s, errors='coerce').notnull() & 
                                     pd.to_numeric(s, errors='coerce').astype(int) == 
                                     pd.to_numeric(s, errors='coerce'),
                coercion_fn=lambda s: pd.to_numeric(s, errors='coerce').astype('Int64')  # Use nullable integer type
            )
        )
        
        # Float
        self.register_type(
            TypeSpec(
                name="float",
                validation_fn=lambda s: pd.to_numeric(s, errors='coerce').notnull(),
                coercion_fn=lambda s: pd.to_numeric(s, errors='coerce')
            )
        )
        
        # Boolean
        self.register_type(
            TypeSpec(
                name="boolean",
                validation_fn=lambda s: (s.astype(str).str.lower().isin(['true', 'false', 't', 'f', 'yes', 'no', 'y', 'n', '1', '0'])) | 
                                     (s.isin([1, 0, True, False])),
                coercion_fn=lambda s: s.map({
                    'true': True, 'false': False, 
                    't': True, 'f': False,
                    'yes': True, 'no': False,
                    'y': True, 'n': False,
                    '1': True, '0': False,
                    1: True, 0: False,
                    True: True, False: False
                }).astype('boolean')  # Use nullable boolean type
            )
        )
        
        # Date
        self.register_type(
            TypeSpec(
                name="date",
                validation_fn=lambda s: pd.to_datetime(s, errors='coerce').notnull(),
                coercion_fn=lambda s, format=None: pd.to_datetime(s, errors='coerce', format=format)
            )
        )
        
        # String
        self.register_type(
            TypeSpec(
                name="string",
                validation_fn=lambda s: s.notnull(),
                coercion_fn=lambda s: s.astype(str)
            )
        )
        
        # Email
        self.register_type(
            TypeSpec(
                name="email",
                validation_fn=lambda s: s.astype(str).str.match(
                    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                ),
                coercion_fn=lambda s: s.astype(str)
            )
        )
        
        # Phone
        self.register_type(
            TypeSpec(
                name="phone",
                validation_fn=lambda s: s.astype(str).str.replace(r'[^0-9]', '', regex=True).str.len().between(10, 15),
                coercion_fn=lambda s: s.astype(str)
            )
        )
        
        # IP Address
        self.register_type(
            TypeSpec(
                name="ip_address",
                validation_fn=lambda s: s.astype(str).str.match(
                    r'^(\d{1,3}\.){3}\d{1,3}$'
                ),
                coercion_fn=lambda s: s.astype(str)
            )
        )
        
        # URL
        self.register_type(
            TypeSpec(
                name="url",
                validation_fn=lambda s: s.astype(str).str.match(
                    r'^(http|https)://[a-zA-Z0-9]+([\-\.]{1}[a-zA-Z0-9]+)*\.[a-zA-Z]{2,}(:[0-9]{1,5})?(\/.*)?$'
                ),
                coercion_fn=lambda s: s.astype(str)
            )
        )
        
        # Currency
        self.register_type(
            TypeSpec(
                name="currency",
                validation_fn=lambda s: pd.to_numeric(
                    s.astype(str).str.replace(r'[$€£¥,]', '', regex=True),
                    errors='coerce'
                ).notnull(),
                coercion_fn=lambda s: pd.to_numeric(
                    s.astype(str).str.replace(r'[$€£¥,]', '', regex=True),
                    errors='coerce'
                )
            )
        )
        
        # Percentage
        self.register_type(
            TypeSpec(
                name="percentage",
                validation_fn=lambda s: pd.to_numeric(
                    s.astype(str).str.replace(r'%', '', regex=True),
                    errors='coerce'
                ).notnull(),
                coercion_fn=lambda s: pd.to_numeric(
                    s.astype(str).str.replace(r'%', '', regex=True),
                    errors='coerce'
                ) / 100
            )
        )
        
        # ZIP/Postal Code (US)
        self.register_type(
            TypeSpec(
                name="zipcode",
                validation_fn=lambda s: s.astype(str).str.match(r'^\d{5}(-\d{4})?$'),
                coercion_fn=lambda s: s.astype(str)
            )
        )
        
        # Social Security Number (US)
        self.register_type(
            TypeSpec(
                name="ssn",
                validation_fn=lambda s: s.astype(str).str.replace(r'[^0-9]', '', regex=True).str.len() == 9,
                coercion_fn=lambda s: s.astype(str).str.replace(r'[^0-9]', '', regex=True)
            )
        )
    
    def register_type(self, type_spec):
        """Register a custom data type specification"""
        self.type_specs[type_spec.name] = type_spec
    
    def validate_column(self, df, column, type_name, **kwargs):
        """
        Validate a column against a specified data type
        
        Args:
            df: Pandas DataFrame
            column: Column name to validate
            type_name: Name of the type specification to use
            **kwargs: Additional parameters for the validation function
            
        Returns:
            Tuple of (is_valid, invalid_count, invalid_indices, invalid_examples)
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if type_name not in self.type_specs:
            raise ValueError(f"Type specification '{type_name}' not found")
        
        type_spec = self.type_specs[type_name]
        
        # Apply validation function
        validation_result = type_spec.validation_fn(df[column], **kwargs)
        
        # Calculate invalid values
        invalid_mask = ~validation_result
        invalid_count = invalid_mask.sum()
        invalid_indices = df[invalid_mask].index.tolist()
        invalid_examples = df.loc[invalid_mask, column].head(5).tolist()
        
        is_valid = invalid_count == 0
        
        return is_valid, invalid_count, invalid_indices, invalid_examples
    
    def coerce_column(self, df, column, type_name, **kwargs):
        """
        Attempt to coerce a column to a specified data type
        
        Args:
            df: Pandas DataFrame
            column: Column name to coerce
            type_name: Name of the type specification to use
            **kwargs: Additional parameters for the coercion function
            
        Returns:
            DataFrame with coerced column and count of values changed
        """
        result_df = df.copy()
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if type_name not in self.type_specs:
            raise ValueError(f"Type specification '{type_name}' not found")
        
        type_spec = self.type_specs[type_name]
        
        if type_spec.coercion_fn is None:
            raise ValueError(f"Type '{type_name}' does not have a coercion function")
        
        # Store original values
        original_values = df[column].copy()
        
        # Apply coercion function
        result_df[column] = type_spec.coercion_fn(df[column], **kwargs)
        
        # Count changed values (excluding NA to NA conversions)
        original_na = original_values.isna()
        result_na = result_df[column].isna()
        
        # Values that changed (excluding NA to NA)
        changed_mask = ((original_values != result_df[column]) | 
                       (result_na & ~original_na))  # New NAs
        
        # Don't count NA to NA as changes
        changed_mask = changed_mask & ~(original_na & result_na)
        
        changed_count = changed_mask.sum()
        
        return result_df, changed_count
    
    def validate_dataframe(self, df, schema):
        """
        Validate a DataFrame against a schema of column types
        
        Args:
            df: Pandas DataFrame
            schema: Dictionary mapping column names to type specifications
                e.g., {'customer_id': 'integer', 'email': 'email', 'birthdate': 'date'}
                
        Returns:
            Dictionary containing validation results
        """
        results = {}
        
        for column, type_spec in schema.items():
            # Handle type specs with parameters
            if isinstance(type_spec, dict):
                type_name = type_spec['type']
                params = {k: v for k, v in type_spec.items() if k != 'type'}
            else:
                type_name = type_spec
                params = {}
            
            # Skip if column doesn't exist
            if column not in df.columns:
                results[column] = {
                    'valid': False,
                    'error': f"Column '{column}' not found in DataFrame",
                    'missing_column': True
                }
                continue
            
            try:
                is_valid, invalid_count, invalid_indices, invalid_examples = self.validate_column(
                    df, column, type_name, **params
                )
                
                results[column] = {
                    'valid': is_valid,
                    'type': type_name,
                    'params': params,
                    'invalid_count': invalid_count,
                    'invalid_percentage': (invalid_count / len(df) * 100) if len(df) > 0 else 0,
                    'invalid_examples': invalid_examples
                }
            except Exception as e:
                results[column] = {
                    'valid': False,
                    'error': str(e)
                }
        
        # Calculate overall validation status
        invalid_columns = [col for col, res in results.items() 
                         if not res.get('valid', False) and not res.get('missing_column', False)]
        
        summary = {
            'valid': len(invalid_columns) == 0,
            'column_count': len(schema),
            'invalid_column_count': len(invalid_columns),
            'invalid_columns': invalid_columns
        }
        
        return {
            'column_results': results,
            'summary': summary
        }
    
    def coerce_dataframe(self, df, schema):
        """
        Coerce a DataFrame to match a schema of column types
        
        Args:
            df: Pandas DataFrame
            schema: Dictionary mapping column names to type specifications
                
        Returns:
            Tuple of (coerced_df, coercion_report)
        """
        result_df = df.copy()
        report = {}
        
        for column, type_spec in schema.items():
            # Handle type specs with parameters
            if isinstance(type_spec, dict):
                type_name = type_spec['type']
                params = {k: v for k, v in type_spec.items() if k != 'type'}
            else:
                type_name = type_spec
                params = {}
            
            # Skip if column doesn't exist
            if column not in df.columns:
                report[column] = {
                    'success': False,
                    'error': f"Column '{column}' not found in DataFrame"
                }
                continue
            
            try:
                # Attempt to coerce the column
                result_df, changed_count = self.coerce_column(
                    result_df, column, type_name, **params
                )
                
                report[column] = {
                    'success': True,
                    'type': type_name,
                    'changed_count': changed_count,
                    'changed_percentage': (changed_count / len(df) * 100) if len(df) > 0 else 0
                }
            except Exception as e:
                report[column] = {
                    'success': False,
                    'error': str(e)
                }
        
        return result_df, report
    
    def generate_schema_from_dataframe(self, df, sample_rows=1000):
        """
        Automatically generate a schema from a DataFrame by inferring types
        
        Args:
            df: Pandas DataFrame
            sample_rows: Number of rows to sample for type inference
            
        Returns:
            Dictionary mapping column names to inferred type specifications
        """
        schema = {}
        
        # Sample the DataFrame if needed
        if sample_rows and len(df) > sample_rows:
            sample_df = df.sample(sample_rows, random_state=42)
        else:
            sample_df = df
        
        for column in df.columns:
            # Skip empty columns
            if sample_df[column].isna().all():
                continue
            
            # Try to infer type based on content
            inferred_type = self._infer_column_type(sample_df[column])
            schema[column] = inferred_type
        
        return schema
    
    def _infer_column_type(self, series):
        """Helper method to infer a column's data type"""
        # Skip if all values are null
        if series.isna().all():
            return 'string'
        
        # Get non-null values
        non_null = series.dropna()
        
        # Check for Boolean
        if set(non_null.unique()) <= {0, 1, True, False, 'true', 'false', 'True', 'False', 'yes', 'no', 'y', 'n', 't', 'f'}:
            return 'boolean'
        
        # Check for Integer
        try:
            if pd.to_numeric(non_null, errors='coerce').notna().all():
                if (pd.to_numeric(non_null) == pd.to_numeric(non_null).astype(int)).all():
                    return 'integer'
        except:
            pass
        
        # Check for Float
        try:
            if pd.to_numeric(non_null, errors='coerce').notna().all():
                return 'float'
        except:
            pass
        
        # Check for Date
        try:
            if pd.to_datetime(non_null, errors='coerce').notna().all():
                return 'date'
        except:
            pass
        
        # Check for Email
        if non_null.astype(str).str.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$').all():
            return 'email'
        
        # Check for Phone
        if non_null.astype(str).str.replace(r'[^0-9]', '', regex=True).str.len().between(10, 15).all():
            return 'phone'
        
        # Check for ZIP code (US)
        if non_null.astype(str).str.match(r'^\d{5}(-\d{4})?$').all():
            return 'zipcode'
        
        # Default to string
        return 'string'
```

## Business Rule Implementation

Creating and enforcing business rules:

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Set, Tuple
import re
import datetime

@dataclass
class BusinessRule:
    """Representation of a business rule"""
    rule_id: str
    name: str
    description: str
    rule_fn: Callable  # Function that checks if the rule is satisfied
    columns: List[str]  # Columns needed for this rule
    severity: str = 'medium'  # 'low', 'medium', 'high'
    category: str = 'data_quality'  # Category of rule (e.g., 'completeness', 'validity')
    active: bool = True
    dependencies: List[str] = field(default_factory=list)  # IDs of rules that must be checked first

@dataclass
class RuleResult:
    """Result of a business rule check"""
    rule_id: str
    satisfied: bool
    violation_count: int
    violation_indices: List[int]
    violation_examples: List[Dict[str, Any]]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

class BusinessRuleEngine:
    """Engine for applying business rules to data"""
    
    def __init__(self):
        self.rules = {}
        self.rule_results = {}
    
    def add_rule(self, rule):
        """Add a business rule to the engine"""
        self.rules[rule.rule_id] = rule
    
    def evaluate_rule(self, df, rule_id, **kwargs):
        """
        Evaluate a single business rule
        
        Args:
            df: Pandas DataFrame
            rule_id: ID of the rule to evaluate
            **kwargs: Additional parameters for the rule function
            
        Returns:
            RuleResult object
        """
        if rule_id not in self.rules:
            raise ValueError(f"Rule '{rule_id}' not found")
        
        rule = self.rules[rule_id]
        
        # Check if rule is active
        if not rule.active:
            return RuleResult(
                rule_id=rule_id,
                satisfied=True,
                violation_count=0,
                violation_indices=[],
                violation_examples=[],
                message=f"Rule '{rule.name}' is inactive and was skipped"
            )
        
        # Check if required columns exist
        missing_columns = [col for col in rule.columns if col not in df.columns]
        if missing_columns:
            return RuleResult(
                rule_id=rule_id,
                satisfied=False,
                violation_count=len(df),
                violation_indices=[],
                violation_examples=[],
                message=f"Rule '{rule.name}' could not be evaluated because columns are missing: {', '.join(missing_columns)}"
            )
        
        # Check dependencies
        for dep_rule_id in rule.dependencies:
            if dep_rule_id not in self.rule_results:
                # Evaluate the dependency first
                self.evaluate_rule(df, dep_rule_id, **kwargs)
            
            # Check if dependency was satisfied
            if not self.rule_results[dep_rule_id].satisfied:
                return RuleResult(
                    rule_id=rule_id,
                    satisfied=False,
                    violation_count=0,
                    violation_indices=[],
                    violation_examples=[],
                    message=f"Rule '{rule.name}' was skipped because dependency '{dep_rule_id}' was not satisfied"
                )
        
        try:
            # Apply the rule function to get a boolean mask of satisfied rows
            satisfied_mask = rule.rule_fn(df, **kwargs)
            
            # Calculate violations
            violations_mask = ~satisfied_mask
            violation_count = violations_mask.sum()
            
            # Get indices of violations
            violation_indices = df.index[violations_mask].tolist()
            
            # Get example violations (limited to 5)
            violation_df = df[violations_mask].head(5)
            violation_examples = []
            
            for _, row in violation_df.iterrows():
                # Include only the relevant columns in the example
                example = {col: row[col] for col in rule.columns if col in row}
                violation_examples.append(example)
            
            # Determine if rule is satisfied overall
            is_satisfied = violation_count == 0
            
            # Generate message
            if is_satisfied:
                message = f"Rule '{rule.name}' is satisfied for all rows"
# Generate message
            if is_satisfied:
                message = f"Rule '{rule.name}' is satisfied for all rows"
            else:
                message = f"Rule '{rule.name}' has {violation_count} violations ({violation_count/len(df)*100:.2f}% of rows)"
            
            # Create result object
            result = RuleResult(
                rule_id=rule_id,
                satisfied=is_satisfied,
                violation_count=violation_count,
                violation_indices=violation_indices,
                violation_examples=violation_examples,
                message=message,
                details={
                    'total_rows': len(df),
                    'params': kwargs
                }
            )
            
            # Store the result
            self.rule_results[rule_id] = result
            
            return result
            
        except Exception as e:
            # Handle errors in rule evaluation
            error_result = RuleResult(
                rule_id=rule_id,
                satisfied=False,
                violation_count=0,
                violation_indices=[],
                violation_examples=[],
                message=f"Error evaluating rule '{rule.name}': {str(e)}"
            )
            
            self.rule_results[rule_id] = error_result
            return error_result
    
    def evaluate_all_rules(self, df, **kwargs):
        """
        Evaluate all active rules against a DataFrame
        
        Args:
            df: Pandas DataFrame to evaluate
            **kwargs: Additional parameters for rule functions
            
        Returns:
            Dictionary with rule results and summary
        """
        # Reset results
        self.rule_results = {}
        
        # Sort rules by dependencies
        sorted_rules = self._sort_rules_by_dependencies()
        
        # Evaluate each rule in order
        for rule_id in sorted_rules:
            self.evaluate_rule(df, rule_id, **kwargs)
        
        # Generate summary
        satisfied_count = sum(1 for r in self.rule_results.values() if r.satisfied)
        total_count = len(self.rule_results)
        success_percentage = (satisfied_count / total_count * 100) if total_count > 0 else 0.0
        
        # Count violations by severity
        high_severity_violations = 0
        medium_severity_violations = 0
        low_severity_violations = 0
        
        for rule_id, result in self.rule_results.items():
            if not result.satisfied:
                severity = self.rules[rule_id].severity
                
                if severity == 'high':
                    high_severity_violations += 1
                elif severity == 'medium':
                    medium_severity_violations += 1
                elif severity == 'low':
                    low_severity_violations += 1
        
        summary = {
            'satisfied_rules': satisfied_count,
            'total_rules': total_count,
            'success_percentage': success_percentage,
            'high_severity_violations': high_severity_violations,
            'medium_severity_violations': medium_severity_violations,
            'low_severity_violations': low_severity_violations,
            'evaluation_time': datetime.datetime.now().isoformat()
        }
        
        return {
            'results': self.rule_results,
            'summary': summary
        }
    
    def _sort_rules_by_dependencies(self):
        """
        Sort rules based on dependencies to ensure they're evaluated in the right order
        
        Returns:
            List of rule IDs in the order they should be evaluated
        """
        # Create a directed graph of rule dependencies
        graph = {rule_id: set(rule.dependencies) for rule_id, rule in self.rules.items() if rule.active}
        
        # Topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node):
            if node in temp_visited:
                # Cyclic dependency detected
                raise ValueError(f"Cyclic dependency detected in business rules involving '{node}'")
            
            if node not in visited:
                temp_visited.add(node)
                
                for dep in graph.get(node, set()):
                    visit(dep)
                
                temp_visited.remove(node)
                visited.add(node)
                order.append(node)
        
        # Visit each node
        for node in graph:
            if node not in visited:
                visit(node)
        
        # Reverse to get the correct order (dependencies first)
        return order[::-1]
    
    def generate_rule_report(self, rule_results, output_file=None):
        """
        Generate an HTML report of business rule evaluation
        
        Args:
            rule_results: Results from evaluate_all_rules
            output_file: Optional file path to save the report as HTML
        """
        summary = rule_results['summary']
        results = rule_results['results']
        
        # Start building HTML
        html = f"""
        <html>
        <head>
            <title>Business Rules Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .satisfied {{ background-color: #c8e6c9; }}
                .violated {{ background-color: #ffcdd2; }}
                .severity-high {{ border-left: 5px solid #f44336; }}
                .severity-medium {{ border-left: 5px solid #ff9800; }}
                .severity-low {{ border-left: 5px solid #4caf50; }}
                .summary {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
                .summary-box {{ padding: 15px; border-radius: 5px; width: 30%; text-align: center; }}
                .success-rate {{ background-color: #e3f2fd; }}
                .violations {{ background-color: #ffebee; }}
                .status {{ background-color: #f5f5f5; }}
                details {{ margin-bottom: 10px; }}
                summary {{ cursor: pointer; padding: 10px; background-color: #f2f2f2; }}
                .violations-list {{ background-color: #f9f9f9; padding: 10px; }}
            </style>
        </head>
        <body>
            <h1>Business Rules Evaluation Report</h1>
            <p>Generated: {summary['evaluation_time']}</p>
            
            <div class="summary">
                <div class="summary-box success-rate">
                    <h2>Success Rate</h2>
                    <p style="font-size: 24px;">{summary['success_percentage']:.1f}%</p>
                    <p>{summary['satisfied_rules']} of {summary['total_rules']} rules satisfied</p>
                </div>
                
                <div class="summary-box violations">
                    <h2>Violations by Severity</h2>
                    <p><strong>High:</strong> {summary['high_severity_violations']}</p>
                    <p><strong>Medium:</strong> {summary['medium_severity_violations']}</p>
                    <p><strong>Low:</strong> {summary['low_severity_violations']}</p>
                </div>
                
                <div class="summary-box status">
                    <h2>Overall Status</h2>
                    <p style="font-size: 18px; font-weight: bold;">
        """
        
        # Determine overall status
        if summary['high_severity_violations'] > 0:
            html += "<span style='color: #f44336;'>FAILED</span>"
        elif summary['medium_severity_violations'] > 0:
            html += "<span style='color: #ff9800;'>WARNING</span>"
        else:
            html += "<span style='color: #4caf50;'>PASSED</span>"
        
        html += """
                    </p>
                </div>
            </div>
            
            <h2>Business Rules Results</h2>
            <table>
                <tr>
                    <th>Rule ID</th>
                    <th>Name</th>
                    <th>Status</th>
                    <th>Violations</th>
                    <th>Message</th>
                </tr>
        """
        
        # Add a row for each rule result
        for rule_id, result in results.items():
            rule = self.rules.get(rule_id)
            if not rule:
                continue
                
            status_class = "satisfied" if result.satisfied else "violated"
            severity_class = f"severity-{rule.severity}"
            
            html += f"""
                <tr class="{status_class} {severity_class}">
                    <td>{rule_id}</td>
                    <td>{rule.name}</td>
                    <td>{"PASS" if result.satisfied else "FAIL"}</td>
                    <td>{result.violation_count}</td>
                    <td>{result.message}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Violation Details</h2>
        """
        
        # Add details for each violated rule
        for rule_id, result in results.items():
            if result.satisfied:
                continue
                
            rule = self.rules.get(rule_id)
            if not rule:
                continue
                
            html += f"""
            <details>
                <summary>
                    <strong>{rule.name}</strong> ({rule_id}) - {result.violation_count} violations
                </summary>
                <div class="violations-list">
                    <p><strong>Description:</strong> {rule.description}</p>
                    <p><strong>Severity:</strong> {rule.severity}</p>
                    <p><strong>Category:</strong> {rule.category}</p>
                    <p><strong>Columns:</strong> {', '.join(rule.columns)}</p>
                    
                    <h4>Example Violations:</h4>
                    <table>
                        <tr>
            """
            
            # Add table headers for columns
            for col in rule.columns:
                html += f"<th>{col}</th>"
            
            html += """
                        </tr>
            """
            
            # Add example violations
            for example in result.violation_examples:
                html += "<tr>"
                for col in rule.columns:
                    html += f"<td>{example.get(col, 'N/A')}</td>"
                html += "</tr>"
            
            html += """
                    </table>
                </div>
            </details>
            """
        
        html += """
        </body>
        </html>
        """
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html)
            print(f"Rule evaluation report saved to {output_file}")
        
        return html
```

## Duplicate Detection

Identifying and handling duplicate records:

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jellyfish
import re

def identify_exact_duplicates(df, subset=None):
    """
    Identify exact duplicate records in a DataFrame
    
    Args:
        df: Pandas DataFrame
        subset: Optional list of columns to consider for duplication
        
    Returns:
        Tuple of (duplicate_mask, groups, stats)
    """
    # Find duplicates
    if subset:
        duplicate_mask = df.duplicated(subset=subset, keep=False)
    else:
        duplicate_mask = df.duplicated(keep=False)
    
    # Get duplicate records
    duplicates = df[duplicate_mask].copy()
    
    # Group by duplicate values
    if subset:
        groups = duplicates.groupby(subset)
    else:
        groups = duplicates.groupby(list(df.columns))
    
    # Calculate statistics
    stats = {
        'total_rows': len(df),
        'duplicate_rows': len(duplicates),
        'duplicate_percentage': (len(duplicates) / len(df) * 100) if len(df) > 0 else 0,
        'unique_groups': groups.ngroups,
        'largest_group_size': max(groups.size()) if not groups.empty else 0
    }
    
    return duplicate_mask, groups, stats

def identify_fuzzy_duplicates(df, match_columns, threshold=0.8):
    """
    Identify fuzzy/similar records that might be duplicates
    
    Args:
        df: Pandas DataFrame
        match_columns: Columns to use for fuzzy matching
        threshold: Similarity threshold (0-1)
        
    Returns:
        DataFrame with potential duplicate pairs
    """
    # Create a copy for processing
    process_df = df.copy()
    
    # Combine match columns into a single string for comparison
    def combine_fields(row):
        values = []
        for col in match_columns:
            if col in row and pd.notna(row[col]):
                values.append(str(row[col]).lower())
        return ' '.join(values)
    
    process_df['_combined'] = process_df.apply(combine_fields, axis=1)
    
    # Calculate TF-IDF vectors
    vectorizer = TfidfVectorizer(min_df=1, analyzer='word')
    tfidf_matrix = vectorizer.fit_transform(process_df['_combined'])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    # Find similar pairs above threshold
    similar_pairs = []
    
    for i in range(len(df)):
        for j in range(i+1, len(df)):  # Compare with subsequent rows only
            if cosine_sim[i, j] >= threshold:
                similarity = cosine_sim[i, j]
                
                pair = {
                    'index_1': df.index[i],
                    'index_2': df.index[j],
                    'similarity': similarity
                }
                
                # Add values for match columns
                for col in match_columns:
                    pair[f'{col}_1'] = df.iloc[i][col]
                    pair[f'{col}_2'] = df.iloc[j][col]
                
                similar_pairs.append(pair)
    
    # Convert to DataFrame
    if similar_pairs:
        return pd.DataFrame(similar_pairs).sort_values('similarity', ascending=False)
    else:
        return pd.DataFrame(columns=['index_1', 'index_2', 'similarity'])

def calculate_string_similarity(str1, str2, method='jaro_winkler'):
    """
    Calculate string similarity using various methods
    
    Args:
        str1: First string
        str2: Second string
        method: Similarity method ('jaro_winkler', 'levenshtein', 'damerau_levenshtein', 'hamming')
        
    Returns:
        Similarity score (0-1)
    """
    if pd.isna(str1) or pd.isna(str2):
        return 0
    
    str1 = str(str1).lower()
    str2 = str(str2).lower()
    
    if method == 'jaro_winkler':
        return jellyfish.jaro_winkler_similarity(str1, str2)
    elif method == 'levenshtein':
        return 1 - (jellyfish.levenshtein_distance(str1, str2) / max(len(str1), len(str2), 1))
    elif method == 'damerau_levenshtein':
        return 1 - (jellyfish.damerau_levenshtein_distance(str1, str2) / max(len(str1), len(str2), 1))
    elif method == 'hamming':
        # Pad shorter string to make lengths equal
        if len(str1) < len(str2):
            str1 = str1 + ' ' * (len(str2) - len(str1))
        elif len(str2) < len(str1):
            str2 = str2 + ' ' * (len(str1) - len(str2))
        return 1 - (jellyfish.hamming_distance(str1, str2) / max(len(str1), 1))
    else:
        raise ValueError(f"Unsupported similarity method: {method}")

def find_similar_records(df, fields_config, threshold=0.8):
    """
    Find similar records using field-specific comparisons
    
    Args:
        df: Pandas DataFrame
        fields_config: Dictionary mapping column names to comparison configs
            Config options: {
                'weight': Weight of this field in overall similarity (default: 1),
                'method': Comparison method to use (default: 'jaro_winkler'),
                'preprocessing': Optional function to preprocess values before comparison
            }
        threshold: Minimum overall similarity score to consider as potential duplicate
        
    Returns:
        DataFrame with potential duplicate pairs
    """
    # Prepare weights
    total_weight = sum(config.get('weight', 1) for config in fields_config.values())
    normalized_weights = {field: config.get('weight', 1) / total_weight 
                       for field, config in fields_config.items()}
    
    # Storage for similar pairs
    similar_pairs = []
    
    # Helper to preprocess field values
    def preprocess_value(value, config):
        preprocessing_fn = config.get('preprocessing')
        if preprocessing_fn and callable(preprocessing_fn):
            return preprocessing_fn(value)
        return value
    
    # Compare all pairs
    for i in range(len(df)):
        for j in range(i+1, len(df)):  # Only compare with subsequent rows
            overall_similarity = 0
            
            # Calculate similarity for each field
            field_similarities = {}
            
            for field, config in fields_config.items():
                if field not in df.columns:
                    continue
                
                value1 = preprocess_value(df.iloc[i][field], config)
                value2 = preprocess_value(df.iloc[j][field], config)
                
                # Calculate similarity
                method = config.get('method', 'jaro_winkler')
                similarity = calculate_string_similarity(value1, value2, method)
                
                # Apply weight
                weight = normalized_weights[field]
                weighted_similarity = similarity * weight
                
                field_similarities[field] = similarity
                overall_similarity += weighted_similarity
            
            # If overall similarity is above threshold, consider as potential duplicate
            if overall_similarity >= threshold:
                pair = {
                    'index_1': df.index[i],
                    'index_2': df.index[j],
                    'overall_similarity': overall_similarity
                }
                
                # Add individual field similarities
                for field in fields_config:
                    if field in field_similarities:
                        pair[f'{field}_similarity'] = field_similarities[field]
                
                # Add values for match fields
                for field in fields_config:
                    if field in df.columns:
                        pair[f'{field}_1'] = df.iloc[i][field]
                        pair[f'{field}_2'] = df.iloc[j][field]
                
                similar_pairs.append(pair)
    
    # Convert to DataFrame
    if similar_pairs:
        return pd.DataFrame(similar_pairs).sort_values('overall_similarity', ascending=False)
    else:
        return pd.DataFrame(columns=['index_1', 'index_2', 'overall_similarity'])

def merge_duplicate_records(df, duplicate_groups, merge_strategy=None):
    """
    Merge duplicate records based on a specified strategy
    
    Args:
        df: Pandas DataFrame
        duplicate_groups: GroupBy object from identify_exact_duplicates
        merge_strategy: Dictionary mapping column names to merge strategies
            Strategies: 'first', 'last', 'most_common', 'longest', 'non_empty',
                        'sum', 'mean', 'min', 'max', or a custom function
            
    Returns:
        DataFrame with duplicates merged
    """
    result_df = df.copy()
    
    # Default merge strategy
    if merge_strategy is None:
        merge_strategy = {col: 'first' for col in df.columns}
    
    # Fill in missing strategies with 'first'
    for col in df.columns:
        if col not in merge_strategy:
            merge_strategy[col] = 'first'
    
    # Process each duplicate group
    merged_records = []
    processed_indices = set()
    
    for _, group in duplicate_groups:
        merged_record = {}
        
        # Track indices in this group to remove later
        group_indices = group.index.tolist()
        processed_indices.update(group_indices)
        
        # Apply merge strategy for each column
        for col in df.columns:
            values = group[col].tolist()
            strategy = merge_strategy.get(col, 'first')
            
            if strategy == 'first':
                merged_record[col] = values[0]
            elif strategy == 'last':
                merged_record[col] = values[-1]
            elif strategy == 'most_common':
                # Get most common non-null value
                value_counts = pd.Series([v for v in values if pd.notna(v)]).value_counts()
                merged_record[col] = value_counts.index[0] if not value_counts.empty else None
            elif strategy == 'longest':
                # Get longest string value
                str_values = [str(v) for v in values if pd.notna(v)]
                merged_record[col] = max(str_values, key=len) if str_values else None
            elif strategy == 'non_empty':
                # First non-empty value
                non_empty = [v for v in values if pd.notna(v) and (not isinstance(v, str) or v.strip())]
                merged_record[col] = non_empty[0] if non_empty else None
            elif strategy == 'sum':
                # Sum numeric values
                numeric_values = [v for v in values if pd.notna(v) and isinstance(v, (int, float))]
                merged_record[col] = sum(numeric_values) if numeric_values else None
            elif strategy == 'mean':
                # Average numeric values
                numeric_values = [v for v in values if pd.notna(v) and isinstance(v, (int, float))]
                merged_record[col] = sum(numeric_values) / len(numeric_values) if numeric_values else None
            elif strategy == 'min':
                # Minimum value
                non_null = [v for v in values if pd.notna(v)]
                merged_record[col] = min(non_null) if non_null else None
            elif strategy == 'max':
                # Maximum value
                non_null = [v for v in values if pd.notna(v)]
                merged_record[col] = max(non_null) if non_null else None
            elif callable(strategy):
                # Custom function
                merged_record[col] = strategy(values)
            else:
                # Default to first value
                merged_record[col] = values[0]
        
        merged_records.append(merged_record)
    
    # Create a new DataFrame without duplicates
    non_duplicates = result_df.loc[~result_df.index.isin(processed_indices)]
    merged_df = pd.DataFrame(merged_records)
    
    # Combine non-duplicates with merged records
    final_df = pd.concat([non_duplicates, merged_df], ignore_index=True)
    
    return final_df
```

## Data Quality Metrics and Reporting

Calculating and reporting data quality metrics:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

class DataQualityMetrics:
    """
    Calculate and track data quality metrics for DataFrames
    """
    
    def __init__(self, metrics_db=None):
        """
        Initialize the DataQualityMetrics calculator
        
        Args:
            metrics_db: Optional file path to store historical metrics
        """
        self.metrics_db = metrics_db
        self.historical_metrics = {}
        
        # Load historical metrics if available
        if metrics_db and os.path.exists(metrics_db):
            try:
                with open(metrics_db, 'r') as f:
                    self.historical_metrics = json.load(f)
            except:
                print(f"Could not load metrics from {metrics_db}")
    
    def calculate_completeness(self, df):
        """
        Calculate completeness metrics (percentage of non-null values)
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            Dictionary with completeness metrics
        """
        # Overall completeness
        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        overall_completeness = ((total_cells - null_cells) / total_cells * 100) if total_cells > 0 else 0
        
        # Completeness by column
        column_completeness = {}
        for column in df.columns:
            non_null_count = df[column].count()
            total_count = len(df)
            completeness = (non_null_count / total_count * 100) if total_count > 0 else 0
            column_completeness[column] = {
                'non_null_count': int(non_null_count),
                'total_count': int(total_count),
                'completeness_percentage': float(completeness)
            }
        
        return {
            'overall_completeness': float(overall_completeness),
            'column_completeness': column_completeness
        }
    
    def calculate_uniqueness(self, df):
        """
        Calculate uniqueness metrics (percentage of unique values)
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            Dictionary with uniqueness metrics
        """
        # Duplicate rows
        duplicate_rows = df.duplicated().sum()
        row_uniqueness = ((len(df) - duplicate_rows) / len(df) * 100) if len(df) > 0 else 0
        
        # Uniqueness by column
        column_uniqueness = {}
        for column in df.columns:
            unique_count = df[column].nunique(dropna=True)
            non_null_count = df[column].count()
            uniqueness = (unique_count / non_null_count * 100) if non_null_count > 0 else 0
            column_uniqueness[column] = {
                'unique_count': int(unique_count),
                'non_null_count': int(non_null_count),
                'uniqueness_percentage': float(uniqueness)
            }
        
        return {
            'row_uniqueness': float(row_uniqueness),
            'duplicate_row_count': int(duplicate_rows),
            'column_uniqueness': column_uniqueness
        }
    
    def calculate_validity(self, df, validation_rules=None):
        """
        Calculate validity metrics based on data type and range checks
        
        Args:
            df: Pandas DataFrame
            validation_rules: Optional dictionary mapping column names to validation functions
            
        Returns:
            Dictionary with validity metrics
        """
        column_validity = {}
        
        # Default validation rules based on data types
        if validation_rules is None:
            validation_rules = {}
            
            # Add default rules based on column types
            for column in df.columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    # Check for extreme outliers for numeric columns
                    validation_rules[column] = lambda s: (
                        ~((s < (s.quantile(0.25) - 3 * (s.quantile(0.75) - s.quantile(0.25)))) | 
                        (s > (s.quantile(0.75) + 3 * (s.quantile(0.75) - s.quantile(0.25)))))
                    )
                elif pd.api.types.is_string_dtype(df[column]):
                    # Check for empty strings or whitespace-only strings for string columns
                    validation_rules[column] = lambda s: ~s.astype(str).str.match(r'^\s*$')
        
        # Apply validation rules
        for column, validation_fn in validation_rules.items():
            if column not in df.columns:
                continue
            
            # Apply validation function and count valid values
            try:
                valid_mask = validation_fn(df[column])
                valid_count = valid_mask.sum()
                total_count = len(df)
                validity = (valid_count / total_count * 100) if total_count > 0 else 0
                
                column_validity[column] = {
                    'valid_count': int(valid_count),
                    'total_count': int(total_count),
                    'validity_percentage': float(validity)
                }
            except Exception as e:
                print(f"Error validating column '{column}': {str(e)}")
        
        # Calculate overall validity
        if column_validity:
            validity_values = [v['validity_percentage'] for v in column_validity.values()]
            overall_validity = sum(validity_values) / len(validity_values)
        else:
            overall_validity = 0
        
        return {
            'overall_validity': float(overall_validity),
            'column_validity': column_validity
        }
    
    def calculate_consistency(self, df, consistency_rules=None):
        """
        Calculate consistency metrics based on business rules or patterns
        
        Args:
            df: Pandas DataFrame
            consistency_rules: Dictionary mapping rule names to rule checking functions
            
        Returns:
            Dictionary with consistency metrics
        """
        # Default consistency rules if none provided
        if consistency_rules is None:
            consistency_rules = {}
            
            # Check for consistent date patterns if date columns are present
            date_columns = []
            for col in df.columns:
                try:
                    if pd.to_datetime(df[col], errors='coerce').notnull().any():
                        date_columns.append(col)
                except:
                    pass
            
            if len(date_columns) >= 2:
                # Check if dates are in consistent order
                consistency_rules['consistent_date_sequence'] = lambda df: all(
                    ((df[date_columns[i]] <= df[date_columns[i+1]]) | 
                    (df[date_columns[i]].isnull()) | 
                    (df[date_columns[i+1]].isnull())).all()
                    for i in range(len(date_columns)-1)
                    if pd.to_datetime(df[date_columns[i]], errors='coerce').notnull().any()
                    and pd.to_datetime(df[date_columns[i+1]], errors='coerce').notnull().any()
                )
        
        # Apply consistency rules
        rule_results = {}
        for rule_name, rule_fn in consistency_rules.items():
            try:
                # Apply rule function
                is_consistent = rule_fn(df)
                
                rule_results[rule_name] = {
                    'is_consistent': bool(is_consistent),
                    'consistency_percentage': 100.0 if is_consistent else 0.0
                }
            except Exception as e:
                print(f"Error checking consistency rule '{rule_name}': {str(e)}")
except Exception as e:
                print(f"Error checking consistency rule '{rule_name}': {str(e)}")
        
        # Calculate overall consistency
        if rule_results:
            consistency_values = [r['consistency_percentage'] for r in rule_results.values()]
            overall_consistency = sum(consistency_values) / len(consistency_values)
        else:
            overall_consistency = 0
        
        return {
            'overall_consistency': float(overall_consistency),
            'rule_results': rule_results
        }
    
    def calculate_accuracy(self, df, reference_df=None, key_column=None, accuracy_rules=None):
        """
        Calculate accuracy metrics by comparing to reference data or rules
        
        Args:
            df: Pandas DataFrame to evaluate
            reference_df: Optional reference DataFrame for comparison
            key_column: Column to use for matching records between dataframes
            accuracy_rules: Dictionary mapping column names to accuracy checking functions
            
        Returns:
            Dictionary with accuracy metrics
        """
        column_accuracy = {}
        
        # Compare to reference data if provided
        if reference_df is not None and key_column is not None:
            # Ensure key column exists in both dataframes
            if key_column not in df.columns or key_column not in reference_df.columns:
                print(f"Key column '{key_column}' not found in both dataframes")
                return {'overall_accuracy': 0.0, 'column_accuracy': {}}
            
            # Merge dataframes on key column
            merged = pd.merge(df, reference_df, on=key_column, suffixes=('', '_ref'))
            
            # Calculate accuracy for each column
            for column in df.columns:
                if column == key_column:
                    continue
                
                ref_column = f"{column}_ref"
                if ref_column not in merged.columns:
                    continue
                
                # Count matching values
                matched_mask = (merged[column] == merged[ref_column]) | (merged[column].isnull() & merged[ref_column].isnull())
                match_count = matched_mask.sum()
                total_count = len(merged)
                accuracy = (match_count / total_count * 100) if total_count > 0 else 0
                
                column_accuracy[column] = {
                    'match_count': int(match_count),
                    'total_count': int(total_count),
                    'accuracy_percentage': float(accuracy)
                }
        
        # Apply accuracy rules if provided
        if accuracy_rules:
            for column, accuracy_fn in accuracy_rules.items():
                if column not in df.columns:
                    continue
                
                try:
                    # Apply accuracy function
                    accurate_mask = accuracy_fn(df[column])
                    accurate_count = accurate_mask.sum()
                    total_count = len(df)
                    accuracy = (accurate_count / total_count * 100) if total_count > 0 else 0
                    
                    column_accuracy[column] = {
                        'accurate_count': int(accurate_count),
                        'total_count': int(total_count),
                        'accuracy_percentage': float(accuracy)
                    }
                except Exception as e:
                    print(f"Error checking accuracy for column '{column}': {str(e)}")
        
        # Calculate overall accuracy
        if column_accuracy:
            accuracy_values = [a['accuracy_percentage'] for a in column_accuracy.values()]
            overall_accuracy = sum(accuracy_values) / len(accuracy_values)
        else:
            overall_accuracy = 0
        
        return {
            'overall_accuracy': float(overall_accuracy),
            'column_accuracy': column_accuracy
        }
    
    def calculate_timeliness(self, df, date_column, max_age=None):
        """
        Calculate timeliness metrics based on date field recency
        
        Args:
            df: Pandas DataFrame
            date_column: Column containing date values
            max_age: Optional maximum age in days for a record to be considered timely
            
        Returns:
            Dictionary with timeliness metrics
        """
        if date_column not in df.columns:
            print(f"Date column '{date_column}' not found")
            return {'timeliness_percentage': 0.0}
        
        try:
            # Convert to datetime
            dates = pd.to_datetime(df[date_column], errors='coerce')
            
            # Calculate age in days
            current_date = pd.Timestamp.now()
            age_days = (current_date - dates).dt.total_seconds() / (24 * 3600)
            
            # Calculate metrics
            if max_age is not None:
                timely_count = (age_days <= max_age).sum()
                total_count = dates.count()
                timeliness = (timely_count / total_count * 100) if total_count > 0 else 0
                
                return {
                    'timeliness_percentage': float(timeliness),
                    'timely_count': int(timely_count),
                    'total_count': int(total_count),
                    'max_age_days': float(max_age),
                    'oldest_record_days': float(age_days.max()) if not age_days.empty else None,
                    'newest_record_days': float(age_days.min()) if not age_days.empty else None,
                    'average_age_days': float(age_days.mean()) if not age_days.empty else None
                }
            else:
                # Without max_age, just return statistics
                return {
                    'oldest_record_days': float(age_days.max()) if not age_days.empty else None,
                    'newest_record_days': float(age_days.min()) if not age_days.empty else None,
                    'average_age_days': float(age_days.mean()) if not age_days.empty else None,
                    'median_age_days': float(age_days.median()) if not age_days.empty else None
                }
        
        except Exception as e:
            print(f"Error calculating timeliness: {str(e)}")
            return {'timeliness_percentage': 0.0, 'error': str(e)}
    
    def calculate_all_metrics(self, df, config=None):
        """
        Calculate all data quality metrics
        
        Args:
            df: Pandas DataFrame to evaluate
            config: Optional configuration dictionary with parameters for each metric
            
        Returns:
            Dictionary with all metrics
        """
        if config is None:
            config = {}
        
        # Calculate all metrics
        completeness = self.calculate_completeness(df)
        uniqueness = self.calculate_uniqueness(df)
        validity = self.calculate_validity(df, config.get('validation_rules'))
        consistency = self.calculate_consistency(df, config.get('consistency_rules'))
        
        # Accuracy requires reference data or rules
        if 'reference_df' in config or 'accuracy_rules' in config:
            accuracy = self.calculate_accuracy(
                df, 
                config.get('reference_df'), 
                config.get('key_column'),
                config.get('accuracy_rules')
            )
        else:
            accuracy = {'overall_accuracy': None, 'column_accuracy': {}}
        
        # Timeliness requires a date column
        if 'date_column' in config:
            timeliness = self.calculate_timeliness(
                df,
                config['date_column'],
                config.get('max_age')
            )
        else:
            timeliness = {'timeliness_percentage': None}
        
        # Calculate overall data quality score
        metrics = [
            completeness['overall_completeness'],
            uniqueness['row_uniqueness'],
            validity['overall_validity'],
            consistency['overall_consistency']
        ]
        
        if accuracy['overall_accuracy'] is not None:
            metrics.append(accuracy['overall_accuracy'])
        
        if timeliness.get('timeliness_percentage') is not None:
            metrics.append(timeliness['timeliness_percentage'])
        
        # Filter out None values
        metrics = [m for m in metrics if m is not None]
        overall_score = sum(metrics) / len(metrics) if metrics else 0
        
        # Compile all metrics
        result = {
            'timestamp': datetime.now().isoformat(),
            'overall_quality_score': float(overall_score),
            'row_count': len(df),
            'column_count': len(df.columns),
            'completeness': completeness,
            'uniqueness': uniqueness,
            'validity': validity,
            'consistency': consistency,
            'accuracy': accuracy,
            'timeliness': timeliness
        }
        
        # Store metrics if a metrics_db is set
        if self.metrics_db:
            # Generate a key for this dataset
            dataset_key = config.get('dataset_name', 'unnamed_dataset')
            
            # Add to historical metrics
            if dataset_key not in self.historical_metrics:
                self.historical_metrics[dataset_key] = []
            
            # Add summary of the current metrics
            self.historical_metrics[dataset_key].append({
                'timestamp': result['timestamp'],
                'overall_quality_score': result['overall_quality_score'],
                'row_count': result['row_count'],
                'completeness': result['completeness']['overall_completeness'],
                'uniqueness': result['uniqueness']['row_uniqueness'],
                'validity': result['validity']['overall_validity'],
                'consistency': result['consistency']['overall_consistency'],
                'accuracy': result['accuracy']['overall_accuracy'] if result['accuracy']['overall_accuracy'] is not None else None,
                'timeliness': result['timeliness'].get('timeliness_percentage')
            })
            
            # Save to file
            with open(self.metrics_db, 'w') as f:
                json.dump(self.historical_metrics, f, indent=2)
        
        return result
    
    def generate_data_quality_report(self, metrics, output_file=None):
        """
        Generate an HTML data quality report
        
        Args:
            metrics: Metrics dictionary from calculate_all_metrics
            output_file: Optional file path to save the report as HTML
        
        Returns:
            HTML report as a string
        """
        # Create visualizations
        import base64
        from io import BytesIO
        
        # Helper function to create a base64 encoded image
        def get_image_base64(plt_figure):
            buf = BytesIO()
            plt_figure.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode()
            plt.close(plt_figure)
            return img_str
        
        # Create a radar chart of main metrics
        def create_radar_chart():
            categories = ['Completeness', 'Uniqueness', 'Validity', 'Consistency']
            values = [
                metrics['completeness']['overall_completeness'],
                metrics['uniqueness']['row_uniqueness'],
                metrics['validity']['overall_validity'],
                metrics['consistency']['overall_consistency']
            ]
            
            # Add accuracy and timeliness if available
            if metrics['accuracy']['overall_accuracy'] is not None:
                categories.append('Accuracy')
                values.append(metrics['accuracy']['overall_accuracy'])
            
            if metrics['timeliness'].get('timeliness_percentage') is not None:
                categories.append('Timeliness')
                values.append(metrics['timeliness']['timeliness_percentage'])
            
            # Create the radar chart
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # Set the angles for each metric
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Add the values (and close the loop)
            values_for_plot = values + [values[0]]
            
            # Draw the chart
            ax.plot(angles, values_for_plot, linewidth=2, linestyle='solid')
            ax.fill(angles, values_for_plot, alpha=0.25)
            
            # Set the labels
            plt.xticks(angles[:-1], categories)
            
            # Set y-axis limits
            ax.set_ylim(0, 100)
            plt.yticks([20, 40, 60, 80, 100], ['20', '40', '60', '80', '100'], color='gray', size=10)
            plt.ylim(0, 100)
            
            plt.title('Data Quality Dimensions', size=15, y=1.1)
            
            return get_image_base64(fig)
        
        # Create column completeness chart
        def create_completeness_chart():
            column_completeness = metrics['completeness']['column_completeness']
            columns = list(column_completeness.keys())
            completeness_values = [column_completeness[col]['completeness_percentage'] for col in columns]
            
            # Sort by completeness
            sorted_indices = np.argsort(completeness_values)
            sorted_columns = [columns[i] for i in sorted_indices]
            sorted_values = [completeness_values[i] for i in sorted_indices]
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(10, max(6, len(columns) * 0.4)))
            bars = ax.barh(sorted_columns, sorted_values, color='skyblue')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(min(width + 1, 105), bar.get_y() + bar.get_height()/2,
                       f'{width:.1f}%', ha='left', va='center')
            
            ax.set_xlim(0, 105)
            ax.set_xlabel('Completeness (%)')
            ax.set_title('Column Completeness')
            
            plt.tight_layout()
            return get_image_base64(fig)
        
        # Build the HTML report
        html = f"""
        <html>
        <head>
            <title>Data Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .good {{ background-color: #c8e6c9; }}
                .moderate {{ background-color: #fff9c4; }}
                .poor {{ background-color: #ffcdd2; }}
                .metric-box {{ float: left; width: 31%; margin: 1%; padding: 10px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.1); }}
                .clearfix::after {{ content: ""; clear: both; display: table; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                .score {{ font-size: 24px; font-weight: bold; text-align: center; margin: 10px 0; }}
                .good-score {{ color: #2e7d32; }}
                .moderate-score {{ color: #f9a825; }}
                .poor-score {{ color: #c62828; }}
            </style>
        </head>
        <body>
            <h1>Data Quality Report</h1>
            <p>Generated: {metrics['timestamp']}</p>
            <p>Dataset: {len(metrics['completeness']['column_completeness'])} columns, {metrics['row_count']} rows</p>
            
            <div class="chart">
                <h2>Overall Data Quality Score</h2>
                <div class="score {
                    'good-score' if metrics['overall_quality_score'] >= 90 else
                    'moderate-score' if metrics['overall_quality_score'] >= 70 else
                    'poor-score'
                }">
                    {metrics['overall_quality_score']:.1f}%
                </div>
                <img src="data:image/png;base64,{create_radar_chart()}" width="500">
            </div>
            
            <div class="clearfix">
                <div class="metric-box {
                    'good' if metrics['completeness']['overall_completeness'] >= 90 else
                    'moderate' if metrics['completeness']['overall_completeness'] >= 70 else
                    'poor'
                }">
                    <h3>Completeness</h3>
                    <div class="score">{metrics['completeness']['overall_completeness']:.1f}%</div>
                    <p>Measures the presence of required data</p>
                </div>
                
                <div class="metric-box {
                    'good' if metrics['uniqueness']['row_uniqueness'] >= 90 else
                    'moderate' if metrics['uniqueness']['row_uniqueness'] >= 70 else
                    'poor'
                }">
                    <h3>Uniqueness</h3>
                    <div class="score">{metrics['uniqueness']['row_uniqueness']:.1f}%</div>
                    <p>Measures absence of duplicates</p>
                </div>
                
                <div class="metric-box {
                    'good' if metrics['validity']['overall_validity'] >= 90 else
                    'moderate' if metrics['validity']['overall_validity'] >= 70 else
                    'poor'
                }">
                    <h3>Validity</h3>
                    <div class="score">{metrics['validity']['overall_validity']:.1f}%</div>
                    <p>Measures adherence to data rules</p>
                </div>
            </div>
            
            <div class="clearfix">
                <div class="metric-box {
                    'good' if metrics['consistency']['overall_consistency'] >= 90 else
                    'moderate' if metrics['consistency']['overall_consistency'] >= 70 else
                    'poor'
                }">
                    <h3>Consistency</h3>
                    <div class="score">{metrics['consistency']['overall_consistency']:.1f}%</div>
                    <p>Measures internal data consistency</p>
                </div>
        """
        
        # Add accuracy and timeliness if available
        if metrics['accuracy']['overall_accuracy'] is not None:
            accuracy_class = (
                'good' if metrics['accuracy']['overall_accuracy'] >= 90 else
                'moderate' if metrics['accuracy']['overall_accuracy'] >= 70 else
                'poor'
            )
            html += f"""
                <div class="metric-box {accuracy_class}">
                    <h3>Accuracy</h3>
                    <div class="score">{metrics['accuracy']['overall_accuracy']:.1f}%</div>
                    <p>Measures correctness of data values</p>
                </div>
            """
        
        if metrics['timeliness'].get('timeliness_percentage') is not None:
            timeliness_class = (
                'good' if metrics['timeliness']['timeliness_percentage'] >= 90 else
                'moderate' if metrics['timeliness']['timeliness_percentage'] >= 70 else
                'poor'
            )
            html += f"""
                <div class="metric-box {timeliness_class}">
                    <h3>Timeliness</h3>
                    <div class="score">{metrics['timeliness']['timeliness_percentage']:.1f}%</div>
                    <p>Measures recency of data</p>
                </div>
            """
        
        html += """
            </div>
            
            <h2>Column Completeness</h2>
            <div class="chart">
                <img src="data:image/png;base64,{}" width="700">
            </div>
        """.format(create_completeness_chart())
        
        # Add detail tables
        html += """
            <h2>Detailed Metrics</h2>
            
            <h3>Completeness by Column</h3>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Non-Null Count</th>
                    <th>Total Count</th>
                    <th>Completeness</th>
                </tr>
        """
        
        # Add completeness details
        for column, details in metrics['completeness']['column_completeness'].items():
            completeness = details['completeness_percentage']
            row_class = (
                'good' if completeness >= 95 else
                'moderate' if completeness >= 80 else
                'poor'
            )
            html += f"""
                <tr class="{row_class}">
                    <td>{column}</td>
                    <td>{details['non_null_count']}</td>
                    <td>{details['total_count']}</td>
                    <td>{completeness:.1f}%</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h3>Uniqueness by Column</h3>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Unique Values</th>
                    <th>Non-Null Count</th>
                    <th>Uniqueness</th>
                </tr>
        """
        
        # Add uniqueness details
        for column, details in metrics['uniqueness']['column_uniqueness'].items():
            uniqueness = details['uniqueness_percentage']
            row_class = 'good'  # Uniqueness is context-dependent, so we don't use colors
            html += f"""
                <tr>
                    <td>{column}</td>
                    <td>{details['unique_count']}</td>
                    <td>{details['non_null_count']}</td>
                    <td>{uniqueness:.1f}%</td>
                </tr>
            """
        
        html += """
            </table>
        """
        
        # Add validity details if available
        if metrics['validity']['column_validity']:
            html += """
                <h3>Validity by Column</h3>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Valid Count</th>
                        <th>Total Count</th>
                        <th>Validity</th>
                    </tr>
            """
            
            for column, details in metrics['validity']['column_validity'].items():
                validity = details['validity_percentage']
                row_class = (
                    'good' if validity >= 95 else
                    'moderate' if validity >= 80 else
                    'poor'
                )
                html += f"""
                    <tr class="{row_class}">
                        <td>{column}</td>
                        <td>{details['valid_count']}</td>
                        <td>{details['total_count']}</td>
                        <td>{validity:.1f}%</td>
                    </tr>
                """
            
            html += """
                </table>
            """
        
        # End HTML
        html += """
        </body>
        </html>
        """
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html)
            print(f"Data quality report saved to {output_file}")
        
        return html
    
    def plot_metrics_history(self, dataset_name, metric_name=None, start_date=None, end_date=None):
        """
        Plot historical metrics for a dataset
        
        Args:
            dataset_name: Name of the dataset to plot metrics for
            metric_name: Optional name of specific metric to plot
            start_date: Optional start date for filtering (ISO format string)
            end_date: Optional end date for filtering (ISO format string)
            
        Returns:
            Matplotlib figure
        """
        if not self.historical_metrics or dataset_name not in self.historical_metrics:
            print(f"No historical metrics found for dataset '{dataset_name}'")
            return None
        
        # Get metrics history for the dataset
        metrics_history = self.historical_metrics[dataset_name]
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(metrics_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Apply date filters if specified
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['timestamp'] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df['timestamp'] <= end_dt]
        
        if df.empty:
            print("No data to plot after applying filters")
            return None
        
        # Create plot
        if metric_name:
            # Plot a single metric
            if metric_name not in df.columns:
                print(f"Metric '{metric_name}' not found in history")
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['timestamp'], df[metric_name], marker='o', linestyle='-')
            ax.set_title(f'{metric_name.title()} History for {dataset_name}')
            ax.set_xlabel('Date')
            ax.set_ylabel(f'{metric_name.title()} (%)')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set y-axis limits
            ax.set_ylim(0, 105)
            
            # Format x-axis dates
            fig.autofmt_xdate()
            
            # Add trend line
            if len(df) > 1:
                z = np.polyfit(range(len(df)), df[metric_name], 1)
                p = np.poly1d(z)
                ax.plot(df['timestamp'], p(range(len(df))), "r--", alpha=0.8)
            
            plt.tight_layout()
            return fig
        else:
            # Plot all available metrics
            metrics = ['overall_quality_score', 'completeness', 'uniqueness', 
                      'validity', 'consistency', 'accuracy', 'timeliness']
            
            # Filter out metrics that are not available
            available_metrics = [m for m in metrics if m in df.columns and not df[m].isna().all()]
            
            if not available_metrics:
                print("No metrics available to plot")
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for metric in available_metrics:
                if df[metric].notna().any():  # Only plot if there are non-NA values
                    ax.plot(df['timestamp'], df[metric], marker='o', linestyle='-', label=metric.title())
            
            ax.set_title(f'Data Quality Metrics History for {dataset_name}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Score (%)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Set y-axis limits
            ax.set_ylim(0, 105)
            
            # Format x-axis dates
            fig.autofmt_xdate()
            
            plt.tight_layout()
            return fig
```

## Mini-Project: Data Quality Pipeline

Let's combine what we've learned to create a complete data quality pipeline:

```python
import pandas as pd
import numpy as np
import os
import json
import logging
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Import our data quality modules
# Assume that the classes we defined earlier are in a package called "data_quality"
from data_quality.profiling import profile_dataset, identify_quality_issues
from data_quality.validation import DataValidator, ValidationRule
from data_quality.cleansing import DataCleanser, CleansingRule
from data_quality.standardization import (standardize_names, standardize_addresses, 
                                        standardize_company_names, standardize_phone_numbers)
from data_quality.duplicate_detection import identify_exact_duplicates, identify_fuzzy_duplicates, merge_duplicate_records
from data_quality.metrics import DataQualityMetrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='data_quality_pipeline.log'
)
logger = logging.getLogger('data_quality_pipeline')

class DataQualityPipeline:
    """
    End-to-end data quality pipeline that combines profiling, validation,
    cleansing, standardization, deduplication, and quality monitoring.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the data quality pipeline
        
        Args:
            config_file: Path to configuration JSON file
        """
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.validator = DataValidator()
        self.cleanser = DataCleanser()
        self.metrics = DataQualityMetrics(metrics_db=self.config.get('metrics_db'))
        
        # Set up output directory
        self.output_dir = self.config.get('output_directory', 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("Data Quality Pipeline initialized")
    
    def _load_config(self, config_file):
        """Load configuration from file or use defaults"""
        default_config = {
            'metrics_db': 'data_quality_metrics.json',
            'output_directory': 'output',
            'email_notifications': {
                'enabled': False,
                'sender': 'dataquality@example.com',
                'recipients': [],
                'smtp_server': 'smtp.example.com',
                'smtp_port': 587,
                'username': '',
                'password': ''
            },
            'quality_thresholds': {
                'overall': 85,
                'completeness': 90,
                'uniqueness': 95,
                'validity': 85,
                'consistency': 90
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Merge with defaults to ensure all keys exist
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict) and key in config:
                        # For nested dictionaries, merge them too
                        for subkey, subvalue in value.items():
                            if subkey not in config[key]:
                                config[key][subkey] = subvalue
                
                logger.info(f"Configuration loaded from {config_file}")
                return config
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                return default_config
        else:
            logger.info("Using default configuration")
            return default_config
    
    def process_dataset(self, input_file, dataset_name=None, schema_file=None, rules_file=None):
def process_dataset(self, input_file, dataset_name=None, schema_file=None, rules_file=None):
        """
        Process a dataset through the data quality pipeline
        
        Args:
            input_file: Path to input CSV/Excel file
            dataset_name: Optional name for the dataset (defaults to filename)
            schema_file: Optional path to schema definition JSON file
            rules_file: Optional path to business rules JSON file
            
        Returns:
            Dictionary with processing results and metrics
        """
        start_time = datetime.now()
        
        # Generate dataset name from filename if not provided
        if not dataset_name:
            dataset_name = os.path.splitext(os.path.basename(input_file))[0]
        
        logger.info(f"Processing dataset: {dataset_name} from {input_file}")
        
        # Load the data
        try:
            if input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
            elif input_file.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(input_file)
            else:
                raise ValueError(f"Unsupported file format: {input_file}")
            
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return {'success': False, 'error': f"Error loading data: {str(e)}"}
        
        # Make a copy of the original data for comparison
        original_df = df.copy()
        
        # Step 1: Profile the data
        logger.info("Profiling data...")
        profile = profile_dataset(df)
        issues = identify_quality_issues(df)
        
        # Save profile report
        profile_file = os.path.join(self.output_dir, f"{dataset_name}_profile.html")
        
        # Step 2: Validate the data
        logger.info("Validating data...")
        
        # Load schema if provided
        schema = None
        if schema_file and os.path.exists(schema_file):
            try:
                with open(schema_file, 'r') as f:
                    schema = json.load(f)
                logger.info(f"Loaded schema from {schema_file}")
            except Exception as e:
                logger.error(f"Error loading schema: {str(e)}")
                schema = None
        
        # Generate schema from data if not provided
        if not schema:
            logger.info("Generating schema from data...")
            schema = self.validator.generate_schema_from_dataframe(df)
            
            # Save generated schema
            schema_output = os.path.join(self.output_dir, f"{dataset_name}_schema.json")
            with open(schema_output, 'w') as f:
                json.dump(schema, f, indent=2)
            logger.info(f"Saved generated schema to {schema_output}")
        
        # Validate against schema
        validation_results = self.validator.validate_dataframe(df, schema)
        
        # Generate validation report
        validation_file = os.path.join(self.output_dir, f"{dataset_name}_validation.html")
        self.validator.generate_validation_report(validation_results, validation_file)
        logger.info(f"Saved validation report to {validation_file}")
        
        # Step 3: Apply business rules
        logger.info("Applying business rules...")
        
        # Load business rules if provided
        business_rules = []
        if rules_file and os.path.exists(rules_file):
            try:
                with open(rules_file, 'r') as f:
                    rule_configs = json.load(f)
                
                # TODO: Parse and register business rules
                logger.info(f"Loaded business rules from {rules_file}")
            except Exception as e:
                logger.error(f"Error loading business rules: {str(e)}")
        
        # Step 4: Cleanse and standardize the data
        logger.info("Cleansing and standardizing data...")
        
        # Generate cleansing plan based on profile and validation results
        cleansing_plan = self._generate_cleansing_plan(df, profile, validation_results)
        
        # Apply cleansing
        cleansed_df, cleansing_log = self.cleanser.cleanse_dataframe(df, cleansing_plan)
        
        # Generate cleansing report
        cleansing_file = os.path.join(self.output_dir, f"{dataset_name}_cleansing.html")
        self.cleanser.generate_cleansing_report(cleansing_log, cleansing_file)
        logger.info(f"Saved cleansing report to {cleansing_file}")
        
        # Step 5: Detect and handle duplicates
        logger.info("Detecting duplicates...")
        
        # Identify exact duplicates
        duplicate_subset = self.config.get('duplicate_detection', {}).get('columns')
        duplicate_mask, duplicate_groups, duplicate_stats = identify_exact_duplicates(cleansed_df, subset=duplicate_subset)
        
        # Handle duplicates if found
        if duplicate_stats['duplicate_rows'] > 0:
            logger.info(f"Found {duplicate_stats['duplicate_rows']} duplicate rows in {duplicate_stats['unique_groups']} groups")
            
            # Apply duplicate merging strategy
            merge_strategy = self.config.get('duplicate_handling', {}).get('strategy', {})
            deduplicated_df = merge_duplicate_records(cleansed_df, duplicate_groups, merge_strategy)
            
            # Save duplicates report
            duplicates_file = os.path.join(self.output_dir, f"{dataset_name}_duplicates.csv")
            cleansed_df[duplicate_mask].to_csv(duplicates_file, index=False)
            logger.info(f"Saved duplicates to {duplicates_file}")
        else:
            logger.info("No duplicates found")
            deduplicated_df = cleansed_df
        
        # Step 6: Calculate data quality metrics
        logger.info("Calculating data quality metrics...")
        
        # Configure metrics calculation
        metrics_config = {
            'dataset_name': dataset_name,
            'validation_rules': self.config.get('validation_rules'),
            'consistency_rules': self.config.get('consistency_rules'),
            'accuracy_rules': self.config.get('accuracy_rules')
        }
        
        # Add timeliness configuration if date column is specified
        if 'timeliness' in self.config and 'date_column' in self.config['timeliness']:
            metrics_config['date_column'] = self.config['timeliness']['date_column']
            metrics_config['max_age'] = self.config['timeliness'].get('max_age')
        
        # Calculate metrics
        quality_metrics = self.metrics.calculate_all_metrics(deduplicated_df, metrics_config)
        
        # Generate metrics report
        metrics_file = os.path.join(self.output_dir, f"{dataset_name}_quality_report.html")
        self.metrics.generate_data_quality_report(quality_metrics, metrics_file)
        logger.info(f"Saved quality metrics report to {metrics_file}")
        
        # Step 7: Save the processed data
        output_file = os.path.join(self.output_dir, f"{dataset_name}_processed.csv")
        deduplicated_df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
        
        # Step 8: Check quality against thresholds and send notifications if needed
        quality_summary = self._check_quality_thresholds(quality_metrics)
        
        if not quality_summary['passed']:
            logger.warning(f"Data quality below thresholds: {quality_summary['failures']}")
            
            # Send email notification if configured
            if self.config['email_notifications']['enabled']:
                self._send_quality_alert(dataset_name, quality_metrics, quality_summary)
        
        # Calculate processing stats
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Compile result summary
        result = {
            'success': True,
            'dataset_name': dataset_name,
            'input_file': input_file,
            'output_file': output_file,
            'original_rows': len(original_df),
            'processed_rows': len(deduplicated_df),
            'quality_metrics': {
                'overall_quality_score': quality_metrics['overall_quality_score'],
                'completeness': quality_metrics['completeness']['overall_completeness'],
                'uniqueness': quality_metrics['uniqueness']['row_uniqueness'],
                'validity': quality_metrics['validity']['overall_validity'],
                'consistency': quality_metrics['consistency']['overall_consistency']
            },
            'quality_passed': quality_summary['passed'],
            'quality_failures': quality_summary.get('failures', []),
            'data_changes': {
                'fields_cleansed': len(cleansing_log),
                'duplicates_removed': duplicate_stats['duplicate_rows'],
                'duplicate_groups': duplicate_stats['unique_groups']
            },
            'processing_time': processing_time,
            'reports': {
                'profile': profile_file,
                'validation': validation_file,
                'cleansing': cleansing_file,
                'quality': metrics_file
            }
        }
        
        logger.info(f"Completed processing dataset: {dataset_name}")
        return result
    
    def _generate_cleansing_plan(self, df, profile, validation_results):
        """Generate a cleansing plan based on profiling and validation results"""
        cleansing_plan = []
        
        # Get invalid columns from validation results
        invalid_columns = validation_results['summary'].get('invalid_columns', [])
        
        # Add rules based on profile findings
        
        # 1. Trim whitespace for text columns
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        if text_columns:
            cleansing_plan.append({
                'rule_name': 'trim_whitespace',
                'columns': text_columns
            })
        
        # 2. Fill missing values based on validation failures
        for column in invalid_columns:
            # Get column type from validation results
            column_result = validation_results['column_results'].get(column, {})
            column_type = column_result.get('type') if 'type' in column_result else None
            
            if column_type == 'string':
                cleansing_plan.append({
                    'rule_name': 'fill_missing',
                    'columns': [column],
                    'default_value': ''  # Empty string for string columns
                })
            elif column_type in ['integer', 'float']:
                cleansing_plan.append({
                    'rule_name': 'fill_missing',
                    'columns': [column],
                    'default_value': 0  # Zero for numeric columns
                })
            elif column_type == 'boolean':
                cleansing_plan.append({
                    'rule_name': 'fill_missing',
                    'columns': [column],
                    'default_value': False  # False for boolean columns
                })
        
        # 3. Normalize spaces in text fields
        if text_columns:
            cleansing_plan.append({
                'rule_name': 'normalize_spaces',
                'columns': text_columns
            })
        
        # 4. Standardize phone numbers if detected
        phone_columns = [col for col in df.columns if 'phone' in col.lower()]
        if phone_columns:
            cleansing_plan.append({
                'rule_name': 'standardize_phone',
                'columns': phone_columns
            })
        
        # 5. Standardize dates if detected
        date_columns = []
        for col in df.columns:
            try:
                if pd.to_datetime(df[col], errors='coerce').notnull().any():
                    date_columns.append(col)
            except:
                pass
        
        if date_columns:
            cleansing_plan.append({
                'rule_name': 'standardize_date',
                'columns': date_columns,
                'output_format': '%Y-%m-%d'
            })
        
        return cleansing_plan
    
    def _check_quality_thresholds(self, quality_metrics):
        """Check quality metrics against configured thresholds"""
        thresholds = self.config.get('quality_thresholds', {})
        failures = []
        
        # Check overall quality score
        if quality_metrics['overall_quality_score'] < thresholds.get('overall', 85):
            failures.append({
                'metric': 'overall_quality_score',
                'value': quality_metrics['overall_quality_score'],
                'threshold': thresholds.get('overall', 85)
            })
        
        # Check completeness
        if quality_metrics['completeness']['overall_completeness'] < thresholds.get('completeness', 90):
            failures.append({
                'metric': 'completeness',
                'value': quality_metrics['completeness']['overall_completeness'],
                'threshold': thresholds.get('completeness', 90)
            })
        
        # Check uniqueness
        if quality_metrics['uniqueness']['row_uniqueness'] < thresholds.get('uniqueness', 95):
            failures.append({
                'metric': 'uniqueness',
                'value': quality_metrics['uniqueness']['row_uniqueness'],
                'threshold': thresholds.get('uniqueness', 95)
            })
        
        # Check validity
        if quality_metrics['validity']['overall_validity'] < thresholds.get('validity', 85):
            failures.append({
                'metric': 'validity',
                'value': quality_metrics['validity']['overall_validity'],
                'threshold': thresholds.get('validity', 85)
            })
        
        # Check consistency
        if quality_metrics['consistency']['overall_consistency'] < thresholds.get('consistency', 90):
            failures.append({
                'metric': 'consistency',
                'value': quality_metrics['consistency']['overall_consistency'],
                'threshold': thresholds.get('consistency', 90)
            })
        
        return {
            'passed': len(failures) == 0,
            'failures': failures
        }
    
    def _send_quality_alert(self, dataset_name, quality_metrics, quality_summary):
        """Send email notification for data quality issues"""
        try:
            email_config = self.config['email_notifications']
            
            if not email_config.get('enabled') or not email_config.get('recipients'):
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config['sender']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"Data Quality Alert: {dataset_name}"
            
            # Create HTML content
            html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    .failure {{ background-color: #ffcdd2; }}
                </style>
            </head>
            <body>
                <h1>Data Quality Alert</h1>
                <p>Dataset: <strong>{dataset_name}</strong></p>
                <p>Overall Quality Score: <strong>{quality_metrics['overall_quality_score']:.2f}%</strong></p>
                
                <h2>Quality Thresholds Failures</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Threshold</th>
                        <th>Difference</th>
                    </tr>
            """
            
            for failure in quality_summary['failures']:
                difference = failure['value'] - failure['threshold']
                html += f"""
                    <tr class="failure">
                        <td>{failure['metric'].replace('_', ' ').title()}</td>
                        <td>{failure['value']:.2f}%</td>
                        <td>{failure['threshold']:.2f}%</td>
                        <td>{difference:.2f}%</td>
                    </tr>
                """
            
            html += """
                </table>
                
                <p>Please review the data quality reports for more information.</p>
            </body>
            </html>
            """
            
            # Attach HTML content
            msg.attach(MIMEText(html, 'html'))
            
            # Send the email
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                if email_config.get('username') and email_config.get('password'):
                    server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
            
            logger.info(f"Sent quality alert email for {dataset_name}")
        
        except Exception as e:
            logger.error(f"Error sending quality alert email: {str(e)}")

def main():
    """Main entry point for the data quality pipeline"""
    parser = argparse.ArgumentParser(description='Data Quality Pipeline')
    parser.add_argument('--input', required=True, help='Input file path')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--schema', help='Schema definition file path')
    parser.add_argument('--rules', help='Business rules file path')
    parser.add_argument('--name', help='Dataset name')
    args = parser.parse_args()
    
    # Initialize the pipeline
    pipeline = DataQualityPipeline(config_file=args.config)
    
    # Process the dataset
    result = pipeline.process_dataset(
        input_file=args.input,
        dataset_name=args.name,
        schema_file=args.schema,
        rules_file=args.rules
    )
    
    # Print result summary
    if result['success']:
        print(f"\nSuccessfully processed dataset: {result['dataset_name']}")
        print(f"- Input rows: {result['original_rows']}")
        print(f"- Output rows: {result['processed_rows']}")
        print(f"- Overall quality score: {result['quality_metrics']['overall_quality_score']:.2f}%")
        print(f"- Processing time: {result['processing_time']:.2f} seconds")
        print("\nOutput files:")
        for report_name, report_path in result['reports'].items():
            print(f"- {report_name.title()} report: {report_path}")
        print(f"- Processed data: {result['output_file']}")
        
        if not result['quality_passed']:
            print("\nWarning: Data quality below thresholds!")
            for failure in result['quality_failures']:
                print(f"- {failure['metric']}: {failure['value']:.2f}% (threshold: {failure['threshold']:.2f}%)")
    else:
        print(f"Error processing dataset: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
```

## Using the Data Quality Pipeline

Here's how to use the data quality pipeline in practice:

1. **Create a configuration file** (config.json):
```json
{
  "metrics_db": "data_quality_metrics.json",
  "output_directory": "dq_output",
  "email_notifications": {
    "enabled": true,
    "sender": "dataquality@yourcompany.com",
    "recipients": ["data.team@yourcompany.com"],
    "smtp_server": "smtp.yourcompany.com",
    "smtp_port": 587,
    "username": "dq_notifier",
    "password": "your_password"
  },
  "quality_thresholds": {
    "overall": 85,
    "completeness": 90,
    "uniqueness": 95,
    "validity": 85,
    "consistency": 90
  },
  "duplicate_detection": {
    "columns": ["customer_id", "email"]
  },
  "timeliness": {
    "date_column": "last_updated",
    "max_age": 30
  }
}
```

2. **Define a schema file** (schema.json):
```json
{
  "customer_id": "integer",
  "name": "string",
  "email": "email",
  "phone": "phone",
  "address": "string",
  "city": "string",
  "state": "string",
  "zip": "zipcode",
  "industry": "string",
  "revenue": "currency",
  "employee_count": "integer",
  "last_updated": "date"
}
```

3. **Run the pipeline**:
```bash
python data_quality_pipeline.py --input customer_data.csv --config config.json --schema schema.json --name "Customer Data"
```

4. **Review the output reports**:
- Profile report: Shows statistics and distributions of your data
- Validation report: Lists validation failures against your schema
- Cleansing report: Details what was changed during data cleansing
- Quality report: Provides comprehensive data quality metrics
- Processed data: The cleansed, deduplicated, and standardized dataset

## Next Steps

After mastering these data quality and validation techniques, you'll be ready to:

1. Develop custom validation rules for your organization's unique data
2. Create automated data quality monitoring workflows
3. Build dashboards to track data quality metrics over time
4. Implement data governance frameworks with quality checks
5. Design ETL pipelines with integrated quality validation

## Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Great Expectations](https://greatexpectations.io/) - Data validation framework
- [Deequ](https://github.com/awslabs/deequ) - Data quality validation for big data
- [Data Management Body of Knowledge (DMBOK)](https://www.dama.org/cpages/body-of-knowledge)
- [The Data Warehouse Toolkit](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/books/data-warehouse-dw-toolkit/)

## Exercises and Projects

For additional practice, try these exercises:

1. Develop a custom validation framework for a specific data domain
2. Create a data quality dashboard for monitoring metrics over time
3. Build a data cleansing pipeline for a messy dataset
4. Implement duplicate detection for customer records
5. Design a data governance workflow with quality checkpoints

## Contributing

If you've found this guide helpful, consider contributing:
- Add examples for handling industry-specific data formats
- Share custom validation rules for common data domains
- Suggest improvements or corrections

Happy data quality monitoring!
