# Data Analysis & Manipulation for Data Professionals

Welcome to the Data Analysis & Manipulation module! This guide focuses on using Python's powerful data analysis libraries to perform effective data transformation, cleaning, and analysis - skills that are essential for data governance and analytics professionals.

## Why Data Analysis & Manipulation Matter

Data rarely comes in the exact format needed for analysis. This module teaches you how to:
- Load data from various sources into Python
- Clean and transform messy datasets
- Extract valuable insights through analysis
- Identify and address data quality issues
- Automate repetitive data preparation tasks
- Create reproducible data pipelines

## Module Overview

This module focuses on practical data manipulation skills:

1. [NumPy Fundamentals](#numpy-fundamentals)
2. [Pandas Basics](#pandas-basics)
3. [Data Loading & Inspection](#data-loading--inspection)
4. [Data Cleaning](#data-cleaning)
5. [Data Transformation](#data-transformation)
6. [Grouping & Aggregation](#grouping--aggregation)
7. [Working with Missing Data](#working-with-missing-data)
8. [Time Series Analysis](#time-series-analysis)
9. [Mini-Project: Data Quality Report](#mini-project-data-quality-report)

## NumPy Fundamentals

NumPy provides the foundation for data analysis in Python:

```python
import numpy as np

# Creating arrays
data = np.array([15, 23, 42, 57, 89, 12])

# Basic statistics
average = np.mean(data)
median = np.median(data)
std_dev = np.std(data)
min_val = np.min(data)
max_val = np.max(data)

# Array operations
scaled_data = data / 100  # Element-wise division
filtered_data = data[data > 30]  # Boolean indexing

# Creating a matrix (2D array)
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Accessing elements
first_row = matrix[0]  # [1, 2, 3]
element = matrix[1, 2]  # 6 (row 1, column 2)

# Array dimensions and shape
rows, cols = matrix.shape  # (3, 3)
```

## Pandas Basics

Pandas is the most important library for data analysis:

```python
import pandas as pd

# Creating a DataFrame
data = {
    'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
    'company': ['Acme Corp', 'TechSolutions', 'GlobalServices', 'DataDynamics', 'Innovatech'],
    'industry': ['Manufacturing', 'Technology', 'Consulting', 'Technology', 'Healthcare'],
    'revenue': [1500000, 2750000, 850000, 1250000, 3100000],
    'employees': [250, 180, 90, 115, 320],
    'active': [True, True, False, True, True]
}

df = pd.DataFrame(data)

# Examining the DataFrame
df.head()  # View first 5 rows
df.info()  # Summary of DataFrame columns, types, and missing values
df.describe()  # Statistical summary of numeric columns

# Accessing data
companies = df['company']  # Select a column
tech_companies = df[df['industry'] == 'Technology']  # Filter rows
first_row = df.iloc[0]  # Select by position
customer_c003 = df.loc[df['customer_id'] == 'C003']  # Select by label/condition

# Basic operations
df['revenue_millions'] = df['revenue'] / 1000000  # Create a new column
sorted_df = df.sort_values('revenue', ascending=False)  # Sort
```

## Data Loading & Inspection

Loading data from various sources:

```python
# From CSV
customers_df = pd.read_csv('customer_data.csv')

# From Excel
financial_df = pd.read_excel('financial_data.xlsx', sheet_name='Q2_2024')

# From SQL database
import sqlite3
conn = sqlite3.connect('company_database.db')
query = "SELECT * FROM customers WHERE active = 1"
active_customers_df = pd.read_sql(query, conn)

# From JSON
config_df = pd.read_json('configuration.json')

# Initial data inspection
def inspect_dataset(df, name="Dataset"):
    """Perform initial inspection of a dataset"""
    print(f"=== {name} Inspection ===")
    print(f"Shape: {df.shape} (rows, columns)")
    print("\nColumn information:")
    
    for col in df.columns:
        missing = df[col].isna().sum()
        missing_pct = (missing / len(df)) * 100
        unique = df[col].nunique()
        dtype = df[col].dtype
        
        print(f"- {col}: {dtype}, {unique} unique values, {missing_pct:.1f}% missing")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates} ({duplicates/len(df)*100:.1f}%)")
    
    # Sample data
    print("\nSample data:")
    print(df.head(3))
```

## Data Cleaning

Common data cleaning operations:

```python
# Handling missing values
df_cleaned = df.dropna(subset=['customer_id'])  # Drop rows with missing IDs
df['revenue'] = df['revenue'].fillna(0)  # Fill missing values
df['industry'] = df['industry'].fillna('Unknown')  # Fill with default

# Removing duplicates
df_unique = df.drop_duplicates(subset=['customer_id'])  # Drop duplicate customers

# Fixing data types
df['customer_id'] = df['customer_id'].astype(str)
df['join_date'] = pd.to_datetime(df['join_date'])
df['active'] = df['active'].astype(bool)

# Standardizing text data
df['company'] = df['company'].str.strip()  # Remove whitespace
df['industry'] = df['industry'].str.upper()  # Standardize case

# Handling outliers
q1 = df['revenue'].quantile(0.25)
q3 = df['revenue'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

outliers = df[(df['revenue'] < lower_bound) | (df['revenue'] > upper_bound)]
df_no_outliers = df[(df['revenue'] >= lower_bound) & (df['revenue'] <= upper_bound)]

# Creating a data cleaning function
def clean_customer_data(df):
    """Clean and standardize customer data"""
    # Make a copy to avoid modifying the original
    cleaned = df.copy()
    
    # Fix data types
    cleaned['customer_id'] = cleaned['customer_id'].astype(str)
    
    # Clean text fields
    text_columns = ['company', 'industry', 'contact_name']
    for col in text_columns:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].str.strip()
    
    # Standardize industry values
    if 'industry' in cleaned.columns:
        # Map variations to standard values
        industry_mapping = {
            'tech': 'Technology',
            'technology': 'Technology',
            'software': 'Technology',
            'it': 'Technology',
            'health': 'Healthcare',
            'healthcare': 'Healthcare',
            'medical': 'Healthcare',
            'manufacturing': 'Manufacturing',
            'retail': 'Retail',
            'finance': 'Financial Services',
            'financial': 'Financial Services',
            'banking': 'Financial Services'
        }
        
        cleaned['industry'] = cleaned['industry'].str.lower()
        cleaned['industry'] = cleaned['industry'].map(industry_mapping).fillna(cleaned['industry'])
        cleaned['industry'] = cleaned['industry'].str.title()
    
    # Handle missing values
    cleaned['active'] = cleaned['active'].fillna(False)
    
    # Remove duplicates
    cleaned = cleaned.drop_duplicates(subset=['customer_id'])
    
    return cleaned
```

## Data Transformation

Reshaping and transforming data:

```python
# Creating new columns based on existing data
df['revenue_per_employee'] = df['revenue'] / df['employees']
df['size_category'] = pd.cut(df['employees'], 
                           bins=[0, 50, 250, 1000, float('inf')],
                           labels=['Small', 'Medium', 'Large', 'Enterprise'])

# Applying functions to columns
def categorize_customer(row):
    """Categorize customers based on revenue and employee count"""
    if row['revenue'] > 5000000 or row['employees'] > 1000:
        return 'Strategic'
    elif row['revenue'] > 1000000 or row['employees'] > 200:
        return 'Key Account'
    else:
        return 'Standard'

df['customer_category'] = df.apply(categorize_customer, axis=1)

# Pivot tables
industry_size_pivot = df.pivot_table(
    values='revenue',
    index='industry',
    columns='size_category',
    aggfunc='sum',
    fill_value=0
)

# Melting from wide to long format
# Starting with a DataFrame having quarterly revenues
quarterly_data = pd.DataFrame({
    'customer_id': ['C001', 'C002', 'C003'],
    'company': ['Acme Corp', 'TechSolutions', 'GlobalServices'],
    'Q1_revenue': [350000, 420000, 180000],
    'Q2_revenue': [380000, 450000, 195000],
    'Q3_revenue': [360000, 430000, 210000],
    'Q4_revenue': [410000, 470000, 225000]
})

# Transform to long format
revenue_long = pd.melt(
    quarterly_data,
    id_vars=['customer_id', 'company'],
    value_vars=['Q1_revenue', 'Q2_revenue', 'Q3_revenue', 'Q4_revenue'],
    var_name='quarter',
    value_name='revenue'
)

# Clean up the quarter column
revenue_long['quarter'] = revenue_long['quarter'].str.replace('_revenue', '')
```

## Grouping & Aggregation

Analyzing data by groups:

```python
# Basic grouping
industry_stats = df.groupby('industry').agg({
    'customer_id': 'count',
    'revenue': ['sum', 'mean', 'median'],
    'employees': ['sum', 'mean']
})

# Multiple grouping levels
region_industry_stats = df.groupby(['region', 'industry']).agg({
    'customer_id': 'count',
    'revenue': 'sum'
})

# Custom aggregations
def revenue_range(x):
    return x.max() - x.min()

custom_stats = df.groupby('industry').agg({
    'revenue': [revenue_range, lambda x: x.quantile(0.75) - x.quantile(0.25)]
})

# Filtering groups
large_industries = df.groupby('industry').filter(lambda x: x['employees'].sum() > 1000)

# Transforming within groups
df['industry_avg_revenue'] = df.groupby('industry')['revenue'].transform('mean')
df['pct_of_industry_avg'] = df['revenue'] / df['industry_avg_revenue'] * 100
```

## Working with Missing Data

Strategies for handling missing values:

```python
# Detecting missing values
missing_values = df.isna().sum()
missing_percentage = (df.isna().sum() / len(df)) * 100

# Visualizing missing data patterns
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(df.isna(), cbar=False, cmap='viridis')
plt.title('Missing Value Patterns')
plt.tight_layout()

# Filling missing values with statistics
df['revenue'] = df['revenue'].fillna(df['revenue'].mean())
df['employees'] = df['employees'].fillna(df.groupby('industry')['employees'].transform('median'))

# Interpolation for time series
time_series_df['value'] = time_series_df['value'].interpolate(method='linear')

# Imputation strategies
from sklearn.impute import SimpleImputer, KNNImputer

# Mean imputation for numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns
imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# KNN imputation for more advanced cases
knn_imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
```

## Time Series Analysis

Working with date and time data:

```python
# Creating a time series DataFrame
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'value': np.random.normal(100, 20, 100)
})

# Setting the date as index
ts_data = ts_data.set_index('date')

# Resampling (aggregating to different time periods)
monthly_avg = ts_data.resample('M').mean()  # Monthly average
weekly_sum = ts_data.resample('W').sum()    # Weekly sum

# Rolling windows
rolling_avg_7d = ts_data.rolling(window=7).mean()  # 7-day moving average
rolling_std_30d = ts_data.rolling(window=30).std() # 30-day standard deviation

# Shifting for calculating period-over-period changes
ts_data['previous_day'] = ts_data['value'].shift(1)
ts_data['daily_change'] = ts_data['value'] - ts_data['previous_day']
ts_data['pct_change'] = ts_data['value'].pct_change() * 100

# Year-over-year comparison
ts_data['last_year'] = ts_data['value'].shift(365)
ts_data['yoy_change'] = (ts_data['value'] / ts_data['last_year'] - 1) * 100

# Date features extraction
ts_data['year'] = ts_data.index.year
ts_data['month'] = ts_data.index.month
ts_data['day_of_week'] = ts_data.index.dayofweek
ts_data['is_weekend'] = ts_data['day_of_week'].isin([5, 6])  # 5=Sat, 6=Sun

# Finding seasonality
monthly_pattern = ts_data.groupby(ts_data.index.month)['value'].mean()
```

## Mini-Project: Data Quality Report

Let's combine what we've learned to create a comprehensive data quality report:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_data_quality_report(df, output_file=None):
    """
    Generate a comprehensive data quality report for a DataFrame.
    
    Args:
        df: The pandas DataFrame to analyze
        output_file: Optional file path to save the report as HTML
    
    Returns:
        A new DataFrame containing data quality metrics
    """
    # Create a DataFrame to store our results
    report = pd.DataFrame(index=df.columns)
    
    # Record basic information
    report['data_type'] = df.dtypes
    report['count'] = df.count()
    report['missing_count'] = df.isna().sum()
    report['missing_percentage'] = (df.isna().sum() / len(df) * 100).round(2)
    
    # Count unique values
    report['unique_count'] = df.nunique()
    report['unique_percentage'] = (df.nunique() / df.count() * 100).round(2)
    
    # Calculate statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    report['min'] = pd.Series({col: df[col].min() for col in numeric_cols})
    report['max'] = pd.Series({col: df[col].max() for col in numeric_cols})
    report['mean'] = pd.Series({col: df[col].mean() for col in numeric_cols})
    report['median'] = pd.Series({col: df[col].median() for col in numeric_cols})
    report['std_dev'] = pd.Series({col: df[col].std() for col in numeric_cols})
    
    # Calculate outlier counts using IQR method for numeric columns
    outliers = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outliers[col] = outlier_count
    
    report['outlier_count'] = pd.Series(outliers)
    report['outlier_percentage'] = (pd.Series(outliers) / df.count() * 100).round(2)
    
    # For string columns, get additional metrics
    string_cols = df.select_dtypes(include=['object']).columns
    
    # Empty strings
    empty_strings = {}
    for col in string_cols:
        empty_strings[col] = (df[col] == '').sum()
    
    report['empty_strings'] = pd.Series(empty_strings)
    
    # Generate data quality scores (0-100)
    # Completeness: Percentage of non-missing values
    report['completeness_score'] = (100 - report['missing_percentage']).round(1)
    
    # Validity: For numeric columns, percentage of non-outliers
    # For string columns, percentage of non-empty strings
    validity = {}
    for col in numeric_cols:
        if col in outliers:
            validity[col] = 100 - report.loc[col, 'outlier_percentage']
    
    for col in string_cols:
        if col in empty_strings:
            empty_pct = (empty_strings[col] / df.count()[col] * 100)
            validity[col] = 100 - empty_pct
    
    report['validity_score'] = pd.Series(validity).round(1)
    
    # Overall quality score (average of completeness and validity)
    report['overall_quality_score'] = (
        (report['completeness_score'] + report['validity_score']) / 2
    ).round(1)
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Value Patterns')
    
    plt.subplot(2, 1, 2)
    quality_scores = report[['completeness_score', 'validity_score', 'overall_quality_score']].copy()
    quality_scores = quality_scores.sort_values('overall_quality_score')
    quality_scores.plot(kind='bar', figsize=(12, 6))
    plt.title('Data Quality Scores by Column')
    plt.ylabel('Score (0-100)')
    plt.tight_layout()
    
    # Save the plot if an output file is specified
    if output_file:
        # Save report to HTML
        html = f"""
        <html>
        <head>
            <title>Data Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333366; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                th {{ background-color: #333366; color: white; }}
                .good {{ background-color: #c8e6c9; }}
                .warning {{ background-color: #fff9c4; }}
                .danger {{ background-color: #ffcdd2; }}
            </style>
        </head>
        <body>
            <h1>Data Quality Report</h1>
            <h2>Dataset Summary</h2>
            <p>Rows: {len(df)}, Columns: {len(df.columns)}</p>
            
            <h2>Data Quality Metrics</h2>
            {report.to_html()}
            
            <h2>Data Quality Visualization</h2>
            <img src="data_quality_plot.png" width="800">
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        plt.savefig('data_quality_plot.png')
    
    return report

# Example usage
if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv('customer_data.csv')
    
    # Generate the report
    quality_report = generate_data_quality_report(df, 'data_quality_report.html')
    
    # Print the top issues
    low_quality_cols = quality_report[quality_report['overall_quality_score'] < 80]
    print("Columns with quality issues:")
    print(low_quality_cols[['missing_percentage', 'outlier_percentage', 'overall_quality_score']])
```

## Next Steps

After mastering these data analysis and manipulation skills, you'll be ready to:

1. Visualize your data with matplotlib, seaborn, and other plotting libraries
2. Create interactive dashboards to monitor data quality
3. Apply statistical methods to analyze patterns and trends
4. Build predictive models using machine learning

## Resources

- [Python for Data Analysis by Wes McKinney](https://wesmckinney.com/book/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Real Python Pandas Tutorials](https://realpython.com/pandas-python-explore-dataset/)

## Exercises and Projects

For additional practice, try these exercises:

1. Create a data cleaning pipeline for a customer database
2. Build a tool that validates data against business rules
3. Analyze trends and patterns in time series data
4. Develop a data profiling report for a complex dataset

## Contributing

If you've found this guide helpful, consider contributing:
- Add new examples relevant to data governance and quality
- Share sample datasets for practice
- Suggest improvements or corrections

Happy data wrangling!
