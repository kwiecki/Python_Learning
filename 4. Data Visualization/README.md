# Data Visualization for Data Professionals

Welcome to the Data Visualization module! This guide focuses on creating effective visualizations to communicate data insights, identify patterns, and monitor data quality - essential skills for data governance and analytics professionals.

## Why Data Visualization Matters

Visualization is critical for data professionals because it allows you to:
- Quickly identify patterns, trends, and outliers in your data
- Communicate complex findings to stakeholders effectively
- Monitor data quality and governance metrics visually
- Create dashboards for ongoing data monitoring
- Support data-driven decision making with clear visual evidence
- Explore relationships in data that might be missed in tabular formats

## Module Overview

This module covers key visualization techniques and libraries:

1. [Matplotlib Fundamentals](#matplotlib-fundamentals)
2. [Seaborn for Statistical Visualization](#seaborn-for-statistical-visualization)
3. [Interactive Visualizations with Plotly](#interactive-visualizations-with-plotly)
4. [Creating Dashboards](#creating-dashboards)
5. [Visualization Best Practices](#visualization-best-practices)
6. [Data Quality Visualizations](#data-quality-visualizations)
7. [Customizing Your Visualizations](#customizing-your-visualizations)
8. [Mini-Project: Data Quality Dashboard](#mini-project-data-quality-dashboard)

## Matplotlib Fundamentals

Matplotlib is the foundation of Python visualization:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sample data
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
data_quality_scores = [92, 88, 95, 84, 90, 93]

# Basic line plot
plt.figure(figsize=(10, 6))
plt.plot(months, data_quality_scores, marker='o', linewidth=2)
plt.title('Monthly Data Quality Scores')
plt.xlabel('Month')
plt.ylabel('Quality Score (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('quality_trend.png', dpi=300)
plt.show()

# Bar chart
plt.figure(figsize=(10, 6))
plt.bar(months, data_quality_scores, color='skyblue', edgecolor='navy')
plt.title('Monthly Data Quality Scores')
plt.xlabel('Month')
plt.ylabel('Quality Score (%)')
plt.ylim(80, 100)  # Focus on the relevant range
plt.tight_layout()
plt.show()

# Creating subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# First subplot - line
ax1.plot(months, data_quality_scores, marker='o', color='green')
ax1.set_title('Quality Trend')
ax1.set_ylabel('Score (%)')
ax1.grid(True)

# Second subplot - bar
ax2.bar(months, data_quality_scores, color='green', alpha=0.7)
ax2.set_title('Quality by Month')
ax2.set_ylabel('Score (%)')

plt.tight_layout()
plt.show()
```

## Seaborn for Statistical Visualization

Seaborn builds on Matplotlib for statistical visualization:

```python
import seaborn as sns

# Sample data
np.random.seed(42)
customer_data = pd.DataFrame({
    'industry': np.random.choice(['Technology', 'Healthcare', 'Finance', 'Retail', 'Manufacturing'], 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'revenue': np.random.normal(1000000, 500000, 100),
    'employees': np.random.lognormal(mean=4.5, sigma=1, size=100).astype(int),
    'data_quality': np.random.normal(85, 10, 100).round(1),
    'customer_satisfaction': np.random.normal(7.5, 1.5, 100).round(1)
})

# Set the aesthetic style
sns.set_theme(style="whitegrid")

# Distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(customer_data['data_quality'], kde=True, bins=20)
plt.title('Distribution of Data Quality Scores')
plt.xlabel('Quality Score (%)')
plt.ylabel('Frequency')
plt.axvline(customer_data['data_quality'].mean(), color='red', linestyle='--', 
            label=f'Mean: {customer_data["data_quality"].mean():.1f}%')
plt.legend()
plt.show()

# Box plot for categorical comparison
plt.figure(figsize=(12, 6))
sns.boxplot(x='industry', y='data_quality', data=customer_data)
plt.title('Data Quality Scores by Industry')
plt.xlabel('Industry')
plt.ylabel('Quality Score (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation heatmap
correlation_cols = ['revenue', 'employees', 'data_quality', 'customer_satisfaction']
correlation_matrix = customer_data[correlation_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Pairplot for multiple relationships
sns.pairplot(customer_data[correlation_cols])
plt.suptitle('Relationships Between Key Metrics', y=1.02)
plt.tight_layout()
plt.show()

# Categorical plot with multiple variables
plt.figure(figsize=(14, 6))
sns.catplot(
    data=customer_data, kind="bar",
    x="industry", y="data_quality", hue="region",
    palette="muted", height=6, aspect=1.5
)
plt.title('Data Quality by Industry and Region')
plt.ylabel('Average Quality Score (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Interactive Visualizations with Plotly

Creating interactive charts for deeper exploration:

```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Interactive scatter plot
fig = px.scatter(
    customer_data, 
    x='revenue', 
    y='data_quality',
    size='employees',  # Bubble size
    color='industry',  # Color by category
    hover_name='industry',
    hover_data=['region', 'customer_satisfaction'],
    title='Data Quality vs. Revenue by Industry',
    labels={'revenue': 'Annual Revenue ($)', 'data_quality': 'Data Quality Score (%)'}
)

fig.update_layout(height=600, width=900)
fig.show()

# Interactive bar chart with dropdown
fig = px.bar(
    customer_data,
    x='industry',
    y='data_quality',
    color='region',
    barmode='group',
    title='Data Quality by Industry and Region',
    labels={'data_quality': 'Data Quality Score (%)', 'industry': 'Industry'},
    height=500
)

fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()

# Interactive dashboard with multiple charts
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Data Quality by Industry', 'Quality vs. Revenue', 
                    'Quality Distribution', 'Quality by Region'),
    specs=[[{'type': 'bar'}, {'type': 'scatter'}],
           [{'type': 'histogram'}, {'type': 'box'}]]
)

# Add bar chart
industry_avg = customer_data.groupby('industry')['data_quality'].mean().reset_index()
fig.add_trace(
    go.Bar(x=industry_avg['industry'], y=industry_avg['data_quality'],
           name='Average Quality Score'),
    row=1, col=1
)

# Add scatter plot
fig.add_trace(
    go.Scatter(x=customer_data['revenue'], y=customer_data['data_quality'],
               mode='markers', name='Quality vs Revenue',
               marker=dict(color=customer_data['data_quality'], 
                          colorscale='Viridis', showscale=True)),
    row=1, col=2
)

# Add histogram
fig.add_trace(
    go.Histogram(x=customer_data['data_quality'], nbinsx=20, name='Quality Distribution'),
    row=2, col=1
)

# Add box plot
fig.add_trace(
    go.Box(x=customer_data['region'], y=customer_data['data_quality'], name='Quality by Region'),
    row=2, col=2
)

fig.update_layout(height=800, width=1000, title_text='Data Quality Dashboard')
fig.show()
```

## Creating Dashboards

Building simple dashboards for monitoring:

```python
# Using Plotly for a dashboard

# Example data - monthly data quality metrics
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
completeness = [95, 93, 96, 94, 97, 98]
accuracy = [88, 90, 92, 89, 91, 94]
consistency = [85, 86, 90, 92, 93, 95]
timeliness = [90, 88, 91, 93, 95, 96]

# Create dashboard
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Data Completeness', 'Data Accuracy', 
                    'Data Consistency', 'Data Timeliness'),
    shared_xaxes=True
)

# Completeness chart
fig.add_trace(
    go.Scatter(x=months, y=completeness, mode='lines+markers', 
               name='Completeness', line=dict(color='blue')),
    row=1, col=1
)

# Accuracy chart
fig.add_trace(
    go.Scatter(x=months, y=accuracy, mode='lines+markers', 
               name='Accuracy', line=dict(color='green')),
    row=1, col=2
)

# Consistency chart
fig.add_trace(
    go.Scatter(x=months, y=consistency, mode='lines+markers', 
               name='Consistency', line=dict(color='orange')),
    row=2, col=1
)

# Timeliness chart
fig.add_trace(
    go.Scatter(x=months, y=timeliness, mode='lines+markers', 
               name='Timeliness', line=dict(color='red')),
    row=2, col=2
)

# Update layout
fig.update_layout(
    title_text='Data Quality Metrics Dashboard',
    height=700,
    width=1000,
    showlegend=False
)

# Add range slider for time selection
fig.update_layout(
    xaxis=dict(
        rangeslider=dict(visible=True),
        type='category'
    )
)

# Set y-axis range for all subplots
for i in range(1, 3):
    for j in range(1, 3):
        fig.update_yaxes(range=[80, 100], row=i, col=j)

fig.show()
```

## Visualization Best Practices

Guidelines for creating effective visualizations:

```python
# Examples of good vs. bad practices

# Sample data - data quality by department and month
departments = ['Sales', 'Marketing', 'Finance', 'Operations', 'IT', 'HR']
quality_scores = {
    'Jan': [92, 88, 95, 90, 91, 87],
    'Feb': [93, 90, 94, 91, 92, 88],
    'Mar': [91, 92, 96, 89, 93, 90],
    'Apr': [94, 91, 97, 92, 94, 91]
}

# GOOD PRACTICE: Use appropriate chart types
plt.figure(figsize=(12, 10))

# 1. Use appropriate color schemes (colorblind-friendly)
plt.subplot(2, 2, 1)
for i, month in enumerate(quality_scores.keys()):
    plt.plot(departments, quality_scores[month], marker='o', label=month)
plt.title('GOOD: Appropriate Line Colors')
plt.ylabel('Quality Score (%)')
plt.ylim(85, 100)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 2. Clear labeling and titles
plt.subplot(2, 2, 2)
x = np.arange(len(departments))
width = 0.2
for i, month in enumerate(quality_scores.keys()):
    plt.bar(x + i*width, quality_scores[month], width, label=month)
plt.title('GOOD: Clear Labels and Title')
plt.xlabel('Department')
plt.ylabel('Quality Score (%)')
plt.xticks(x + width*1.5, departments, rotation=45)
plt.ylim(85, 100)
plt.legend(title='Month')

# 3. Focus on the data (no chartjunk)
plt.subplot(2, 2, 3)
df_quality = pd.DataFrame(quality_scores, index=departments)
sns.heatmap(df_quality, annot=True, cmap='YlGnBu', vmin=85, vmax=100)
plt.title('GOOD: Focus on Data, No Distractions')
plt.ylabel('Department')
plt.xlabel('Month')

# 4. Choose appropriate scales
plt.subplot(2, 2, 4)
plt.plot(departments, quality_scores['Apr'], marker='o', color='green')
plt.title('GOOD: Appropriate Y-Axis Scale')
plt.ylabel('Quality Score (%)')
plt.ylim(85, 100)  # Focused scale to show meaningful differences
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# BAD PRACTICE: Examples to avoid
plt.figure(figsize=(12, 10))

# 1. Poor color choices and 3D effects
plt.subplot(2, 2, 1)
ax = plt.axes(projection='3d')
xpos = np.arange(len(departments))
for i, month in enumerate(list(quality_scores.keys())[:3]):
    ax.bar3d(xpos, i, 0, 0.8, 0.8, quality_scores[month], shade=True)
ax.set_title('BAD: Unnecessary 3D Effects')
ax.set_ylabel('Month')
ax.set_zlabel('Score')
ax.set_xticks(xpos)
ax.set_xticklabels(departments, rotation=45)

# 2. Too many pie charts
plt.subplot(2, 2, 2)
explode = (0.1, 0, 0, 0, 0, 0)
plt.pie(quality_scores['Jan'], explode=explode, labels=departments, autopct='%1.1f%%', shadow=True)
plt.title('BAD: Pie Chart for Comparison')

# 3. Poor axis scaling
plt.subplot(2, 2, 3)
plt.plot(departments, quality_scores['Apr'], marker='o', color='red')
plt.title('BAD: Poor Y-Axis Scale')
plt.ylabel('Quality Score (%)')
plt.ylim(0, 100)  # Scale starts at 0, making differences harder to see

# 4. Cluttered visualization
plt.subplot(2, 2, 4)
for i, month in enumerate(quality_scores.keys()):
    plt.plot(departments, quality_scores[month], marker='s', linestyle='-.' if i % 2 else '--')
    plt.bar(departments, [s-85 for s in quality_scores[month]], alpha=0.3)
plt.title('BAD: Cluttered Visualization')
plt.xticks(rotation=90)
plt.grid(True)
plt.legend(quality_scores.keys())

plt.tight_layout()
plt.show()
```

## Data Quality Visualizations

Specialized visualizations for data quality monitoring:

```python
# Create a dataset with data quality metrics
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=90, freq='D')
data_quality = pd.DataFrame({
    'date': dates,
    'completeness': np.random.normal(92, 3, 90).clip(70, 100),
    'accuracy': np.random.normal(88, 4, 90).clip(70, 100),
    'consistency': np.random.normal(90, 5, 90).clip(70, 100),
    'timeliness': np.random.normal(85, 6, 90).clip(70, 100)
})

# Add some trends and events
data_quality.loc[30:, 'completeness'] += 5  # Improvement after day 30
data_quality.loc[45:55, 'accuracy'] -= 15   # Issue between days 45-55
data_quality.loc[60:, 'consistency'] += 7   # Improvement after day 60
data_quality.loc[20:40, 'timeliness'] += np.linspace(0, 10, 21)  # Gradual improvement

# 1. Time series chart with events
plt.figure(figsize=(12, 6))
plt.plot(data_quality['date'], data_quality['completeness'], label='Completeness')
plt.plot(data_quality['date'], data_quality['accuracy'], label='Accuracy')
plt.plot(data_quality['date'], data_quality['consistency'], label='Consistency')
plt.plot(data_quality['date'], data_quality['timeliness'], label='Timeliness')

# Highlight events
plt.axvspan(dates[30], dates[31], color='green', alpha=0.2, label='Completeness Fix')
plt.axvspan(dates[45], dates[55], color='red', alpha=0.2, label='Accuracy Issue')
plt.axvspan(dates[60], dates[61], color='blue', alpha=0.2, label='Consistency Improvement')

plt.title('Data Quality Metrics Over Time with Key Events')
plt.xlabel('Date')
plt.ylabel('Quality Score (%)')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(70, 100)
plt.tight_layout()
plt.show()

# 2. Data quality heatmap
# Melt the dataframe for heatmap
melted_data = pd.melt(
    data_quality, 
    id_vars=['date'], 
    value_vars=['completeness', 'accuracy', 'consistency', 'timeliness'],
    var_name='metric', 
    value_name='score'
)

# Create a pivot table for the heatmap
pivot_data = melted_data.pivot_table(
    index=pd.Grouper(key='date', freq='W'),  # Group by week
    columns='metric',
    values='score',
    aggfunc='mean'
)

plt.figure(figsize=(12, 8))
sns.heatmap(
    pivot_data, 
    annot=False, 
    fmt=".1f", 
    cmap="RdYlGn", 
    vmin=70, 
    vmax=100,
    cbar_kws={'label': 'Quality Score (%)'}
)
plt.title('Weekly Data Quality Scores Heatmap')
plt.ylabel('Week')
plt.tight_layout()
plt.show()

# 3. Data quality radar chart
# Get the latest scores
latest_scores = data_quality.iloc[-1][['completeness', 'accuracy', 'consistency', 'timeliness']].values
target_scores = np.array([95, 95, 95, 95])
previous_scores = data_quality.iloc[-30][['completeness', 'accuracy', 'consistency', 'timeliness']].values

categories = ['Completeness', 'Accuracy', 'Consistency', 'Timeliness']
N = len(categories)

# Create angles for each metric (in radians)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # Close the loop

# Extend all score arrays to close the loop
latest_scores = np.append(latest_scores, latest_scores[0])
target_scores = np.append(target_scores, target_scores[0])
previous_scores = np.append(previous_scores, previous_scores[0])
categories += [categories[0]]  # Close the loop for labels

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(angles, latest_scores, 'o-', linewidth=2, label='Current')
ax.plot(angles, previous_scores, 'o-', linewidth=2, label='30 Days Ago')
ax.plot(angles, target_scores, 'o-', linewidth=2, label='Target')
ax.fill(angles, latest_scores, alpha=0.1)
ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
ax.set_ylim(70, 100)
ax.grid(True)
ax.set_title('Data Quality Radar Chart')
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()

# 4. Missing values visualization
# Create a dataset with missing values
np.random.seed(42)
columns = ['customer_id', 'name', 'email', 'phone', 'address', 'city', 'state', 'zip', 'country']
rows = 50

# Create a matrix of missing value patterns
missing_matrix = np.random.choice([True, False], size=(rows, len(columns)), p=[0.1, 0.9])
# Make some patterns in the missing data
missing_matrix[:, 3] = np.random.choice([True, False], size=rows, p=[0.3, 0.7])  # More missing phones
missing_matrix[:, 4:6] = np.random.choice([True, False], size=(rows, 2), p=[0.2, 0.8])  # Address/city related
missing_matrix[:, 0] = False  # No missing IDs

# Convert to a DataFrame
df_missing = pd.DataFrame(np.where(missing_matrix, np.nan, 'data'), columns=columns)

# Visualize missing values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(df_missing.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Value Patterns')
plt.xlabel('Fields')
plt.ylabel('Records')

plt.subplot(1, 2, 2)
missing_counts = df_missing.isnull().sum().sort_values(ascending=False)
plt.barh(missing_counts.index, missing_counts.values)
plt.title('Missing Values by Field')
plt.xlabel('Count of Missing Values')
plt.tight_layout()
plt.show()
```

## Customizing Your Visualizations

Creating visually appealing and branded visualizations:

```python
# Setting a consistent style for your visualizations
def set_corporate_style():
    """Apply a consistent corporate style to matplotlib visualizations"""
    # Set colors
    corporate_colors = {
        'primary': '#003366',       # Dark blue
        'secondary': '#669999',     # Teal
        'accent1': '#FF9933',       # Orange
        'accent2': '#CC3366',       # Magenta
        'gray_dark': '#4D4D4D',     # Dark gray
        'gray_light': '#E6E6E6',    # Light gray
        'success': '#339966',       # Green
        'warning': '#FFCC33',       # Yellow
        'danger': '#CC3333'         # Red
    }
    
    # Create a custom colormap
    corporate_palette = [corporate_colors['primary'], corporate_colors['secondary'], 
                        corporate_colors['accent1'], corporate_colors['accent2']]
    
    # Set matplotlib parameters
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = corporate_colors['gray_light']
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.format'] = 'png'
    
    # Create a custom matplotlib style dictionary
    corporate_style = {
        'axes.prop_cycle': plt.cycler('color', corporate_palette),
        'figure.figsize': (10, 6),
        'figure.titlesize': 16,
        'figure.titleweight': 'bold',
    }
    
    # Apply the style
    plt.rcParams.update(corporate_style)
    
    # Return the color palette for use in other plots
    return corporate_colors

# Apply the corporate style
corporate_colors = set_corporate_style()

# Example of using the custom style with Seaborn
sns.set_theme(style="whitegrid", rc=plt.rcParams)

# Create a sample visualization with custom styling
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
metrics = {
    'Completeness': [92, 94, 95, 93, 96, 98],
    'Accuracy': [88, 89, 91, 94, 92, 95],
    'Consistency': [85, 87, 90, 92, 94, 95],
    'Timeliness': [90, 92, 91, 93, 95, 97]
}

data_df = pd.DataFrame(metrics, index=months)

# Example of a custom-styled line chart
plt.figure(figsize=(12, 6))
for i, (metric, values) in enumerate(metrics.items()):
    plt.plot(months, values, marker='o', linewidth=2.5, label=metric)

# Customize the chart
plt.title('Data Quality Metrics Trend', pad=20)
plt.ylabel('Score (%)')
plt.ylim(80, 100)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Metrics', frameon=True, facecolor='white', framealpha=0.9, 
           edgecolor=corporate_colors['gray_light'])

# Add annotations
best_metric = max(metrics.items(), key=lambda x: x[1][-1])
best_value = best_metric[1][-1]
plt.annotate(f'Best: {best_value}%', 
             xy=(5, best_value), 
             xytext=(5, best_value+2),
             arrowprops=dict(facecolor=corporate_colors['primary'], shrink=0.05),
             fontsize=10,
             color=corporate_colors['primary'],
             fontweight='bold')

plt.tight_layout()
plt.savefig('corporate_styled_chart.png')
plt.show()
```

## Mini-Project: Data Quality Dashboard

Let's combine what we've learned to create a comprehensive data quality dashboard:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Generate sample data for our dashboard
def generate_data_quality_metrics(days=90, departments=None):
    """Generate sample data quality metrics for demonstration"""
    if departments is None:
        departments = ['Sales', 'Marketing', 'Finance', 'HR', 'Operations']
    
    np.random.seed(42)  # For reproducibility
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    
    # Generate metrics for each department
    for dept in departments:
        # Base values for this department with some randomness
        base_completeness = np.random.uniform(85, 95)
        base_accuracy = np.random.uniform(80, 90)
        base_consistency = np.random.uniform(75, 85)
        base_timeliness = np.random.uniform(70, 80)
        
        # Trends and patterns
        completeness_trend = np.random.choice([-0.05, 0, 0.05, 0.1])
        accuracy_trend = np.random.choice([-0.05, 0, 0.05, 0.1])
        consistency_trend = np.random.choice([-0.05, 0, 0.05, 0.1])
        timeliness_trend = np.random.choice([-0.05, 0, 0.05, 0.1])
        
        # Add data points for each day
        for i, date in enumerate(dates):
            # Add trends and some random variation
            completeness = min(100, max(0, base_completeness + i * completeness_trend + np.random.normal(0, 1)))
            accuracy = min(100, max(0, base_accuracy + i * accuracy_trend + np.random.normal(0, 1.2)))
            consistency = min(100, max(0, base_consistency + i * consistency_trend + np.random.normal(0, 1.5)))
            timeliness = min(100, max(0, base_timeliness + i * timeliness_trend + np.random.normal(0, 1.8)))
            
            # Add some "events"
            # Data quality improvement initiative at day 30
            if i == 30:
                completeness += 3
                accuracy += 3
                consistency += 3
                timeliness += 3
            
            # System issue at day 45-55
            if 45 <= i <= 55:
                accuracy -= 5
                timeliness -= 8
            
            # Add row to the dataset
            data.append({
                'date': date,
                'department': dept,
                'completeness': round(completeness, 1),
                'accuracy': round(accuracy, 1),
                'consistency': round(consistency, 1),
                'timeliness': round(timeliness, 1)
            })
    
    return pd.DataFrame(data)

# Generate the sample data
data_quality_df = generate_data_quality_metrics()

# Calculate the overall quality score
data_quality_df['overall_score'] = data_quality_df[['completeness', 'accuracy', 'consistency', 'timeliness']].mean(axis=1).round(1)

# Create an interactive dashboard
def create_data_quality_dashboard(df):
    """Create an interactive data quality dashboard using Plotly"""
    
    # Create the dashboard layout
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"type": "indicator"}, {"type": "indicator"}],
[{"type": "pie"}, {"type": "indicator"}]
        ],
        subplot_titles=(
            "Overall Data Quality Trend by Department", 
            "", "",
            "Data Quality Dimensions", "Data Quality by Department"
        ),
        column_widths=[0.6, 0.4],
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )
    
    # 1. Overall trend line chart - top section
    departments = df['department'].unique()
    
    for dept in departments:
        dept_data = df[df['department'] == dept]
        fig.add_trace(
            go.Scatter(
                x=dept_data['date'], 
                y=dept_data['overall_score'],
                mode='lines',
                name=dept,
                hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 2. Latest overall score gauge - middle left
    latest_overall = df['overall_score'].iloc[-1]
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=latest_overall,
            title={"text": "Overall Quality Score"},
            delta={'reference': df['overall_score'].iloc[-30], 'valueformat': '.1f'},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 70], 'color': "red"},
                    {'range': [70, 85], 'color': "orange"},
                    {'range': [85, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=2, col=1
    )
    
    # 3. Data quality dimensions comparison - middle right
    latest_data = df.iloc[-1]
    dimensions = ['completeness', 'accuracy', 'consistency', 'timeliness']
    
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=latest_data['completeness'],
            title={"text": "Completeness"},
            domain={'row': 0, 'column': 0},
            delta={'reference': df['completeness'].iloc[-30], 'valueformat': '.1f'}
        ),
        row=2, col=2
    )
    
    # 4. Department quality pie chart - bottom left
    dept_avg = df.groupby('department')['overall_score'].mean().reset_index()
    
    fig.add_trace(
        go.Pie(
            labels=dept_avg['department'],
            values=dept_avg['overall_score'],
            hole=.3,
            hoverinfo='label+percent',
            textinfo='label+value',
            textfont_size=12,
            marker=dict(
                colors=['#3366CC', '#DC3912', '#FF9900', '#109618', '#990099'],
                line=dict(color='#FFFFFF', width=2)
            )
        ),
        row=3, col=1
    )
    
    # 5. Latest quality dimensions by department - bottom right
    latest_by_dept = df[df['date'] == df['date'].max()].reset_index(drop=True)
    
    # Create a heatmap of the latest data by department
    z_data = []
    for dim in dimensions:
        z_data.append(latest_by_dept.pivot_table(index='department', values=dim, aggfunc='mean').values.flatten())
    
    fig.add_trace(
        go.Heatmap(
            z=z_data,
            x=latest_by_dept['department'].unique(),
            y=dimensions,
            colorscale='RdYlGn',
            zmin=70,
            zmax=100,
            colorbar=dict(title="Score"),
            hovertemplate='Department: %{x}<br>Dimension: %{y}<br>Score: %{z:.1f}%<extra></extra>'
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Data Quality Management Dashboard",
        title_font_size=24,
        height=900,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add date range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeslider_thickness=0.05,
        row=1, col=1
    )
    
    # Add annotations
    # Highlight system issue period
    start_date = df['date'].min() + timedelta(days=45)
    end_date = df['date'].min() + timedelta(days=55)
    
    fig.add_vrect(
        x0=start_date, 
        x1=end_date,
        fillcolor="red", 
        opacity=0.1,
        layer="below", 
        line_width=0,
        row=1, col=1
    )
    
    fig.add_annotation(
        x=start_date + (end_date - start_date)/2,
        y=100,
        text="System Issue",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-30,
        row=1, col=1
    )
    
    return fig

# Create and display the dashboard
quality_dashboard = create_data_quality_dashboard(data_quality_df)
quality_dashboard.show()
```

## Creating Your Data Quality Dashboard

To build a data quality dashboard for your organization:

1. **Identify key metrics**: Determine the data quality dimensions most important for your business (completeness, accuracy, consistency, timeliness, etc.)

2. **Set up data collection**: Create automated processes to calculate quality scores for your datasets, potentially using pandas functions from the previous module

3. **Choose visualization components**:
   - Time series for tracking trends
   - Gauges for current status
   - Heatmaps for comparing dimensions across data domains
   - Drill-down capabilities for investigating issues

4. **Add alerting**: Configure thresholds for quality scores and highlight periods where metrics fall below acceptable levels

5. **Create a dashboard code template**:
   ```python
   def create_quality_dashboard(data_file, output_file=None):
       """
       Create a data quality dashboard from quality metrics
       
       Args:
           data_file: CSV file with quality metrics
           output_file: Optional HTML file to save the dashboard
       """
       # Load the data
       df = pd.read_csv(data_file)
       
       # Create the dashboard (using code from the example above)
       dashboard = create_data_quality_dashboard(df)
       
       # Save or display
       if output_file:
           dashboard.write_html(output_file)
       else:
           dashboard.show()
   ```

## Next Steps

After mastering these visualization techniques, you'll be ready to:

1. Create interactive web-based dashboards with Dash or Streamlit
2. Connect visualizations to live data sources
3. Automate the generation and distribution of data quality reports
4. Develop custom visualization components for specific data governance needs

## Resources

- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Data Visualization Best Practices](https://www.tableau.com/learn/articles/data-visualization-tips)
- [Information Dashboard Design (Book by Stephen Few)](https://www.perceptualedge.com/library.php)

## Exercises and Projects

For additional practice, try these exercises:

1. Create a data quality dashboard for a sample dataset
2. Develop visualizations that show the impact of data quality on business metrics
3. Build a missing value visualization tool for your datasets
4. Create custom color schemes and styles for your organization's visualizations

## Contributing

If you've found this guide helpful, consider contributing:
- Add new visualization examples relevant to data governance
- Share templates for common data quality reports
- Suggest improvements or corrections

Happy visualizing!
