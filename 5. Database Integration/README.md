# Database Integration for Data Professionals

Welcome to the Database Integration module! This guide focuses on using Python to connect to, query, and manipulate databases - critical skills for data governance and analytics professionals who need to work with enterprise data sources.

## Why Database Integration Matters

For data professionals, database skills are essential because they allow you to:
- Access and analyze data directly from its source
- Automate data quality checks and validations
- Create data pipelines that enforce governance rules
- Implement robust data management workflows
- Document and monitor data lineage
- Perform sophisticated queries across multiple tables
- Ensure data security and proper access controls

## Module Overview

This module covers key database integration techniques:

1. [SQL Fundamentals in Python](#sql-fundamentals-in-python)
2. [Database Connections](#database-connections)
3. [Querying Databases](#querying-databases)
4. [Working with SQLAlchemy](#working-with-sqlalchemy)
5. [Managing Database Schema](#managing-database-schema)
6. [Data Migration Techniques](#data-migration-techniques)
7. [Connection Pooling and Management](#connection-pooling-and-management)
8. [Mini-Project: Data Quality Monitor](#mini-project-data-quality-monitor)

## SQL Fundamentals in Python

SQL is the universal language of databases:

```python
# SQL syntax for common operations
"""
# Selecting data
SELECT column1, column2
FROM table_name
WHERE condition;

# Joining tables
SELECT t1.column1, t2.column2
FROM table1 t1
JOIN table2 t2 ON t1.id = t2.table1_id;

# Aggregating data
SELECT category, COUNT(*) as count, AVG(value) as avg_value
FROM table_name
GROUP BY category
HAVING count > 10;

# Inserting data
INSERT INTO table_name (column1, column2)
VALUES (value1, value2);

# Updating data
UPDATE table_name
SET column1 = value1
WHERE condition;

# Deleting data
DELETE FROM table_name
WHERE condition;
"""

# SQL best practices for data quality
"""
# Enforcing data consistency
ALTER TABLE customer_data
ADD CONSTRAINT valid_email 
CHECK (email LIKE '%@%.%');

# Creating indexes for performance
CREATE INDEX idx_customer_id
ON customer_data(customer_id);

# Setting up foreign key constraints
ALTER TABLE orders
ADD CONSTRAINT fk_customer
FOREIGN KEY (customer_id) REFERENCES customers(id);

# Creating a view for data quality metrics
CREATE VIEW data_quality_metrics AS
SELECT 
    COUNT(*) as total_records,
    SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) as missing_emails,
    SUM(CASE WHEN phone IS NULL THEN 1 ELSE 0 END) as missing_phones,
    SUM(CASE WHEN email LIKE '%@%.%' THEN 1 ELSE 0 END) as valid_emails
FROM customer_data;
"""
```

## Database Connections

Connecting to various database types:

```python
# SQLite - lightweight file-based database
import sqlite3

# Create connection
conn = sqlite3.connect('customer_data.db')
cursor = conn.cursor()

# Create a table
cursor.execute('''
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    phone TEXT,
    created_date TEXT DEFAULT CURRENT_TIMESTAMP
)
''')

# Insert sample data
cursor.execute('''
INSERT INTO customers (name, email, phone)
VALUES (?, ?, ?)
''', ('Acme Corporation', 'contact@acme.com', '555-123-4567'))

# Commit changes and close
conn.commit()
conn.close()

# PostgreSQL - enterprise-grade relational database
import psycopg2

try:
    # Connection parameters
    conn_params = {
        'dbname': 'customer_db',
        'user': 'username',
        'password': 'password',
        'host': 'localhost',
        'port': '5432'
    }
    
    # Create connection
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()
    
    # Execute a query
    cursor.execute("SELECT version();")
    db_version = cursor.fetchone()
    print(f"PostgreSQL version: {db_version[0]}")
    
    # Close connection
    cursor.close()
    conn.close()
    
except (Exception, psycopg2.DatabaseError) as error:
    print(f"Error connecting to PostgreSQL: {error}")

# MySQL / MariaDB
import mysql.connector

try:
    # Create connection
    conn = mysql.connector.connect(
        host="localhost",
        user="username",
        password="password",
        database="customer_db"
    )
    
    cursor = conn.cursor()
    
    # Execute a query
    cursor.execute("SELECT COUNT(*) FROM customers")
    count = cursor.fetchone()[0]
    print(f"Customer count: {count}")
    
    # Close connection
    cursor.close()
    conn.close()
    
except mysql.connector.Error as error:
    print(f"Error connecting to MySQL: {error}")

# Microsoft SQL Server
import pyodbc

try:
    # Create connection
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=customer_db;"
        "UID=username;"
        "PWD=password"
    )
    
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    # Execute a query
    cursor.execute("SELECT TOP 5 * FROM customers ORDER BY created_date DESC")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    
    # Close connection
    cursor.close()
    conn.close()
    
except pyodbc.Error as error:
    print(f"Error connecting to SQL Server: {error}")
```

## Querying Databases

Effective ways to query and process data:

```python
import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('customer_data.db')

# Basic query with parameters
def get_customer_by_id(customer_id):
    """Retrieve a customer by ID"""
    query = "SELECT * FROM customers WHERE id = ?"
    cursor = conn.cursor()
    cursor.execute(query, (customer_id,))
    return cursor.fetchone()

# Query that returns multiple rows
def get_customers_by_industry(industry):
    """Retrieve all customers in a specific industry"""
    query = "SELECT * FROM customers WHERE industry = ? ORDER BY name"
    cursor = conn.cursor()
    cursor.execute(query, (industry,))
    return cursor.fetchall()

# Loading query results directly into pandas
def get_customer_metrics():
    """Get customer metrics as a pandas DataFrame"""
    query = """
    SELECT 
        industry,
        COUNT(*) as customer_count,
        AVG(annual_revenue) as avg_revenue,
        SUM(CASE WHEN active = 1 THEN 1 ELSE 0 END) as active_customers,
        SUM(CASE WHEN active = 0 THEN 1 ELSE 0 END) as inactive_customers
    FROM customers
    GROUP BY industry
    ORDER BY customer_count DESC
    """
    return pd.read_sql_query(query, conn)

# Query with JOIN across multiple tables
def get_customer_orders(customer_id):
    """Get all orders for a specific customer"""
    query = """
    SELECT 
        o.id as order_id,
        o.order_date,
        o.total_amount,
        p.name as product_name,
        oi.quantity,
        oi.unit_price
    FROM orders o
    JOIN order_items oi ON o.id = oi.order_id
    JOIN products p ON oi.product_id = p.id
    WHERE o.customer_id = ?
    ORDER BY o.order_date DESC
    """
    return pd.read_sql_query(query, conn, params=(customer_id,))

# Handling large result sets efficiently
def process_large_dataset():
    """Process a large dataset in chunks to avoid memory issues"""
    query = "SELECT * FROM large_table"
    
    # Using pandas chunks
    for chunk in pd.read_sql_query(query, conn, chunksize=10000):
        # Process each chunk
        process_data_chunk(chunk)
        
    # Using cursor iteration
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM large_table")
    
    # Fetch and process rows in batches
    batch_size = 10000
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        process_row_batch(rows)

# Dynamic query building (safely)
def search_customers(name=None, industry=None, min_revenue=None, active=None):
    """Search customers with dynamic filters"""
    query = "SELECT * FROM customers WHERE 1=1"
    params = []
    
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    
    if industry:
        query += " AND industry = ?"
        params.append(industry)
    
    if min_revenue:
        query += " AND annual_revenue >= ?"
        params.append(min_revenue)
    
    if active is not None:
        query += " AND active = ?"
        params.append(1 if active else 0)
    
    return pd.read_sql_query(query, conn, params=tuple(params))

# Don't forget to close your connections
def cleanup():
    """Close the database connection properly"""
    conn.close()
```

## Working with SQLAlchemy

SQLAlchemy provides a more Pythonic way to work with databases:

```python
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Table, MetaData, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# Create engine and base
engine = create_engine('sqlite:///customer_data.db', echo=True)
Base = declarative_base()

# Define models
class Customer(Base):
    __tablename__ = 'customers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    industry = Column(String(50))
    email = Column(String(100), unique=True)
    phone = Column(String(20))
    annual_revenue = Column(Float)
    active = Column(Boolean, default=True)
    created_date = Column(DateTime, default=datetime.now)
    
    # Relationship to orders
    orders = relationship("Order", back_populates="customer")
    
    def __repr__(self):
        return f"<Customer(name='{self.name}', industry='{self.industry}')>"

class Order(Base):
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'))
    order_date = Column(DateTime, default=datetime.now)
    total_amount = Column(Float, nullable=False)
    
    # Relationships
    customer = relationship("Customer", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")
    
    def __repr__(self):
        return f"<Order(id={self.id}, total=${self.total_amount:.2f})>"

class Product(Base):
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    category = Column(String(50))
    unit_price = Column(Float, nullable=False)
    
    # Relationship
    order_items = relationship("OrderItem", back_populates="product")
    
    def __repr__(self):
        return f"<Product(name='{self.name}', price=${self.unit_price:.2f})>"

class OrderItem(Base):
    __tablename__ = 'order_items'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey('orders.id'))
    product_id = Column(Integer, ForeignKey('products.id'))
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)
    
    # Relationships
    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")
    
    @property
    def subtotal(self):
        return self.quantity * self.unit_price
    
    def __repr__(self):
        return f"<OrderItem(product_id={self.product_id}, quantity={self.quantity})>"

# Create all tables
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Adding data
def add_sample_customer():
    """Add a sample customer with SQLAlchemy ORM"""
    new_customer = Customer(
        name="TechSolutions Inc.",
        industry="Technology",
        email="contact@techsolutions.com",
        phone="555-987-6543",
        annual_revenue=2500000.00,
        active=True
    )
    
    session.add(new_customer)
    session.commit()
    return new_customer.id

# Querying data with SQLAlchemy ORM
def get_active_customers():
    """Get all active customers using SQLAlchemy ORM"""
    return session.query(Customer).filter(Customer.active == True).all()

def get_high_value_customers(revenue_threshold=1000000):
    """Get customers with revenue above threshold"""
    return session.query(Customer).\
        filter(Customer.annual_revenue >= revenue_threshold).\
        order_by(Customer.annual_revenue.desc()).\
        all()

def get_customer_with_orders(customer_id):
    """Get a customer and their orders in one query"""
    return session.query(Customer).\
        filter(Customer.id == customer_id).\
        options(joinedload(Customer.orders)).\
        first()

# Updating data
def update_customer_status(customer_id, is_active):
    """Update a customer's active status"""
    customer = session.query(Customer).get(customer_id)
    if customer:
        customer.active = is_active
        session.commit()
        return True
    return False

# Raw SQL with SQLAlchemy
def get_industry_metrics():
    """Run a raw SQL query with SQLAlchemy"""
    sql = text("""
    SELECT 
        industry,
        COUNT(*) as customer_count,
        AVG(annual_revenue) as avg_revenue,
        SUM(CASE WHEN active = 1 THEN 1 ELSE 0 END) as active_customers
    FROM customers
    GROUP BY industry
    ORDER BY customer_count DESC
    """)
    
    result = engine.execute(sql)
    return [dict(row) for row in result]

# Always close session when done
def cleanup():
    """Close the SQLAlchemy session"""
    session.close()
```

## Managing Database Schema

Techniques for managing and evolving database structure:

```python
import sqlite3
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, text
import pandas as pd
from datetime import datetime

# Connect to database
conn = sqlite3.connect('customer_data.db')
cursor = conn.cursor()

# Schema version tracking
def initialize_schema_version():
    """Create and initialize a schema version table"""
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS schema_version (
        id INTEGER PRIMARY KEY,
        version INTEGER NOT NULL,
        applied_date TEXT NOT NULL,
        description TEXT
    )
    ''')
    
    # Check if we need to initialize the version
    cursor.execute("SELECT COUNT(*) FROM schema_version")
    if cursor.fetchone()[0] == 0:
        cursor.execute('''
        INSERT INTO schema_version (version, applied_date, description)
        VALUES (1, ?, 'Initial schema creation')
        ''', (datetime.now().isoformat(),))
        conn.commit()

def get_current_schema_version():
    """Get the current schema version number"""
    cursor.execute("SELECT MAX(version) FROM schema_version")
    return cursor.fetchone()[0]

# Adding a new column
def add_column_to_table(table_name, column_name, column_definition):
    """Add a new column to an existing table"""
    try:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")
        conn.commit()
        print(f"Added column {column_name} to {table_name}")
        return True
    except sqlite3.Error as e:
        print(f"Error adding column: {e}")
        return False

# Example schema migration
def migrate_to_version_2():
    """Migrate schema to version 2 - add last_updated column to customers"""
    current_version = get_current_schema_version()
    
    if current_version < 2:
        print("Migrating to schema version 2...")
        
        # Add the new column
        add_column_to_table("customers", "last_updated", "TEXT")
        
        # Update the schema version
        cursor.execute('''
        INSERT INTO schema_version (version, applied_date, description)
        VALUES (2, ?, 'Added last_updated column to customers')
        ''', (datetime.now().isoformat(),))
        
        conn.commit()
        print("Migration to version 2 complete")
    else:
        print("Schema is already at version 2 or higher")

# Safe schema changes using SQLAlchemy
def safe_schema_update_with_sqlalchemy():
    """Demonstrate safer schema changes with SQLAlchemy"""
    # Create an engine
    engine = create_engine('sqlite:///customer_data.db')
    metadata = MetaData()
    
    # Reflect existing tables
    metadata.reflect(bind=engine)
    
    # Get the customers table
    customers = metadata.tables['customers']
    
    # Check if column exists before trying to add it
    if 'loyalty_points' not in customers.columns:
        # Add a new column
        engine.execute('ALTER TABLE customers ADD COLUMN loyalty_points INTEGER DEFAULT 0')
        print("Added loyalty_points column")
    else:
        print("loyalty_points column already exists")
        
# Table transformations
def transform_table_structure():
    """Transform a table structure (create new, copy data, replace)"""
    # Create a new table with the desired structure
    cursor.execute('''
    CREATE TABLE customers_new (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        phone TEXT,
        industry TEXT,
        annual_revenue REAL,
        active INTEGER DEFAULT 1,
        created_date TEXT,
        last_updated TEXT,
        loyalty_tier TEXT
    )
    ''')
    
    # Copy data from old table to new, setting default for new columns
    cursor.execute('''
    INSERT INTO customers_new (id, name, email, phone, industry, annual_revenue, active, created_date, last_updated)
    SELECT id, name, email, phone, industry, annual_revenue, active, created_date, datetime('now')
    FROM customers
    ''')
    
    # Update loyalty tier based on revenue (example transformation)
    cursor.execute('''
    UPDATE customers_new
    SET loyalty_tier = CASE
        WHEN annual_revenue >= 5000000 THEN 'Platinum'
        WHEN annual_revenue >= 1000000 THEN 'Gold'
        WHEN annual_revenue >= 100000 THEN 'Silver'
        ELSE 'Bronze'
    END
    ''')
    
    # Drop old table and rename new one
    cursor.execute("DROP TABLE customers")
    cursor.execute("ALTER TABLE customers_new RENAME TO customers")
    
    conn.commit()
    print("Table structure transformation complete")

# Adding indexes for performance
def add_indexes_for_performance():
    """Add indexes to improve query performance"""
    try:
        # Index for email lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email)")
        
        # Index for industry filtering and sorting
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_industry ON customers(industry)")
        
        # Composite index for common query pattern
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_customers_industry_active_revenue 
        ON customers(industry, active, annual_revenue DESC)
        """)
        
        conn.commit()
        print("Indexes created successfully")
    except sqlite3.Error as e:
        print(f"Error creating indexes: {e}")

# Schema reporting
def generate_schema_report():
    """Generate a report of the current database schema"""
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    schema_report = {}
    
    for table in tables:
        table_name = table[0]
        
        # Skip SQLite internal tables
        if table_name.startswith('sqlite_'):
            continue
            
        # Get column information
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # Get index information
        cursor.execute(f"PRAGMA index_list({table_name})")
        indexes = cursor.fetchall()
        
        # Store in our report dictionary
        schema_report[table_name] = {
            'columns': [{'name': col[1], 'type': col[2], 'nullable': not col[3], 'pk': col[5]} for col in columns],
            'indexes': [{'name': idx[1], 'unique': idx[2]} for idx in indexes]
        }
    
    return schema_report

# Print schema report
def print_schema_report(schema_report):
    """Print a formatted schema report"""
    for table_name, table_info in schema_report.items():
        print(f"\nTABLE: {table_name}")
        print("Columns:")
        for col in table_info['columns']:
            pk_marker = "PK" if col['pk'] else "  "
            null_marker = "NULL" if col['nullable'] else "NOT NULL"
            print(f"  [{pk_marker}] {col['name']} ({col['type']}) {null_marker}")
        
        print("Indexes:")
        for idx in table_info['indexes']:
            unique_marker = "UNIQUE" if idx['unique'] else "     "
            print(f"  [{unique_marker}] {idx['name']}")
        print("-" * 50)
```

## Data Migration Techniques

Moving and transforming data between databases:

```python
import sqlite3
import pandas as pd
import csv
from sqlalchemy import create_engine
import json
import os
from datetime import datetime

# Source and destination databases
source_conn = sqlite3.connect('source_data.db')
dest_conn = sqlite3.connect('customer_data.db')

# Basic data migration function
def migrate_table_data(source_table, dest_table, column_mapping=None):
    """
    Migrate data from source table to destination table
    
    Args:
        source_table: Name of the source table
        dest_table: Name of the destination table
        column_mapping: Dictionary mapping source columns to destination columns
                        If None, assumes identical column names
    """
    # Default to identical column names if no mapping provided
    if column_mapping is None:
        # Get source table columns
        source_cursor = source_conn.cursor()
        source_cursor.execute(f"PRAGMA table_info({source_table})")
        source_columns = [col[1] for col in source_cursor.fetchall()]
        
        # Create an identity mapping
        column_mapping = {col: col for col in source_columns}
    
    # Prepare column lists for the query
    source_cols = ", ".join(column_mapping.keys())
    dest_cols = ", ".join(column_mapping.values())
    
    # Create source and destination cursors
    source_cursor = source_conn.cursor()
    dest_cursor = dest_conn.cursor()
    
    # Begin transactions
    source_conn.execute("BEGIN TRANSACTION")
    dest_conn.execute("BEGIN TRANSACTION")
    
    try:
        # Read data from source table in batches
        batch_size = 1000
        source_cursor.execute(f"SELECT {source_cols} FROM {source_table}")
        
        # Process in batches to avoid memory issues
        batch = source_cursor.fetchmany(batch_size)
        total_rows = 0
        
        while batch:
            # Prepare placeholders for the INSERT statement
            placeholders = ", ".join(["?"] * len(column_mapping))
            
            # Insert batch into destination
            dest_cursor.executemany(
                f"INSERT INTO {dest_table} ({dest_cols}) VALUES ({placeholders})",
                batch
            )
            
            total_rows += len(batch)
            print(f"Migrated {total_rows} rows...")
            
            # Get next batch
            batch = source_cursor.fetchmany(batch_size)
        
        # Commit the transactions
        dest_conn.commit()
        source_conn.commit()
        
        print(f"Migration complete: {total_rows} rows migrated from {source_table} to {dest_table}")
        return total_rows
    
    except Exception as e:
        # Rollback on error
        source_conn.rollback()
        dest_conn.rollback()
        print(f"Error during migration: {e}")
        return 0

# Migration with data transformation
def migrate_with_transformation(source_table, dest_table, transform_function):
    """
    Migrate data with a transformation function
    
    Args:
        source_table: Name of the source table
        dest_table: Name of the destination table
        transform_function: Function that takes a source row dict and returns a dest row dict
    """
    source_cursor = source_conn.cursor()
    dest_cursor = dest_conn.cursor()
    
    # Get source table columns
    source_cursor.execute(f"PRAGMA table_info({source_table})")
    source_columns = [col[1] for col in source_cursor.fetchall()]
    
    # Begin transactions
    source_conn.execute("BEGIN TRANSACTION")
    dest_conn.execute("BEGIN TRANSACTION")
    
    try:
        # Read data from source table in batches
        batch_size = 1000
        source_cursor.execute(f"SELECT {', '.join(source_columns)} FROM {source_table}")
        
        # Process in batches
        source_batch = source_cursor.fetchmany(batch_size)
        total_rows = 0
        
        while source_batch:
            # Transform each row in the batch
            transformed_batch = []
            for row in source_batch:
                # Convert row tuple to dict with column names
                row_dict = {source_columns[i]: value for i, value in enumerate(row)}
                
                # Apply transformation function
                transformed_row = transform_function(row_dict)
                
                if transformed_row:  # Skip if transformation returns None
                    # Extract values in the right order for insertion
                    transformed_values = tuple(transformed_row.values())
                    transformed_batch.append(transformed_values)
            
            if transformed_batch:
                # Prepare placeholders for the INSERT statement
                dest_columns = list(transformed_batch[0])
                placeholders = ", ".join(["?"] * len(dest_columns))
                
                # Insert transformed batch
                dest_cursor.executemany(
                    f"INSERT INTO {dest_table} ({', '.join(dest_columns)}) VALUES ({placeholders})",
                    transformed_batch
                )
            
            total_rows += len(transformed_batch)
            print(f"Transformed and migrated {total_rows} rows...")
            
            # Get next batch
            source_batch = source_cursor.fetchmany(batch_size)
        
        # Commit the transactions
        dest_conn.commit()
        source_conn.commit()
        
        print(f"Migration with transformation complete: {total_rows} rows processed")
        return total_rows
    
    except Exception as e:
        # Rollback on error
        source_conn.rollback()
        dest_conn.rollback()
        print(f"Error during migration: {e}")
        return 0

# Example transformation function
def transform_customer_data(source_row):
    """Transform customer data from source to destination format"""
    # Skip inactive customers
    if source_row.get('status') == 'inactive':
        return None
    
    # Create new transformed row
    return {
        'name': source_row.get('company_name', '').strip(),
        'industry': source_row.get('sector', 'Unknown').title(),
        'email': source_row.get('primary_email', '').lower(),
        'phone': format_phone_number(source_row.get('phone', '')),
        'annual_revenue': parse_revenue(source_row.get('annual_sales', 0)),
        'active': 1,  # All included customers are active
        'created_date': format_date(source_row.get('created_timestamp')),
        'last_updated': datetime.now().isoformat()
    }

# Helper functions for the transformation
def format_phone_number(phone):
    """Format phone number to consistent format"""
    # Remove non-digit characters
    digits = ''.join(c for c in phone if c.isdigit())
    
    # Format as (XXX) XXX-XXXX if it's a 10-digit number
    if len(digits) == 10:
        return f"({digits[0:3]}) {digits[3:6]}-{digits[6:]}"
    else:
        return phone  # Return original if not 10 digits

def parse_revenue(revenue_str):
    """Parse revenue string to a numeric value"""
    if not revenue_str:
        return 0
    
    # Handle different formats
    try:
        # Remove currency symbols, commas, etc.
        clean_str = revenue_str.replace('$', '').replace(',', '')
        
        # Handle 'K' and 'M' suffixes
        if 'K' in clean_str or 'k' in clean_str:
            return float(clean_str.replace('K', '').replace('k', '')) * 1000
        elif 'M' in clean_str or 'm' in clean_str:
            return float(clean_str.replace('M', '').replace('m', '')) * 1000000
        else:
            return float(clean_str)
    except:
        return 0

def format_date(date_str):
    """Format date string to ISO format"""
    if not date_str:
        return datetime.now().isoformat()
    
    try:
        # Try different date formats
        for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%d-%b-%Y', '%Y/%m/%d'):
            try:
                return datetime.strptime(date_str, fmt).isoformat()
            except:
                continue
        
        # If no format worked, return current date
        return datetime.now().isoformat()
    except:
        return datetime.now().isoformat()

# CSV Import/Export
def export_table_to_csv(table_name, csv_file):
    """Export a table to a CSV file"""
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", source_conn)
    df.to_csv(csv_file, index=False)
    print(f"Exported {len(df)} rows from {table_name} to {csv_file}")

def import_csv_to_table(csv_file, table_name, if_exists='append'):
    """Import a CSV file to a database table"""
    df = pd.read_csv(csv_file)
    
    # Optional data cleaning/transformation here
    
    # Create SQLAlchemy engine and import
    engine = create_engine('sqlite:///customer_
# Create SQLAlchemy engine and import
    engine = create_engine('sqlite:///customer_data.db')
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"Imported {len(df)} rows into {table_name}")

# JSON Import/Export
def export_table_to_json(table_name, json_file):
    """Export a table to a JSON file"""
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", source_conn)
    df.to_json(json_file, orient='records', date_format='iso')
    print(f"Exported {len(df)} rows from {table_name} to {json_file}")

def import_json_to_table(json_file, table_name, if_exists='append'):
    """Import a JSON file to a database table"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Create SQLAlchemy engine and import
    engine = create_engine('sqlite:///customer_data.db')
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"Imported {len(df)} rows into {table_name}")

# Cross-database migration
def migrate_between_db_types():
    """Migrate data between different database types"""
    # Source: SQLite
    sqlite_engine = create_engine('sqlite:///source_data.db')
    
    # Destination: PostgreSQL (example - not actually run)
    # pg_engine = create_engine('postgresql://username:password@localhost:5432/dest_db')
    
    # For demonstration, we'll use another SQLite DB
    dest_engine = create_engine('sqlite:///customer_data.db')
    
    # Read from source
    customers_df = pd.read_sql_table('legacy_customers', sqlite_engine)
    
    # Optional transformation
    # Example: standardize column names to snake_case
    customers_df.columns = [col.lower().replace(' ', '_') for col in customers_df.columns]
    
    # Write to destination
    customers_df.to_sql('customers', dest_engine, if_exists='append', index=False)
    print(f"Migrated {len(customers_df)} rows to destination database")

# Clean up connections
def cleanup():
    """Close all database connections"""
    source_conn.close()
    dest_conn.close()
    print("Database connections closed")
```

## Connection Pooling and Management

Efficiently managing database connections:

```python
import sqlite3
import time
import queue
import threading
import contextlib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool

# Basic connection pool implementation
class ConnectionPool:
    """A simple database connection pool"""
    
    def __init__(self, db_file, max_connections=5, timeout=5):
        self.db_file = db_file
        self.max_connections = max_connections
        self.timeout = timeout
        self.connections = queue.Queue(maxsize=max_connections)
        self.active_connections = 0
        
        # Initialize the pool with connections
        for _ in range(max_connections):
            conn = sqlite3.connect(db_file)
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            # Use row factory for named access
            conn.row_factory = sqlite3.Row
            self.connections.put(conn)
    
    def get_connection(self):
        """Get a connection from the pool"""
        try:
            # Try to get a connection from the queue
            conn = self.connections.get(timeout=self.timeout)
            self.active_connections += 1
            return conn
        except queue.Empty:
            raise TimeoutError("Timed out waiting for a database connection")
    
    def release_connection(self, conn):
        """Return a connection to the pool"""
        # Check if connection is still valid
        try:
            conn.execute("SELECT 1")  # Simple test query
            self.connections.put(conn)
        except sqlite3.Error:
            # Connection is broken, create a new one
            conn = sqlite3.connect(self.db_file)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.row_factory = sqlite3.Row
            self.connections.put(conn)
        
        self.active_connections -= 1
    
    def close_all(self):
        """Close all connections in the pool"""
        while not self.connections.empty():
            try:
                conn = self.connections.get(block=False)
                conn.close()
            except queue.Empty:
                break
        
        print(f"Closed all connections in the pool")

# Context manager for automatic connection handling
@contextlib.contextmanager
def pooled_connection(pool):
    """Context manager for database connections"""
    conn = None
    try:
        conn = pool.get_connection()
        yield conn
    finally:
        if conn:
            pool.release_connection(conn)

# Usage example
def connection_pool_example():
    """Demonstrate connection pool usage"""
    # Create the pool
    pool = ConnectionPool('customer_data.db', max_connections=3)
    
    # Function that uses a connection
    def query_customers(customer_id):
        with pooled_connection(pool) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM customers WHERE id = ?", (customer_id,))
            result = cursor.fetchone()
            # Simulate some processing time
            time.sleep(0.5)
            if result:
                return dict(result)
            return None
    
    # Simulate multiple concurrent requests
    results = []
    threads = []
    
    for i in range(5):  # 5 concurrent requests, but only 3 connections
        customer_id = i + 1
        thread = threading.Thread(target=lambda: results.append(query_customers(customer_id)))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Print results
    for result in results:
        if result:
            print(f"Found customer: {result['name']}")
        else:
            print("Customer not found")
    
    # Close all connections
    pool.close_all()

# SQLAlchemy connection pooling
def sqlalchemy_pool_example():
    """Demonstrate SQLAlchemy connection pooling"""
    # Create an engine with connection pooling
    engine = create_engine(
        'sqlite:///customer_data.db',
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600  # Recycle connections after 1 hour
    )
    
    # Create a scoped session factory
    # This ensures thread-local session management
    session_factory = sessionmaker(bind=engine)
    Session = scoped_session(session_factory)
    
    def query_with_session():
        # Get a session from the scoped_session
        session = Session()
        try:
            # Execute a query
            result = session.execute("SELECT COUNT(*) FROM customers").scalar()
            print(f"Customer count: {result}")
            
            # Simulate some work
            time.sleep(0.5)
            
            return result
        except Exception as e:
            print(f"Error in query: {e}")
            session.rollback()
            raise
        finally:
            # Always close the session
            session.close()
    
    # Run multiple queries concurrently
    threads = []
    for _ in range(10):
        thread = threading.Thread(target=query_with_session)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Remove the session registry
    Session.remove()
    
    # Check pool status
    print(f"Pool size: {engine.pool.size()}")
    print(f"Checkedout connections: {engine.pool.checkedout()}")
    
    # Dispose of the engine to close all connections
    engine.dispose()
    print("Engine disposed, all connections closed")

# Connection management best practices
def connection_best_practices():
    """Demonstrate connection management best practices"""
    # 1. Always use context managers for connections
    with sqlite3.connect('customer_data.db') as conn:
        # Do work with the connection
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM customers")
        count = cursor.fetchone()[0]
        print(f"Customer count: {count}")
        # Connection automatically closed when exiting the context
    
    # 2. Use SQLAlchemy's sessionmaker and scoped_session
    engine = create_engine('sqlite:///customer_data.db')
    Session = scoped_session(sessionmaker(bind=engine))
    
    # Get a session
    session = Session()
    try:
        # Do work with the session
        result = session.execute("SELECT COUNT(*) FROM orders").scalar()
        print(f"Order count: {result}")
        # Commit the transaction
        session.commit()
    except:
        # Rollback on error
        session.rollback()
        raise
    finally:
        # Always close the session
        session.close()
    
    # Clean up the session registry
    Session.remove()
    
    # 3. Connection timeouts and retries
    def execute_with_retry(query, params=None, max_retries=3, retry_delay=1):
        """Execute a query with retry logic"""
        retries = 0
        while retries < max_retries:
            try:
                with sqlite3.connect('customer_data.db', timeout=10) as conn:
                    cursor = conn.cursor()
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                    
                    result = cursor.fetchall()
                    conn.commit()
                    return result
            except sqlite3.OperationalError as e:
                # This could be a database lock or timeout
                retries += 1
                if retries >= max_retries:
                    raise
                print(f"Database error, retrying ({retries}/{max_retries}): {e}")
                time.sleep(retry_delay)
    
    # Example usage of retry function
    try:
        result = execute_with_retry("SELECT * FROM customers WHERE industry = ?", ("Technology",))
        print(f"Found {len(result)} technology customers")
    except Exception as e:
        print(f"Query failed after retries: {e}")
```

## Mini-Project: Data Quality Monitor

Let's combine what we've learned to create a database data quality monitoring system:

```python
import sqlite3
import pandas as pd
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='data_quality_monitor.log'
)
logger = logging.getLogger('DataQualityMonitor')

class DataQualityMonitor:
    """A system to monitor and report on database data quality"""
    
    def __init__(self, db_file, config_file=None):
        """Initialize the data quality monitor"""
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file)
        self.conn.row_factory = sqlite3.Row
        
        # Load configuration
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                'tables': ['customers', 'orders', 'products'],
                'quality_checks': {
                    'missing_values': True,
                    'duplicate_records': True,
                    'data_format': True,
                    'referential_integrity': True,
                    'value_distribution': True
                },
                'thresholds': {
                    'missing_values_pct': 5.0,
                    'duplicate_records_pct': 1.0
                },
                'email_notifications': {
                    'enabled': False,
                    'smtp_server': 'smtp.example.com',
                    'smtp_port': 587,
                    'username': 'user@example.com',
                    'password': 'password',
                    'recipients': ['admin@example.com']
                },
                'history_tracking': {
                    'enabled': True,
                    'retention_days': 90
                }
            }
        
        # Ensure quality metrics table exists
        self._initialize_metrics_table()
        
        logger.info(f"Initialized DataQualityMonitor for {db_file}")
    
    def _initialize_metrics_table(self):
        """Create the table to store quality metrics if it doesn't exist"""
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_quality_metrics (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            table_name TEXT NOT NULL,
            check_type TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            details TEXT,
            status TEXT
        )
        ''')
        
        # Create index on timestamp and table_name for faster queries
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_metrics_timestamp_table
        ON data_quality_metrics(timestamp, table_name)
        ''')
        
        self.conn.commit()
    
    def run_all_checks(self):
        """Run all configured data quality checks"""
        timestamp = datetime.now().isoformat()
        results = {}
        
        for table in self.config['tables']:
            logger.info(f"Running quality checks for table: {table}")
            table_results = {
                'missing_values': None,
                'duplicate_records': None,
                'data_format': None,
                'referential_integrity': None,
                'value_distribution': None
            }
            
            # Run enabled checks
            if self.config['quality_checks']['missing_values']:
                table_results['missing_values'] = self.check_missing_values(table, timestamp)
            
            if self.config['quality_checks']['duplicate_records']:
                table_results['duplicate_records'] = self.check_duplicate_records(table, timestamp)
            
            if self.config['quality_checks']['data_format']:
                table_results['data_format'] = self.check_data_formats(table, timestamp)
            
            if self.config['quality_checks']['referential_integrity']:
                table_results['referential_integrity'] = self.check_referential_integrity(table, timestamp)
            
            if self.config['quality_checks']['value_distribution']:
                table_results['value_distribution'] = self.check_value_distribution(table, timestamp)
            
            results[table] = table_results
        
        # Generate and store summary
        summary = self.generate_summary(results, timestamp)
        
        # Send notifications if enabled and issues found
        if self.config['email_notifications']['enabled'] and summary['has_issues']:
            self.send_email_alert(summary)
        
        logger.info(f"Completed all quality checks")
        return summary
    
    def check_missing_values(self, table, timestamp):
        """Check for missing values in the table"""
        cursor = self.conn.cursor()
        
        # Get column information
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Get total row count
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        total_rows = cursor.fetchone()[0]
        
        if total_rows == 0:
            logger.warning(f"Table {table} is empty")
            return {
                'status': 'WARNING',
                'message': 'Table is empty',
                'details': {'total_rows': 0}
            }
        
        # Check for NULL values in each column
        results = {}
        issues_found = False
        
        for column in columns:
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL")
            null_count = cursor.fetchone()[0]
            null_percentage = (null_count / total_rows) * 100
            
            # Determine status based on threshold
            if null_percentage >= self.config['thresholds']['missing_values_pct']:
                status = 'FAIL'
                issues_found = True
            else:
                status = 'PASS'
            
            # Store result for this column
            results[column] = {
                'null_count': null_count,
                'null_percentage': null_percentage,
                'status': status
            }
            
            # Insert into metrics table
            cursor.execute('''
            INSERT INTO data_quality_metrics 
            (timestamp, table_name, check_type, metric_name, metric_value, details, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, 
                table, 
                'missing_values', 
                f"{column}_null_pct", 
                null_percentage,
                json.dumps({'null_count': null_count, 'total_rows': total_rows}),
                status
            ))
        
        self.conn.commit()
        
        overall_status = 'FAIL' if issues_found else 'PASS'
        logger.info(f"Missing values check for {table}: {overall_status}")
        
        return {
            'status': overall_status,
            'message': 'Missing values check completed',
            'details': results
        }
    
    def check_duplicate_records(self, table, timestamp):
        """Check for duplicate records in the table"""
        cursor = self.conn.cursor()
        
        # Get primary key information
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        pk_columns = [col[1] for col in columns if col[5] > 0]  # col[5] is the PK flag
        
        if not pk_columns:
            # If no PK defined, use all columns (less reliable)
            pk_columns = [col[1] for col in columns]
        
        # Get total row count
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        total_rows = cursor.fetchone()[0]
        
        if total_rows == 0:
            return {
                'status': 'WARNING',
                'message': 'Table is empty',
                'details': {'total_rows': 0}
            }
        
        # Group by all fields we're checking and count occurrences
        columns_str = ', '.join(pk_columns)
        cursor.execute(f"""
        SELECT {columns_str}, COUNT(*) as count
        FROM {table}
        GROUP BY {columns_str}
        HAVING COUNT(*) > 1
        """)
        
        duplicate_groups = cursor.fetchall()
        total_duplicates = sum(row['count'] - 1 for row in duplicate_groups)
        duplicate_percentage = (total_duplicates / total_rows) * 100
        
        # Determine status based on threshold
        if duplicate_percentage >= self.config['thresholds']['duplicate_records_pct']:
            status = 'FAIL'
        else:
            status = 'PASS'
        
        # Store details about the duplicates
        duplicate_details = []
        for group in duplicate_groups:
            detail = {col: group[col] for col in pk_columns}
            detail['count'] = group['count']
            duplicate_details.append(detail)
        
        # Insert into metrics table
        cursor.execute('''
        INSERT INTO data_quality_metrics 
        (timestamp, table_name, check_type, metric_name, metric_value, details, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, 
            table, 
            'duplicate_records', 
            'duplicate_pct', 
            duplicate_percentage,
            json.dumps({
                'total_duplicates': total_duplicates, 
                'duplicate_groups': len(duplicate_groups),
                'total_rows': total_rows
            }),
            status
        ))
        
        self.conn.commit()
        
        logger.info(f"Duplicate records check for {table}: {status}")
        
        return {
            'status': status,
            'message': 'Duplicate records check completed',
            'details': {
                'total_duplicates': total_duplicates,
                'duplicate_groups': len(duplicate_groups),
                'duplicate_percentage': duplicate_percentage,
                'duplicate_details': duplicate_details[:10] if duplicate_details else []  # Limit for brevity
            }
        }
    
    def check_data_formats(self, table, timestamp):
        """Check for data format issues (e.g., dates, emails, phone numbers)"""
        cursor = self.conn.cursor()
        
        # Get column information
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        
        # Format checks based on column names and types
        format_checks = []
        for col in columns:
            col_name = col[1]
            col_type = col[2].lower()
            
            # Email format check
            if 'email' in col_name.lower() and 'text' in col_type:
                format_checks.append({
                    'column': col_name,
                    'check_name': 'email_format',
                    'query': f"SELECT COUNT(*) FROM {table} WHERE {col_name} IS NOT NULL AND {col_name} NOT LIKE '%@%.%'"
                })
            
            # Phone format check (simple check for digits, parens, dashes, spaces)
            if ('phone' in col_name.lower() or 'mobile' in col_name.lower()) and 'text' in col_type:
                format_checks.append({
                    'column': col_name,
                    'check_name': 'phone_format',
                    'query': f"SELECT COUNT(*) FROM {table} WHERE {col_name} IS NOT NULL AND {col_name} NOT REGEXP '^[0-9()\\-\\s]+$'"
                })
            
            # Date format check
            if ('date' in col_name.lower() or 'time' in col_name.lower()) and 'text' in col_type:
                format_checks.append({
                    'column': col_name,
                    'check_name': 'date_format',
                    # Basic ISO format check
                    'query': f"SELECT COUNT(*) FROM {table} WHERE {col_name} IS NOT NULL AND datetime({col_name}) IS NULL"
                })
            
            # Numeric value in text field
            if any(x in col_name.lower() for x in ['amount', 'price', 'cost', 'revenue']) and 'text' in col_type:
                format_checks.append({
                    'column': col_name,
                    'check_name': 'numeric_text',
                    'query': f"SELECT COUNT(*) FROM {table} WHERE {col_name} IS NOT NULL AND {col_name} NOT REGEXP '^[0-9]*\\.?[0-9]+$'"
                })
        
        # Get total row count
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        total_rows = cursor.fetchone()[0]
        
        if total_rows == 0:
            return {
                'status': 'WARNING',
                'message': 'Table is empty',
                'details': {'total_rows': 0}
            }
        
        # Run the format checks
        results = {}
        issues_found = False
        
        for check in format_checks:
            try:
                cursor.execute(check['query'])
                invalid_count = cursor.fetchone()[0]
                invalid_percentage = (invalid_count / total_rows) * 100
                
                # Determine status
                if invalid_count > 0:
                    status = 'FAIL'
                    issues_found = True
                else:
                    status = 'PASS'
                
                # Store result
                results[f"{check['column']}_{check['check_name']}"] = {
                    'invalid_count': invalid_count,
                    'invalid_percentage': invalid_percentage,
                    'status': status
                }
                
                # Insert into metrics table
                cursor.execute('''
                INSERT INTO data_quality_metrics 
                (timestamp, table_name, check_type, metric_name, metric_value, details, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp, 
                    table, 
                    'data_format', 
                    f"{check['column']}_{check['check_name']}", 
                    invalid_percentage,
                    json.dumps({'invalid_count': invalid_count, 'total_rows': total_rows}),
                    status
                ))
            except sqlite3.OperationalError as e:
                # SQLite doesn't support REGEXP by default, this would need custom function
                logger.warning(f"Could not run format check: {e}")
                continue
        
        self.conn.commit()
        
        overall_status = 'FAIL' if issues_found else 'PASS'
        logger.info(f"Data format check for {table}: {overall_status}")
        
        return {
            'status': overall_status,
            'message': 'Data format check completed',
            'details': results
        }
    
    def check_referential_integrity(self, table, timestamp):
        """Check for referential integrity issues"""
        cursor = self.conn.cursor()
        
        # Get foreign key information
        cursor.execute(f"PRAGMA foreign_key_list({table})")
        foreign_keys = cursor.fetchall()
        
        if not foreign_keys:
            return {
                'status': 'INFO',
                'message': 'No foreign keys defined for this table',
                'details': {}
            }
        
        # Check each foreign key
        results = {}
        issues_found = False
        
        for fk in foreign_keys:
            fk_column = fk['from']  # Local column
            ref_table = fk['table']  # Referenced table
            ref_column = fk['to']    # Referenced column
            
            # Count orphaned records (those without a matching reference)
            query = f"""
            SELECT COUNT(*) FROM {table} t
            LEFT JOIN {ref_table} r ON t.{fk_column} = r.{ref_column}
            WHERE t.{fk_column} IS NOT NULL AND r.{ref_column} IS NULL
            """
            
            try:
                cursor.execute(query)
                orphaned_count = cursor.fetchone()[0]
                
                # Get total non-null values
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {fk_column} IS NOT NULL")
                total_refs = cursor.fetchone()[0]
                
                if total_refs > 0:
                    orphaned_percentage = (orphaned_count / total_refs) * 100
                else:
                    orphaned_percentage = 0
                
                # Determine status
                if orphaned_count > 0:
                    status = 'FAIL'
                    issues_found = True
                else:
                    status = 'PASS'
                
                # Store result
                fk_name = f"{table}.{fk_column} -> {ref_table}.{ref_column}"
                results[fk_name] = {
                    'orphaned_count': orphaned_count,
                    'orphaned_percentage': orphaned_percentage,
                    'total_refs': total_refs,
                    'status': status
                }
                
                # Insert into metrics table
                cursor.execute('''
                INSERT INTO data_quality_metrics 
                (timestamp, table_name, check_type, metric_name, metric_value, details, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp, 
                    table, 
                    'referential_integrity', 
                    f"fk_{fk_column}_to_{ref_table}", 
                    orphaned_percentage,
                    json.dumps({'orphaned_count': orphaned_count, 'total_refs': total_refs}),
                    status
                ))
            except sqlite3.OperationalError as e:
                logger.error(f"Error checking referential integrity: {e}")
                continue
        
        self.conn.commit()
        
        overall_status = 'FAIL' if issues_found else 'PASS'
        logger.info(f"Referential integrity check for {table}: {overall_status}")
        
        return {
            'status': overall_status,
            'message': 'Referential integrity check completed',
            'details': results
        }
    
    def check_value_distribution(self, table, timestamp):
        """Check value distributions for anomalies"""
        cursor = self.conn.cursor()
        
        # Get column information
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        
        # Get total row count
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        total_rows = cursor.fetchone()[0]
        
        if total_rows == 0:
            return {
                'status': 'WARNING',
                'message': 'Table is empty',
                'details': {'total_rows': 0}
            }
        
        # Check each column
        results = {}
        
        for col in columns:
            col_name = col[1]
            col_type = col[2].lower()
            
            try:
                # For numeric columns, get basic statistics
                if any(t in col_type for t in ['int', 'real', 'decimal', 'numeric', 'float']):
                    cursor.execute(f"""
                    SELECT 
                        COUNT(*) as count,
                        AVG({col_name}) as avg,
                        MIN({col_name}) as min,
                        MAX({col_name}) as max,
                        (MAX({col_name}) - MIN({col_name})) as range
                    FROM {table}
                    WHERE {col_name} IS NOT NULL
                    """)
                    stats = dict(cursor.fetchone())
                    
                    # Check for outliers (simple Z-score)
                    if stats['count'] > 10:  # Only if we have enough data
                        cursor.execute(f"""
                        SELECT 
                            AVG({col_name}) as avg,
                            SQRT(AVG(({col_name} - (SELECT AVG({col_name}) FROM {table})) * 
                                   ({col_name} - (SELECT AVG({col_name}) FROM {table})))) as std
                        FROM {table}
                        WHERE {col_name} IS NOT NULL
                        """)
                        dist = dict(cursor.fetchone())
                        
                        # Count outliers (values more than 3 standard deviations from mean)
                        if dist['std'] > 0:  # Avoid division by zero
                            cursor.execute(f"""
                            SELECT COUNT(*) FROM {table}
                            WHERE ABS({col_name} - ?) / ? > 3
                            """, (dist['avg'], dist['std']))
                            outlier_count = cursor.fetchone()[0]
                            outlier_percentage = (outlier_count / stats['count']) * 100
outlier_percentage = (outlier_count / stats['count']) * 100
                            
                            stats['outlier_count'] = outlier_count
                            stats['outlier_percentage'] = outlier_percentage
                            
                            # Determine status based on outliers
                            if outlier_percentage > 5:  # More than 5% outliers
                                status = 'WARNING'
                            else:
                                status = 'PASS'
                        else:
                            status = 'PASS'
                            stats['outlier_count'] = 0
                            stats['outlier_percentage'] = 0
                    else:
                        status = 'INFO'
                        stats['outlier_count'] = 0
                        stats['outlier_percentage'] = 0
                    
                    # Insert into metrics table
                    cursor.execute('''
                    INSERT INTO data_quality_metrics 
                    (timestamp, table_name, check_type, metric_name, metric_value, details, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp, 
                        table, 
                        'value_distribution', 
                        f"{col_name}_distribution", 
                        stats.get('outlier_percentage', 0),
                        json.dumps(stats),
                        status
                    ))
                    
                    results[col_name] = {
                        'stats': stats,
                        'status': status
                    }
                
                # For categorical columns, check for domination by a single value
                elif 'text' in col_type and not ('date' in col_name.lower() or 'time' in col_name.lower()):
                    cursor.execute(f"""
                    SELECT {col_name}, COUNT(*) as count
                    FROM {table}
                    WHERE {col_name} IS NOT NULL
                    GROUP BY {col_name}
                    ORDER BY count DESC
                    LIMIT 1
                    """)
                    
                    top_value = cursor.fetchone()
                    if top_value:
                        top_value = dict(top_value)
                        top_percentage = (top_value['count'] / total_rows) * 100
                        
                        # Determine status based on dominance
                        if top_percentage > 95:  # One value is >95% of all values
                            status = 'WARNING'
                        else:
                            status = 'PASS'
                        
                        # Insert into metrics table
                        cursor.execute('''
                        INSERT INTO data_quality_metrics 
                        (timestamp, table_name, check_type, metric_name, metric_value, details, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            timestamp, 
                            table, 
                            'value_distribution', 
                            f"{col_name}_dominance", 
                            top_percentage,
                            json.dumps({
                                'top_value': str(top_value[col_name]),
                                'top_count': top_value['count'],
                                'top_percentage': top_percentage
                            }),
                            status
                        ))
                        
                        results[col_name] = {
                            'top_value': str(top_value[col_name]),
                            'top_count': top_value['count'],
                            'top_percentage': top_percentage,
                            'status': status
                        }
            except sqlite3.OperationalError as e:
                logger.warning(f"Could not check value distribution for {col_name}: {e}")
                continue
        
        self.conn.commit()
        
        # Determine overall status
        if any(r.get('status') == 'WARNING' for r in results.values()):
            overall_status = 'WARNING'
        else:
            overall_status = 'PASS'
        
        logger.info(f"Value distribution check for {table}: {overall_status}")
        
        return {
            'status': overall_status,
            'message': 'Value distribution check completed',
            'details': results
        }
    
    def generate_summary(self, results, timestamp):
        """Generate a summary of all quality checks"""
        summary = {
            'timestamp': timestamp,
            'tables_checked': len(results),
            'has_issues': False,
            'tables': {}
        }
        
        for table, checks in results.items():
            table_summary = {
                'status': 'PASS',
                'checks': {}
            }
            
            for check_name, check_result in checks.items():
                if check_result:
                    table_summary['checks'][check_name] = {
                        'status': check_result['status'],
                        'message': check_result['message']
                    }
                    
                    # Update table status based on check status
                    if check_result['status'] == 'FAIL':
                        table_summary['status'] = 'FAIL'
                        summary['has_issues'] = True
                    elif check_result['status'] == 'WARNING' and table_summary['status'] != 'FAIL':
                        table_summary['status'] = 'WARNING'
                        summary['has_issues'] = True
            
            summary['tables'][table] = table_summary
        
        return summary
    
    def get_historical_metrics(self, table=None, days=30, metric_name=None):
        """Get historical data quality metrics"""
        cursor = self.conn.cursor()
        
        query = """
        SELECT timestamp, table_name, check_type, metric_name, metric_value, status
        FROM data_quality_metrics
        WHERE timestamp >= datetime('now', '-{} days')
        """.format(days)
        
        params = []
        
        if table:
            query += " AND table_name = ?"
            params.append(table)
        
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
        
        query += " ORDER BY timestamp"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert to a list of dicts
        metrics = []
        for row in rows:
            metrics.append(dict(row))
        
        return metrics
    
    def generate_trend_chart(self, metrics, title, output_file=None):
        """Generate a trend chart from historical metrics"""
        if not metrics:
            return None
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(metrics)
        
        # Convert timestamp strings to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by date and metric name
        pivot_df = df.pivot_table(
            index='timestamp', 
            columns='metric_name', 
            values='metric_value',
            aggfunc='mean'
        )
        
        # Plot
        plt.figure(figsize=(12, 6))
        pivot_df.plot(ax=plt.gca())
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Metric Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Metrics')
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            return output_file
        else:
            # Create in-memory image
            from io import BytesIO
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            return buffer
    
    def send_email_alert(self, summary):
        """Send an email alert for data quality issues"""
        if not self.config['email_notifications']['enabled']:
            logger.info("Email notifications are disabled")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config['email_notifications']['username']
            msg['To'] = ', '.join(self.config['email_notifications']['recipients'])
            msg['Subject'] = f"Data Quality Alert - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Create HTML content
            html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    .fail {{ background-color: #ffcccc; }}
                    .warning {{ background-color: #fff4cc; }}
                    .pass {{ background-color: #ccffcc; }}
                </style>
            </head>
            <body>
                <h2>Data Quality Alert</h2>
                <p>Timestamp: {summary['timestamp']}</p>
                <p>Tables Checked: {summary['tables_checked']}</p>
                
                <h3>Summary by Table</h3>
                <table>
                    <tr>
                        <th>Table</th>
                        <th>Status</th>
                        <th>Checks</th>
                    </tr>
            """
            
            for table, table_summary in summary['tables'].items():
                status_class = table_summary['status'].lower()
                html += f"""
                    <tr class="{status_class}">
                        <td>{table}</td>
                        <td>{table_summary['status']}</td>
                        <td>
                """
                
                for check_name, check_result in table_summary['checks'].items():
                    check_status_class = check_result['status'].lower()
                    html += f'<div class="{check_status_class}">{check_name}: {check_result["status"]}</div>'
                
                html += """
                        </td>
                    </tr>
                """
            
            html += """
                </table>
                
                <p>This is an automated message. Please check the data quality monitor for details.</p>
            </body>
            </html>
            """
            
            # Attach HTML content
            msg.attach(MIMEText(html, 'html'))
            
            # Generate and attach trend charts for tables with issues
            for table, table_summary in summary['tables'].items():
                if table_summary['status'] in ('FAIL', 'WARNING'):
                    metrics = self.get_historical_metrics(table=table, days=30)
                    if metrics:
                        chart_buffer = self.generate_trend_chart(
                            metrics, 
                            f"Data Quality Trends for {table}"
                        )
                        
                        if chart_buffer:
                            img = MIMEImage(chart_buffer.read())
                            img.add_header('Content-ID', f'<{table}_chart>')
                            img.add_header('Content-Disposition', 'inline', filename=f"{table}_trend.png")
                            msg.attach(img)
            
            # Send the email
            with smtplib.SMTP(self.config['email_notifications']['smtp_server'], 
                              self.config['email_notifications']['smtp_port']) as server:
                server.starttls()
                server.login(
                    self.config['email_notifications']['username'],
                    self.config['email_notifications']['password']
                )
                server.send_message(msg)
            
            logger.info("Email alert sent successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False
    
    def cleanup_old_metrics(self):
        """Clean up old metrics data based on retention policy"""
        if not self.config['history_tracking']['enabled']:
            return
        
        retention_days = self.config['history_tracking']['retention_days']
        cursor = self.conn.cursor()
        
        cursor.execute("""
        DELETE FROM data_quality_metrics
        WHERE timestamp < datetime('now', '-{} days')
        """.format(retention_days))
        
        deleted_count = cursor.rowcount
        self.conn.commit()
        
        logger.info(f"Cleaned up {deleted_count} old metrics records")
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

# Main function to run the monitor
def main():
    parser = argparse.ArgumentParser(description='Database Data Quality Monitor')
    parser.add_argument('--db', required=True, help='Database file path')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--output', help='Output report file path')
    args = parser.parse_args()
    
    # Initialize and run the monitor
    monitor = DataQualityMonitor(args.db, args.config)
    try:
        summary = monitor.run_all_checks()
        
        # Generate report if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Report saved to {args.output}")
        
        # Clean up old metrics
        monitor.cleanup_old_metrics()
        
        # Print summary to console
        print(f"Data quality check completed at {summary['timestamp']}")
        print(f"Tables checked: {summary['tables_checked']}")
        print(f"Issues found: {summary['has_issues']}")
        
        for table, table_summary in summary['tables'].items():
            print(f"\nTable: {table} - Status: {table_summary['status']}")
            for check_name, check_result in table_summary['checks'].items():
                print(f"  - {check_name}: {check_result['status']}")
    
    finally:
        monitor.close()

if __name__ == "__main__":
    main()
```

## Setting Up Data Quality Monitoring

To set up database monitoring in your organization:

1. **Customize Configuration**: Edit the JSON configuration to match your database structure:
   ```json
   {
     "tables": ["customers", "orders", "products"],
     "quality_checks": {
       "missing_values": true,
       "duplicate_records": true,
       "data_format": true,
       "referential_integrity": true,
       "value_distribution": true
     },
     "thresholds": {
       "missing_values_pct": 5.0,
       "duplicate_records_pct": 1.0
     },
     "email_notifications": {
       "enabled": true,
       "smtp_server": "smtp.company.com",
       "smtp_port": 587,
       "username": "data-quality@company.com",
       "password": "your-password",
       "recipients": ["data-team@company.com"]
     },
     "history_tracking": {
       "enabled": true,
       "retention_days": 90
     }
   }
   ```

2. **Create a Scheduled Task**: Set up a cron job or scheduled task to run the monitor:
   ```bash
   # Run daily at 2 AM
   0 2 * * * python data_quality_monitor.py --db /path/to/database.db --config /path/to/config.json --output /path/to/reports/report_$(date +\%Y\%m\%d).json
   ```

3. **Create a Quality Dashboard**: Use the historical metrics to create a data quality dashboard:
   ```python
   import sqlite3
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   from datetime import datetime, timedelta
   
   # Connect to the metrics database
   conn = sqlite3.connect('customer_data.db')
   
   # Get the data for the last 30 days
   end_date = datetime.now()
   start_date = end_date - timedelta(days=30)
   
   query = """
   SELECT * FROM data_quality_metrics
   WHERE timestamp BETWEEN ? AND ?
   ORDER BY timestamp
   """
   
   df = pd.read_sql_query(query, conn, params=(start_date.isoformat(), end_date.isoformat()))
   
   # Create dashboard visualizations
   # Example: Quality trend by table
   plt.figure(figsize=(12, 6))
   pivot = df.pivot_table(
       index=pd.to_datetime(df['timestamp']).dt.date,
       columns='table_name',
       values='metric_value',
       aggfunc='mean'
   )
   
   pivot.plot(figsize=(12, 6))
   plt.title('Data Quality Score by Table')
   plt.xlabel('Date')
   plt.ylabel('Quality Score')
   plt.grid(True)
   plt.savefig('quality_trend.png')
   
   # Close connection
   conn.close()
   ```

## Next Steps

After mastering database integration, you'll be ready to:

1. Build ETL (Extract, Transform, Load) pipelines for data governance
2. Implement advanced data quality monitoring systems
3. Create data lineage tracking for compliance
4. Develop custom data validation rules for business domains
5. Connect Python applications to enterprise databases

## Resources

- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [PostgreSQL Python Tutorial](https://www.postgresql.org/docs/current/tutorial-python.html)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Python Database API Specification](https://www.python.org/dev/peps/pep-0249/)
- [Data Governance with Python (Book)](https://www.oreilly.com/library/view/hands-on-data-governance/9781801810661/)

## Exercises and Projects

For additional practice, try these exercises:

1. Set up a data quality monitoring system for a sample database
2. Create a database migration script that validates data during transfer
3. Build a simple ETL pipeline that enforces data quality rules
4. Design a database schema with proper constraints for a data governance use case
5. Create a data lineage tracking system to document data flows

## Contributing

If you've found this guide helpful, consider contributing:
- Add examples for other database systems (Oracle, MongoDB, etc.)
- Share scripts for common data quality checks
- Suggest improvements or corrections

Happy database integrating!
