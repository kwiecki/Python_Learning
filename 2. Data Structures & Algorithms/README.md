# Data Structures & Algorithms for Data Analysts

Welcome to the Data Structures & Algorithms module! This guide focuses specifically on how Python data structures and algorithms can help data analysts and data governance professionals work more efficiently with datasets.

## Why Data Structures & Algorithms Matter for Data Work

Understanding data structures and algorithms provides several key benefits:
- Choose the right data structure for different analytical tasks
- Write more efficient code that processes large datasets faster
- Implement common data transformation operations
- Solve data quality and governance problems programmatically
- Automate repetitive data tasks with elegant solutions

## Module Overview

This module covers Python data structures and algorithms with practical applications for data analysis:

1. [Core Data Structures](#core-data-structures)
2. [Working with Collections](#working-with-collections)
3. [List and Dictionary Comprehensions](#list-and-dictionary-comprehensions)
4. [String Processing](#string-processing)
5. [Sorting and Searching](#sorting-and-searching)
6. [Working with Sets](#working-with-sets)
7. [Memory Management](#memory-management)
8. [Mini-Project: Data Deduplication Tool](#mini-project-data-deduplication-tool)

## Core Data Structures

### Lists

Lists are ordered, mutable collections ideal for storing records or values in sequence:

```python
# Using lists to store customer IDs by creation date
recent_customer_ids = [1005, 1006, 1007, 1008]

# Adding new customers
recent_customer_ids.append(1009)  # [1005, 1006, 1007, 1008, 1009]

# Inserting a customer at a specific position
recent_customer_ids.insert(0, 1004)  # [1004, 1005, 1006, 1007, 1008, 1009]

# Removing a customer
recent_customer_ids.remove(1006)  # [1004, 1005, 1007, 1008, 1009]

# Checking if a customer ID exists
if 1007 in recent_customer_ids:
    print("Customer #1007 is in the recent list")
```

### Tuples

Tuples are immutable sequences, useful for data that shouldn't change:

```python
# Storing record schema (field name, data type, required)
customer_schema = [
    ("customer_id", "string", True),
    ("company_name", "string", True),
    ("industry", "string", False),
    ("annual_revenue", "decimal", False),
    ("active", "boolean", True)
]

# Accessing tuple elements
first_field = customer_schema[0]  # ("customer_id", "string", True)
field_name = first_field[0]  # "customer_id"
is_required = first_field[2]  # True
```

### Dictionaries

Dictionaries map keys to values, perfect for lookup tables and records:

```python
# Storing a customer record
customer = {
    "customer_id": "ACME001",
    "company_name": "Acme Corporation",
    "industry": "Manufacturing",
    "annual_revenue": 1500000,
    "active": True
}

# Accessing values
name = customer["company_name"]  # "Acme Corporation"

# Safe access with get() to handle missing keys
contact_email = customer.get("contact_email", "No email provided")

# Checking if a key exists
if "industry" in customer:
    print(f"Customer industry: {customer['industry']}")

# Adding or updating values
customer["contact_email"] = "contact@acme.com"

# Removing a key-value pair
del customer["annual_revenue"]
```

### Sets

Sets store unique values, making them ideal for deduplication:

```python
# Finding unique values in a list of categories
categories = ["Retail", "Healthcare", "Retail", "Manufacturing", "Healthcare", "Technology"]
unique_categories = set(categories)  # {"Retail", "Healthcare", "Manufacturing", "Technology"}

# Checking if a value exists
if "Retail" in unique_categories:
    print("We have retail customers")

# Adding new values
unique_categories.add("Education")

# Removing values
unique_categories.remove("Manufacturing")
```

## Working with Collections

### Nested Data Structures

Most real-world data involves nested structures:

```python
# A list of customer dictionaries
customers = [
    {
        "customer_id": "ACME001",
        "company_name": "Acme Corporation",
        "contacts": [
            {"name": "John Smith", "email": "john@acme.com"},
            {"name": "Sarah Jones", "email": "sarah@acme.com"}
        ],
        "locations": ["New York", "Boston"]
    },
    {
        "customer_id": "TECH001",
        "company_name": "TechSolutions Inc.",
        "contacts": [
            {"name": "Lisa Brown", "email": "lisa@techsolutions.com"}
        ],
        "locations": ["San Francisco"]
    }
]

# Accessing nested data
first_customer_name = customers[0]["company_name"]
tech_contact_email = customers[1]["contacts"][0]["email"]

# Adding a new contact to the first customer
customers[0]["contacts"].append({"name": "Mike Johnson", "email": "mike@acme.com"})
```

### Collection Operations

Operations for processing groups of data:

```python
# Counting records by category
industry_counts = {}
for customer in customers:
    industry = customer.get("industry", "Unknown")
    if industry in industry_counts:
        industry_counts[industry] += 1
    else:
        industry_counts[industry] = 1

# Finding highest and lowest values
revenues = [c.get("annual_revenue", 0) for c in customers if "annual_revenue" in c]
highest_revenue = max(revenues)
lowest_revenue = min(revenues)
average_revenue = sum(revenues) / len(revenues) if revenues else 0

# Filtering records
active_customers = [c for c in customers if c.get("active", False)]
```

## List and Dictionary Comprehensions

Comprehensions provide concise syntax for creating and transforming collections:

### List Comprehensions

```python
# Extracting all customer IDs
customer_ids = [customer["customer_id"] for customer in customers]

# Filtering customers with revenue over $1M
high_value_customers = [c for c in customers if c.get("annual_revenue", 0) > 1000000]

# Creating a list of (ID, name) tuples
customer_names = [(c["customer_id"], c["company_name"]) for c in customers]

# Handling missing data with conditionals
active_status = ["Active" if c.get("active", False) else "Inactive" for c in customers]
```

### Dictionary Comprehensions

```python
# Creating a lookup by customer ID
customer_lookup = {c["customer_id"]: c for c in customers}

# Quick access to any customer by ID
acme_customer = customer_lookup["ACME001"]

# Creating a dictionary of company names by ID
company_names = {c["customer_id"]: c["company_name"] for c in customers}

# Filtering keys in a dictionary comprehension
active_customers = {id: name for id, name in company_names.items() 
                   if customer_lookup[id].get("active", False)}
```

### Nested Comprehensions

```python
# Flattening a list of all contacts from all customers
all_contacts = [contact for customer in customers 
                for contact in customer.get("contacts", [])]

# Creating a mapping of emails to customer IDs
email_to_customer = {contact["email"]: customer["customer_id"] 
                     for customer in customers
                     for contact in customer.get("contacts", [])}
```

## String Processing

String manipulation is critical for data cleaning:

```python
# Standardizing company names
def standardize_company_name(name):
    # Remove trailing spaces
    name = name.strip()
    
    # Convert to title case
    name = name.title()
    
    # Replace common abbreviations
    replacements = {
        " Inc": ", Inc.",
        " Llc": ", LLC",
        " Ltd": ", Ltd.",
        "Corporation": "Corp."
    }
    
    for old, new in replacements.items():
        if name.endswith(old):
            name = name[:-len(old)] + new
    
    return name

# Standardizing phone numbers
def standardize_phone(phone):
    # Remove non-digit characters
    digits = ''.join(c for c in phone if c.isdigit())
    
    # Format as (XXX) XXX-XXXX if it's a 10-digit number
    if len(digits) == 10:
        return f"({digits[0:3]}) {digits[3:6]}-{digits[6:]}"
    else:
        return phone  # Return original if not 10 digits
```

## Sorting and Searching

Organizing and finding data efficiently:

```python
# Sorting customers by revenue (highest first)
sorted_customers = sorted(customers, 
                         key=lambda c: c.get("annual_revenue", 0),
                         reverse=True)

# Sorting by multiple criteria (active status, then name)
sorted_customers = sorted(customers,
                         key=lambda c: (not c.get("active", False),  # False values come last
                                      c.get("company_name", "")))

# Binary search (for sorted lists)
def binary_search(sorted_list, target, key=lambda x: x):
    """
    Find an item in a sorted list.
    
    Args:
        sorted_list: A list sorted in ascending order
        target: The value to search for
        key: A function to extract the comparison value
        
    Returns:
        The index of the target or -1 if not found
    """
    left, right = 0, len(sorted_list) - 1
    
    while left <= right:
        mid = (left + right) // 2
        mid_val = key(sorted_list[mid])
        
        if mid_val == target:
            return mid
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return -1
```

## Working with Sets

Sets are powerful for data comparison operations:

```python
# Finding customers with missing data
complete_fields = {"customer_id", "company_name", "industry", "annual_revenue"}

customers_missing_data = []
for customer in customers:
    customer_fields = set(customer.keys())
    missing_fields = complete_fields - customer_fields
    
    if missing_fields:
        customers_missing_data.append({
            "customer_id": customer.get("customer_id", "Unknown"),
            "missing_fields": missing_fields
        })

# Finding common categories between two datasets
dataset1_categories = {"Retail", "Healthcare", "Manufacturing"}
dataset2_categories = {"Healthcare", "Technology", "Retail", "Education"}

common_categories = dataset1_categories.intersection(dataset2_categories)
unique_to_dataset1 = dataset1_categories - dataset2_categories
all_categories = dataset1_categories.union(dataset2_categories)
```

## Memory Management

Efficiently handling large datasets:

```python
# Using generators for memory efficiency
def process_large_file(filename):
    """Process a large data file line by line instead of loading it all at once"""
    with open(filename, 'r') as f:
        for line in f:  # Reads one line at a time
            customer_data = line.strip().split(',')
            # Process each line...
            yield customer_data  # Returns processed data without loading entire file

# Example usage
for customer in process_large_file("huge_customer_dataset.csv"):
    # Process each customer without loading the entire file
    print(f"Processing customer: {customer[0]}")
```

## Mini-Project: Data Deduplication Tool

Let's combine what we've learned to create a data deduplication tool:

```python
def find_duplicate_customers(customers, match_fields=None):
    """
    Find potential duplicate customer records based on specified fields.
    
    Args:
        customers: List of customer dictionaries
        match_fields: List of fields to compare (default: company_name and email)
        
    Returns:
        List of groups of potential duplicate records
    """
    if match_fields is None:
        match_fields = ["company_name", "email"]
    
    # Create a dictionary to group records by match key
    duplicates = {}
    
    for customer in customers:
        # Create a match key based on specified fields
        match_values = []
        
        for field in match_fields:
            # Get the field value, normalize it for comparison
            value = customer.get(field, "")
            if isinstance(value, str):
                value = value.lower().strip()
            match_values.append(str(value))
        
        # Create a tuple key (tuples are immutable and can be dictionary keys)
        match_key = tuple(match_values)
        
        # Group records by match key
        if match_key in duplicates:
            duplicates[match_key].append(customer)
        else:
            duplicates[match_key] = [customer]
    
    # Filter out groups with only one record (no duplicates)
    return [group for group in duplicates.values() if len(group) > 1]

def merge_duplicate_records(duplicates):
    """
    Merge each group of duplicate records into a single record.
    
    Args:
        duplicates: List of groups of duplicate records
        
    Returns:
        List of merged records
    """
    merged_records = []
    
    for group in duplicates:
        # Start with the first record as the base
        merged = group[0].copy()
        
        # For each subsequent record, merge in non-empty fields
        for record in group[1:]:
            for key, value in record.items():
                # If the key doesn't exist in the merged record, add it
                if key not in merged:
                    merged[key] = value
                # If the value in the merged record is empty, use this value
                elif not merged[key] and value:
                    merged[key] = value
        
        # Add a note about the merge
        merged["_merged"] = True
        merged["_duplicate_count"] = len(group)
        
        merged_records.append(merged)
    
    return merged_records

# Example usage
import csv

def deduplicate_customer_file(input_file, output_file, match_fields=None):
    """
    Read a CSV file, find and merge duplicate records, and write results.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        match_fields: Fields to use for matching duplicates
    """
    # Read the CSV file
    customers = []
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            customers.append(row)
    
    print(f"Read {len(customers)} records from {input_file}")
    
    # Find duplicates
    duplicate_groups = find_duplicate_customers(customers, match_fields)
    duplicate_count = sum(len(group) for group in duplicate_groups) - len(duplicate_groups)
    
    print(f"Found {duplicate_count} duplicate records in {len(duplicate_groups)} groups")
    
    # Create a set of all duplicate record IDs
    duplicate_ids = set()
    for group in duplicate_groups:
        for record in group:
            if "id" in record:
                duplicate_ids.add(record["id"])
    
    # Merge duplicates
    merged_records = merge_duplicate_records(duplicate_groups)
    
    # Create the final dataset with duplicates removed and merged records added
    final_records = [r for r in customers if r.get("id") not in duplicate_ids]
    final_records.extend(merged_records)
    
    # Write the output file
    if final_records:
        with open(output_file, 'w', newline='') as csvfile:
            # Use all fields from the first record to define the fieldnames
            fieldnames = list(final_records[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for record in final_records:
                writer.writerow(record)
    
    print(f"Wrote {len(final_records)} records to {output_file}")
    print(f"Removed {duplicate_count} duplicates")
```

## Next Steps

After mastering these data structures and algorithms, you'll be ready to:

1. Use pandas for more advanced data manipulation
2. Work with NumPy for numerical operations
3. Implement more sophisticated algorithms for data cleaning
4. Create custom data transformation pipelines

## Resources

- [Python Data Structures Tutorial](https://realpython.com/python-data-structures/)
- [Problem Solving with Algorithms and Data Structures](https://runestone.academy/runestone/books/published/pythonds/index.html)
- [Data Structures for Data Science](https://towardsdatascience.com/data-structures-for-data-science-b0c6a78ab839)

## Exercises and Projects

1. Create a script that finds and removes duplicate customer records
2. Build a tool that standardizes company names and addresses
3. Write a program that matches records between two datasets based on fuzzy matching
4. Implement a data validation system that uses efficient lookup structures

## Contributing

If you've found this guide helpful, consider contributing:
- Add new examples relevant to data governance and quality
- Share additional algorithms for common data tasks
- Suggest improvements or corrections

Happy coding!
