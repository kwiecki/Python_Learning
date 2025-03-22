# API Development & Integration for Data Professionals

Welcome to the API Development & Integration module! This guide focuses on how data professionals can leverage APIs to access external data sources and build APIs to serve data - essential skills for modern data governance and analytics work.

## Why APIs Matter for Data Professionals

APIs (Application Programming Interfaces) are crucial for data work because they:

- Provide programmatic access to data from external systems
- Enable real-time data exchange between applications
- Support data integration across disparate systems
- Allow automated data quality monitoring
- Make data available to authorized consumers in a controlled manner
- Support data governance through access controls and usage tracking
- Reduce data duplication through centralized data services

## Module Overview

This module covers API integration from both consumer and provider perspectives:

1. [REST API Fundamentals](#rest-api-fundamentals)
2. [Making HTTP Requests](#making-http-requests)
3. [Working with API Responses](#working-with-api-responses)
4. [Authentication and Security](#authentication-and-security)
5. [Building APIs with Flask](#building-apis-with-flask)
6. [API Documentation](#api-documentation)
7. [Rate Limiting and Error Handling](#rate-limiting-and-error-handling)
8. [Mini-Project: Data Quality API](#mini-project-data-quality-api)

## REST API Fundamentals

Understanding the basics of REST APIs:

```python
"""
REST (Representational State Transfer) is an architectural style for designing networked applications. 
RESTful APIs typically use HTTP methods explicitly:

- GET: Retrieve a resource or collection of resources
- POST: Create a new resource
- PUT: Update a resource by replacing it entirely
- PATCH: Update a resource partially
- DELETE: Remove a resource

Resource paths follow a hierarchy, such as:
- /customers            # Collection of all customers
- /customers/123        # A specific customer with ID 123
- /customers/123/orders # Orders belonging to customer 123

Responses typically return:
- Status codes (200 OK, 201 Created, 400 Bad Request, 401 Unauthorized, 404 Not Found, etc.)
- JSON formatted data
- Headers with metadata about the response
"""
```

## Making HTTP Requests

Using Python to interact with APIs:

```python
import requests

# Basic GET request
def get_customer(customer_id):
    """Get customer data from the API"""
    url = f"https://api.example.com/customers/{customer_id}"
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

# Passing parameters in a GET request
def search_customers(name=None, industry=None, limit=10):
    """Search for customers with optional filters"""
    url = "https://api.example.com/customers"
    
    # Build query parameters
    params = {
        'limit': limit
    }
    
    if name:
        params['name'] = name
    
    if industry:
        params['industry'] = industry
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

# POST request to create a resource
def create_customer(name, email, industry):
    """Create a new customer"""
    url = "https://api.example.com/customers"
    
    # Data to send in the request body
    data = {
        "name": name,
        "email": email,
        "industry": industry
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 201:  # 201 Created
        return response.json()
    else:
        print(f"Error creating customer: {response.status_code}")
        print(response.text)
        return None

# PUT request to update a resource
def update_customer(customer_id, name=None, email=None, industry=None):
    """Update a customer's information"""
    url = f"https://api.example.com/customers/{customer_id}"
    
    # First, get the current data
    current_data = get_customer(customer_id)
    if not current_data:
        return None
    
    # Update the data with new values
    if name:
        current_data['name'] = name
    if email:
        current_data['email'] = email
    if industry:
        current_data['industry'] = industry
    
    # Send the full updated resource
    response = requests.put(url, json=current_data)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error updating customer: {response.status_code}")
        print(response.text)
        return None

# DELETE request
def delete_customer(customer_id):
    """Delete a customer"""
    url = f"https://api.example.com/customers/{customer_id}"
    
    response = requests.delete(url)
    
    if response.status_code == 204:  # 204 No Content
        return True
    else:
        print(f"Error deleting customer: {response.status_code}")
        print(response.text)
        return False

# Handling timeouts and retries
def get_with_retry(url, max_retries=3, timeout=5):
    """Make a GET request with retry logic"""
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, timeout=timeout)
            return response
        except requests.exceptions.ConnectionError:
            print(f"Connection error, retrying ({retries+1}/{max_retries})...")
            retries += 1
        except requests.exceptions.Timeout:
            print(f"Request timed out, retrying ({retries+1}/{max_retries})...")
            retries += 1
    
    print("Maximum retries reached")
    return None
```

## Working with API Responses

Processing and handling different response formats:

```python
import requests
import json
import pandas as pd
import xml.etree.ElementTree as ET
import csv
from io import StringIO

# Handling JSON responses (most common)
def process_json_response():
    """Process a JSON API response"""
    response = requests.get("https://api.example.com/customers")
    
    if response.status_code == 200:
        # Parse JSON response
        data = response.json()
        
        # Process the data
        if isinstance(data, list):
            # Response is a list of items
            print(f"Received {len(data)} items")
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(data)
            print(df.head())
            
            # Calculate summary statistics
            if 'revenue' in df.columns:
                print(f"Average revenue: {df['revenue'].mean()}")
                print(f"Maximum revenue: {df['revenue'].max()}")
        
        elif isinstance(data, dict):
            # Response is a single object or a object with metadata
            if 'items' in data:
                # Paginated response with items array
                items = data['items']
                total = data.get('total', len(items))
                print(f"Received {len(items)} of {total} total items")
                
                # Process metadata
                if 'meta' in data:
                    print("Metadata:", data['meta'])
            else:
                # Single object response
                print("Received single object:", list(data.keys()))
    else:
        print(f"Error: {response.status_code}")

# Handling XML responses
def process_xml_response():
    """Process an XML API response"""
    response = requests.get("https://api.example.com/data.xml")
    
    if response.status_code == 200:
        # Parse XML response
        root = ET.fromstring(response.text)
        
        # Extract data from XML
        items = []
        for item in root.findall('./item'):
            item_data = {}
            for child in item:
                item_data[child.tag] = child.text
            items.append(item_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(items)
        print(df.head())
    else:
        print(f"Error: {response.status_code}")

# Handling CSV responses
def process_csv_response():
    """Process a CSV API response"""
    response = requests.get("https://api.example.com/data.csv")
    
    if response.status_code == 200:
        # Parse CSV response
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        
        # Process the DataFrame
        print(f"Received {len(df)} rows and {len(df.columns)} columns")
        print(df.head())
    else:
        print(f"Error: {response.status_code}")

# Handling binary responses (e.g., file downloads)
def download_file(url, local_filename):
    """Download a file from an API"""
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File downloaded to {local_filename}")
        return True
    else:
        print(f"Error downloading file: {response.status_code}")
        return False

# Handling paginated responses
def get_all_paginated_data(base_url, params=None):
    """Fetch all data from a paginated API endpoint"""
    if params is None:
        params = {}
    
    all_data = []
    page = 1
    more_pages = True
    
    while more_pages:
        # Update params with current page
        params['page'] = page
        
        # Make the request
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check response structure (varies by API)
            if isinstance(data, dict) and 'items' in data:
                items = data['items']
                all_data.extend(items)
                
                # Check if there are more pages
                if 'next_page' in data and data['next_page']:
                    page += 1
                else:
                    more_pages = False
            
            elif isinstance(data, list):
                # If the API returns an array directly
                if len(data) > 0:
                    all_data.extend(data)
                    page += 1
                else:
                    # Empty array means no more data
                    more_pages = False
        else:
            print(f"Error: {response.status_code}")
            more_pages = False
    
    print(f"Retrieved {len(all_data)} total items")
    return all_data
```

## Authentication and Security

Handling API authentication and security:

```python
import requests
import base64
import hashlib
import hmac
import time
import jwt  # pip install PyJWT
from requests.auth import HTTPBasicAuth
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

# Basic Authentication
def basic_auth_request(url, username, password):
    """Make a request with Basic Authentication"""
    response = requests.get(url, auth=HTTPBasicAuth(username, password))
    return response

# API Key Authentication (in header)
def api_key_request(url, api_key, key_name='X-API-Key'):
    """Make a request with an API key in the header"""
    headers = {key_name: api_key}
    response = requests.get(url, headers=headers)
    return response

# API Key Authentication (in query parameter)
def api_key_param_request(url, api_key, param_name='api_key'):
    """Make a request with an API key as a query parameter"""
    params = {param_name: api_key}
    response = requests.get(url, params=params)
    return response

# JWT (JSON Web Token) Authentication
def jwt_request(url, token):
    """Make a request with a JWT token"""
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(url, headers=headers)
    return response

# OAuth 2.0 Client Credentials Flow
def get_oauth2_token(token_url, client_id, client_secret):
    """Get an OAuth 2.0 token using client credentials flow"""
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)
    token = oauth.fetch_token(
        token_url=token_url,
        client_id=client_id,
        client_secret=client_secret
    )
    return token

def oauth2_request(url, token):
    """Make a request with OAuth 2.0 authentication"""
    headers = {'Authorization': f'Bearer {token["access_token"]}'}
    response = requests.get(url, headers=headers)
    return response

# HMAC Authentication (used by some APIs like AWS)
def generate_hmac_signature(secret_key, message):
    """Generate an HMAC signature"""
    byte_key = secret_key.encode('utf-8')
    message = message.encode('utf-8')
    return base64.b64encode(hmac.new(byte_key, message, hashlib.sha256).digest()).decode('utf-8')

def hmac_request(url, access_key, secret_key, method='GET'):
    """Make a request with HMAC authentication (simplified example)"""
    timestamp = str(int(time.time()))
    
    # String to sign (varies by API)
    string_to_sign = f"{method}\n{url}\n{timestamp}\n{access_key}"
    
    # Generate signature
    signature = generate_hmac_signature(secret_key, string_to_sign)
    
    # Build headers
    headers = {
        'AccessKey': access_key,
        'Timestamp': timestamp,
        'Signature': signature
    }
    
    # Make the request
    response = requests.get(url, headers=headers)
    return response

# Handling token refresh
class TokenManager:
    """Manage OAuth 2.0 tokens with automatic refresh"""
    
    def __init__(self, token_url, client_id, client_secret):
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.token_expiry = 0
    
    def get_valid_token(self):
        """Get a valid token, refreshing if necessary"""
        current_time = time.time()
        
        # Check if token is missing or expired (with 60-second buffer)
        if not self.token or self.token_expiry <= current_time + 60:
            self.refresh_token()
        
        return self.token['access_token']
    
    def refresh_token(self):
        """Fetch a new token"""
        client = BackendApplicationClient(client_id=self.client_id)
        oauth = OAuth2Session(client=client)
        self.token = oauth.fetch_token(
            token_url=self.token_url,
            client_id=self.client_id,
            client_secret=self.client_secret
        )
        
        # Calculate expiry time
        self.token_expiry = time.time() + self.token['expires_in']
        print(f"Token refreshed, expires in {self.token['expires_in']} seconds")

# Store credentials securely
def get_credentials(service_name):
    """Get credentials from a secure source (environment variables, vault, etc.)"""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get credentials based on service name
    if service_name == 'example_api':
        return {
            'api_key': os.getenv('EXAMPLE_API_KEY'),
            'api_secret': os.getenv('EXAMPLE_API_SECRET')
        }
    elif service_name == 'oauth_service':
        return {
            'client_id': os.getenv('OAUTH_CLIENT_ID'),
            'client_secret': os.getenv('OAUTH_CLIENT_SECRET')
        }
    else:
        raise ValueError(f"Unknown service: {service_name}")
```

## Building APIs with Flask

Creating your own APIs to serve data:

```python
from flask import Flask, request, jsonify
import sqlite3
import json
from functools import wraps
import jwt
import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
DATABASE = 'customer_data.db'

# Helper function to get database connection
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# JWT Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check if token is in headers
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            # Verify the token
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            # You could use the data to fetch the current user from a database
        except:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(*args, **kwargs)
    
    return decorated

# Authentication endpoint
@app.route('/login', methods=['POST'])
def login():
    auth = request.json
    
    if not auth or not auth.get('username') or not auth.get('password'):
        return jsonify({'message': 'Could not verify'}), 401
    
    # In a real app, check credentials against a database
    if auth.get('username') == 'admin' and auth.get('password') == 'password':
        # Generate token
        token = jwt.encode({
            'user': auth.get('username'),
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm="HS256")
        
        return jsonify({'token': token})
    
    return jsonify({'message': 'Invalid credentials'}), 401

# GET endpoint for all customers
@app.route('/api/customers', methods=['GET'])
@token_required
def get_customers():
    conn = get_db_connection()
    
    # Handle query parameters for filtering
    industry = request.args.get('industry')
    limit = request.args.get('limit', default=100, type=int)
    
    query = "SELECT * FROM customers"
    params = []
    
    if industry:
        query += " WHERE industry = ?"
        params.append(industry)
    
    query += f" LIMIT {limit}"
    
    cursor = conn.cursor()
    cursor.execute(query, params)
    
    customers = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify(customers)

# GET endpoint for a specific customer
@app.route('/api/customers/<int:customer_id>', methods=['GET'])
@token_required
def get_customer(customer_id):
    conn = get_db_connection()
    
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM customers WHERE id = ?", (customer_id,))
    
    customer = cursor.fetchone()
    conn.close()
    
    if customer is None:
        return jsonify({'message': 'Customer not found'}), 404
    
    return jsonify(dict(customer))

# POST endpoint to create a customer
@app.route('/api/customers', methods=['POST'])
@token_required
def create_customer():
    if not request.json:
        return jsonify({'message': 'No data provided'}), 400
    
    data = request.json
    
    # Validate required fields
    required_fields = ['name', 'email', 'industry']
    for field in required_fields:
        if field not in data:
            return jsonify({'message': f'Missing required field: {field}'}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "INSERT INTO customers (name, email, industry, active) VALUES (?, ?, ?, ?)",
            (data['name'], data['email'], data['industry'], data.get('active', True))
        )
        conn.commit()
        
        # Get the ID of the newly created customer
        customer_id = cursor.lastrowid
        
        # Return the created customer
        cursor.execute("SELECT * FROM customers WHERE id = ?", (customer_id,))
        customer = dict(cursor.fetchone())
        
        conn.close()
        return jsonify(customer), 201
    
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'message': 'Email already exists'}), 400
    
    except Exception as e:
        conn.close()
        return jsonify({'message': str(e)}), 500

# PUT endpoint to update a customer
@app.route('/api/customers/<int:customer_id>', methods=['PUT'])
@token_required
def update_customer(customer_id):
    if not request.json:
        return jsonify({'message': 'No data provided'}), 400
    
    data = request.json
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if customer exists
    cursor.execute("SELECT * FROM customers WHERE id = ?", (customer_id,))
    if not cursor.fetchone():
        conn.close()
        return jsonify({'message': 'Customer not found'}), 404
    
    # Update the customer
    try:
        update_fields = []
        params = []
        
        for field in ['name', 'email', 'industry', 'active']:
            if field in data:
                update_fields.append(f"{field} = ?")
                params.append(data[field])
        
        if not update_fields:
            conn.close()
            return jsonify({'message': 'No fields to update'}), 400
        
        params.append(customer_id)
        
        query = f"UPDATE customers SET {', '.join(update_fields)} WHERE id = ?"
        cursor.execute(query, params)
        conn.commit()
        
        # Return the updated customer
        cursor.execute("SELECT * FROM customers WHERE id = ?", (customer_id,))
        customer = dict(cursor.fetchone())
        
        conn.close()
        return jsonify(customer)
    
    except Exception as e:
        conn.close()
        return jsonify({'message': str(e)}), 500

# DELETE endpoint to delete a customer
@app.route('/api/customers/<int:customer_id>', methods=['DELETE'])
@token_required
def delete_customer(customer_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if customer exists
    cursor.execute("SELECT * FROM customers WHERE id = ?", (customer_id,))
    if not cursor.fetchone():
        conn.close()
        return jsonify({'message': 'Customer not found'}), 404
    
    # Delete the customer
    cursor.execute("DELETE FROM customers WHERE id = ?", (customer_id,))
    conn.commit()
    conn.close()
    
    return '', 204

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.datetime.now().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'message': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'message': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

## API Documentation

Creating documentation for your APIs:

```python
# Using Flask-RESTX for automatic API documentation
from flask import Flask
from flask_restx import Api, Resource, fields

app = Flask(__name__)
api = Api(app, version='1.0', title='Customer API',
          description='A simple Customer API')

# Define namespaces
ns = api.namespace('customers', description='Customer operations')

# Define models (for documentation and validation)
customer_model = api.model('Customer', {
    'id': fields.Integer(readonly=True, description='Customer ID'),
    'name': fields.String(required=True, description='Customer name'),
    'email': fields.String(required=True, description='Customer email'),
    'industry': fields.String(required=True, description='Customer industry'),
    'active': fields.Boolean(default=True, description='Customer status')
})

customer_list_model = api.model('CustomerList', {
    'customers': fields.List(fields.Nested(customer_model)),
    'count': fields.Integer
})

# Sample data
customers = [
    {'id': 1, 'name': 'Acme Inc.', 'email': 'contact@acme.com', 'industry': 'Manufacturing', 'active': True},
    {'id': 2, 'name': 'TechCorp', 'email': 'info@techcorp.com', 'industry': 'Technology', 'active': True}
]

@ns.route('/')
class CustomerList(Resource):
    @ns.doc('list_customers')
    @ns.marshal_with(customer_list_model)
    @ns.param('industry', 'Filter by industry')
    def get(self):
        """List all customers"""
        industry = api.request.args.get('industry')
        if industry:
            filtered = [c for c in customers if c['industry'] == industry]
            return {'customers': filtered, 'count': len(filtered)}
        return {'customers': customers, 'count': len(customers)}
    
    @ns.doc('create_customer')
    @ns.expect(customer_model)
    @ns.marshal_with(customer_model, code=201)
    def post(self):
        """Create a new customer"""
        data = api.payload
        new_id = max(c['id'] for c in customers) + 1 if customers else 1
        customer = {
            'id': new_id,
            'name': data['name'],
            'email': data['email'],
            'industry': data['industry'],
            'active': data.get('active', True)
        }
        customers.append(customer)
        return customer, 201

@ns.route('/<int:id>')
@ns.param('id', 'The customer identifier')
@ns.response(404, 'Customer not found')
class Customer(Resource):
    @ns.doc('get_customer')
    @ns.marshal_with(customer_model)
    def get(self, id):
        """Fetch a customer by ID"""
        for customer in customers:
            if customer['id'] == id:
                return customer
        api.abort(404, "Customer {} doesn't exist".format(id))
    
    @ns.doc('update_customer')
    @ns.expect(customer_model)
    @ns.marshal_with(customer_model)
    def put(self, id):
        """Update a customer"""
        for i, customer in enumerate(customers):
            if customer['id'] == id:
                data = api.payload
                customers[i] = {
                    'id': id,
                    'name': data['name'],
                    'email': data['email'],
                    'industry': data['industry'],
                    'active': data.get('active', True)
                }
                return customers[i]
        api.abort(404, "Customer {} doesn't exist".format(id))
    
    @ns.doc('delete_customer')
    @ns.response(204, 'Customer deleted')
    def delete(self, id):
        """Delete a customer"""
        for i, customer in enumerate(customers):
            if customer['id'] == id:
                del customers[i]
                return '', 204
        api.abort(404, "Customer {} doesn't exist".format(id))

if __name__ == '__main__':
    app.run(debug=True)

# Alternatively, you can document your API with OpenAPI/Swagger
"""
openapi: 3.0.0
info:
  title: Customer API
  description: A simple API for managing customer data
  version: 1.0.0
paths:
  /api/customers:
    get:
      summary: Returns a list of customers
      parameters:
        - in: query
          name: industry
          schema:
            type: string
          description: Filter by industry
        - in: query
          name: limit
          schema:
            type: integer
            default: 100
          description: Maximum number of records to return
      responses:
        '200':
          description: A JSON array of customers
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Customer'
    post:
      summary: Creates a new customer
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CustomerInput'
      responses:
        '201':
          description: Created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Customer'
        '400':
          description: Bad request
  /api/customers/{customerId}:
    get:
      summary: Returns a customer by ID
      parameters:
        - in: path
          name: customerId
          required: true
          schema:
            type: integer
          description: The customer ID
      responses:
        '200':
          description: A customer object
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Customer'
        '404':
          description: Customer not found
components:
  schemas:
    Customer:
      type: object
      properties:
        id:
          type: integer
          description: The customer ID
        name:
          type: string
          description: The customer name
        email:
          type: string
          description: The customer email
        industry:
          type: string
          description: The customer industry
        active:
          type: boolean
          description: Whether the customer is active
      required:
        - id
        - name
        - email
        - industry
    CustomerInput:
      type: object
      properties:
        name:
          type: string
          description: The customer name
        email:
          type: string
          description: The customer email
        industry:
          type: string
          description: The customer industry
        active:
          type: boolean
          description: Whether the customer is active
      required:
        - name
        - email
        - industry
"""
```

## Rate Limiting and Error Handling

Managing API usage and handling errors gracefully:

```python
from flask import Flask, request, jsonify
import time
import sqlite3
import json
from functools import wraps
import logging

app = Flask(__name__)
DATABASE = 'customer_data.db'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='api.log'
)
logger = logging.getLogger('customer_api')

# Simple in-memory rate limiter
class RateLimiter:
    def __init__(self, limit=100, window=3600):
        self.limit = limit  # Max requests per window
        self.window = window  # Time window in seconds
        self.clients = {}
    
    def is_allowed(self, client_id):
        current_time = time.time()
# Rate Limiting and Error Handling (continued)

```python
    def is_allowed(self, client_id):
        current_time = time.time()
        
        if client_id not in self.clients:
            self.clients[client_id] = {
                'requests': 1,
                'window_start': current_time
            }
            return True
        
        client_data = self.clients[client_id]
        
        # Check if the window has reset
        time_passed = current_time - client_data['window_start']
        if time_passed > self.window:
            # Reset window
            client_data['requests'] = 1
            client_data['window_start'] = current_time
            return True
        
        # Check if limit is reached
        if client_data['requests'] >= self.limit:
            return False
        
        # Increment request count
        client_data['requests'] += 1
        return True
    
    def get_remaining(self, client_id):
        """Get remaining requests in the current window"""
        if client_id not in self.clients:
            return self.limit
        
        client_data = self.clients[client_id]
        current_time = time.time()
        
        # Check if the window has reset
        time_passed = current_time - client_data['window_start']
        if time_passed > self.window:
            return self.limit
        
        return max(0, self.limit - client_data['requests'])


# Initialize rate limiter
rate_limiter = RateLimiter(limit=100, window=3600)  # 100 requests per hour

# Rate limiting middleware
def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Identify client (in production, use API key or IP address)
        client_id = request.headers.get('X-API-Key', request.remote_addr)
        
        # Check if client is allowed
        if not rate_limiter.is_allowed(client_id):
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': 'Too many requests, please try again later'
            }), 429
        
        # Add rate limit headers
        remaining = rate_limiter.get_remaining(client_id)
        response = f(*args, **kwargs)
        
        # If response is a tuple (response, status_code)
        if isinstance(response, tuple):
            resp_obj, status_code = response
            resp_obj.headers['X-RateLimit-Limit'] = str(rate_limiter.limit)
            resp_obj.headers['X-RateLimit-Remaining'] = str(remaining)
            return resp_obj, status_code
        
        # If response is just a response object
        response.headers['X-RateLimit-Limit'] = str(rate_limiter.limit)
        response.headers['X-RateLimit-Remaining'] = str(remaining)
        return response
    
    return decorated

# Helper function to get database connection
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# Error handler for database errors
def handle_db_error(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except sqlite3.Error as e:
            logger.error(f"Database error: {str(e)}")
            return jsonify({
                'error': 'Database error',
                'message': 'A database error occurred'
            }), 500
    
    return decorated

# Example API route with rate limiting and error handling
@app.route('/api/customers', methods=['GET'])
@rate_limit
@handle_db_error
def get_customers():
    conn = get_db_connection()
    
    try:
        # Handle query parameters for filtering
        industry = request.args.get('industry')
        limit = request.args.get('limit', default=100, type=int)
        
        query = "SELECT * FROM customers"
        params = []
        
        if industry:
            query += " WHERE industry = ?"
            params.append(industry)
        
        query += f" LIMIT {limit}"
        
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        customers = [dict(row) for row in cursor.fetchall()]
        
        # Log successful request
        logger.info(f"Retrieved {len(customers)} customers")
        
        return jsonify(customers)
    
    finally:
        conn.close()

# Global error handlers
@app.errorhandler(400)
def bad_request(error):
    logger.warning(f"Bad request: {error}")
    return jsonify({
        'error': 'Bad request',
        'message': 'The request was malformed'
    }), 400

@app.errorhandler(404)
def not_found(error):
    logger.warning(f"Resource not found: {error}")
    return jsonify({
        'error': 'Not found',
        'message': 'The requested resource was not found'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    logger.warning(f"Method not allowed: {error}")
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The HTTP method is not allowed for this resource'
    }), 405

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

# Request logging middleware
@app.before_request
def log_request():
    logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def log_response(response):
    logger.info(f"Response: {response.status_code}")
    return response

# Client-side rate limiting and error handling
def api_request_with_backoff(url, max_retries=5, initial_backoff=1):
    """Make a request with exponential backoff for rate limits"""
    import requests
    import time
    
    retries = 0
    backoff = initial_backoff
    
    while retries < max_retries:
        try:
            response = requests.get(url)
            
            # If rate limited, wait and retry
            if response.status_code == 429:
                # Get retry-after header if available
                retry_after = response.headers.get('Retry-After')
                if retry_after and retry_after.isdigit():
                    wait_time = int(retry_after)
                else:
                    wait_time = backoff
                
                print(f"Rate limited. Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
                
                # Increase backoff for next retry (exponential backoff)
                backoff *= 2
                retries += 1
                continue
            
            # Handle other errors (4xx/5xx)
            if response.status_code >= 400:
                print(f"Error {response.status_code}: {response.text}")
                
                # Only retry on server errors (5xx)
                if response.status_code >= 500:
                    print(f"Server error. Retrying after {backoff} seconds...")
                    time.sleep(backoff)
                    backoff *= 2
                    retries += 1
                    continue
                else:
                    # Don't retry on client errors (4xx)
                    return response
            
            # Success
            return response
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            time.sleep(backoff)
            backoff *= 2
            retries += 1
    
    print(f"Max retries ({max_retries}) reached")
    return None
```

## WebSocket Communication

Real-time data exchange with WebSockets:

```python
# Server-side WebSocket implementation with Flask-SocketIO
from flask import Flask
from flask_socketio import SocketIO, emit
import json
import threading
import time
import random

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Connection event
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('welcome', {'message': 'Connected to data stream'})

# Disconnection event
@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# Custom event handler
@socketio.on('subscribe')
def handle_subscribe(data):
    """Handle client subscription to data channels"""
    if 'channel' not in data:
        emit('error', {'message': 'No channel specified'})
        return
    
    channel = data['channel']
    print(f"Client subscribed to {channel}")
    
    # Acknowledge subscription
    emit('subscribed', {
        'channel': channel,
        'message': f'Successfully subscribed to {channel}'
    })

# Example data generator for streaming
def generate_metric_data():
    """Generate random metric data for streaming"""
    while True:
        # Generate random quality metrics
        data = {
            'timestamp': time.time(),
            'metrics': {
                'completeness': round(random.uniform(90, 100), 2),
                'accuracy': round(random.uniform(85, 99), 2),
                'consistency': round(random.uniform(80, 98), 2),
                'timeliness': round(random.uniform(75, 95), 2)
            }
        }
        
        # Emit data to all connected clients subscribed to 'metrics'
        socketio.emit('metric_update', data, namespace='/')
        
        # Sleep for a bit
        time.sleep(5)

# Start the data generator in a background thread
@app.before_first_request
def start_data_generator():
    thread = threading.Thread(target=generate_metric_data)
    thread.daemon = True
    thread.start()

# Client-side WebSocket implementation with JavaScript
"""
// Using Socket.IO client library
const socket = io('http://localhost:5000');

// Connection established
socket.on('connect', () => {
    console.log('Connected to server');
    
    // Subscribe to metrics channel
    socket.emit('subscribe', { channel: 'metrics' });
});

// Handle subscription confirmation
socket.on('subscribed', (data) => {
    console.log(`Subscribed to ${data.channel}`);
});

// Handle incoming metric updates
socket.on('metric_update', (data) => {
    console.log('Received metric update:', data);
    
    // Update UI with new metrics
    updateMetricDisplay(data.metrics);
});

// Handle errors
socket.on('error', (data) => {
    console.error('Socket error:', data.message);
});

// Handle disconnection
socket.on('disconnect', () => {
    console.log('Disconnected from server');
});

// Function to update UI with metrics
function updateMetricDisplay(metrics) {
    document.getElementById('completeness').textContent = metrics.completeness.toFixed(2) + '%';
    document.getElementById('accuracy').textContent = metrics.accuracy.toFixed(2) + '%';
    document.getElementById('consistency').textContent = metrics.consistency.toFixed(2) + '%';
    document.getElementById('timeliness').textContent = metrics.timeliness.toFixed(2) + '%';
    
    // Update timestamp
    const now = new Date();
    document.getElementById('last-update').textContent = now.toLocaleTimeString();
}
"""

# Client-side WebSocket with Python
import socketio
import time

def handle_data_stream():
    """Connect to a WebSocket data stream and process updates"""
    # Initialize Socket.IO client
    sio = socketio.Client()
    
    # Define event handlers
    @sio.event
    def connect():
        print('Connected to server')
        # Subscribe to metrics channel
        sio.emit('subscribe', {'channel': 'metrics'})
    
    @sio.event
    def disconnect():
        print('Disconnected from server')
    
    @sio.on('subscribed')
    def on_subscribed(data):
        print(f"Subscribed to {data['channel']}")
    
    @sio.on('metric_update')
    def on_metric_update(data):
        print(f"Received metrics at {time.ctime(data['timestamp'])}:")
        for metric, value in data['metrics'].items():
            print(f"  {metric}: {value}%")
        
        # Process the data (e.g., store in database, trigger alerts)
        process_metrics(data['metrics'])
    
    # Connect to the server
    try:
        sio.connect('http://localhost:5000')
        sio.wait()
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        if sio.connected:
            sio.disconnect()

def process_metrics(metrics):
    """Process received metrics data"""
    # Check for quality issues
    alerts = []
    
    for metric, value in metrics.items():
        if value < 90:
            alerts.append(f"Low {metric}: {value}%")
    
    if alerts:
        print("ALERTS:", ", ".join(alerts))
        # Send notifications, update dashboards, etc.
```

## Mini-Project: Data Quality API

Let's combine what we've learned to create a data quality monitoring API:

```python
from flask import Flask, request, jsonify
import sqlite3
import json
import jwt
import datetime
import logging
from functools import wraps
import pandas as pd
import statistics

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
DATABASE = 'data_quality.db'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='data_quality_api.log'
)
logger = logging.getLogger('data_quality_api')

# Initialize database
def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        description TEXT,
        owner TEXT,
        created_date TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS quality_metrics (
        id INTEGER PRIMARY KEY,
        dataset_id INTEGER,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        metric_name TEXT NOT NULL,
        metric_value REAL,
        details TEXT,
        FOREIGN KEY (dataset_id) REFERENCES datasets(id)
    )
    ''')
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_dataset ON quality_metrics(dataset_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON quality_metrics(timestamp)')
    
    conn.commit()
    conn.close()
    
    logger.info("Database initialized")

# Helper function to get database connection
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# JWT Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check if token is in headers
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            # Verify the token
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['user']
        except:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

# Authentication endpoint
@app.route('/api/login', methods=['POST'])
def login():
    auth = request.json
    
    if not auth or not auth.get('username') or not auth.get('password'):
        return jsonify({'message': 'Could not verify'}), 401
    
    # In a real app, check credentials against a database
    if auth.get('username') == 'admin' and auth.get('password') == 'password':
        # Generate token
        token = jwt.encode({
            'user': auth.get('username'),
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm="HS256")
        
        return jsonify({'token': token})
    
    return jsonify({'message': 'Invalid credentials'}), 401

# Dataset management endpoints
@app.route('/api/datasets', methods=['GET'])
@token_required
def get_datasets(current_user):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all datasets
    cursor.execute('SELECT * FROM datasets')
    datasets = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return jsonify(datasets)

@app.route('/api/datasets/<int:dataset_id>', methods=['GET'])
@token_required
def get_dataset(current_user, dataset_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get dataset
    cursor.execute('SELECT * FROM datasets WHERE id = ?', (dataset_id,))
    dataset = cursor.fetchone()
    
    if not dataset:
        conn.close()
        return jsonify({'message': 'Dataset not found'}), 404
    
    # Get recent metrics
    cursor.execute('''
    SELECT * FROM quality_metrics 
    WHERE dataset_id = ? 
    ORDER BY timestamp DESC
    LIMIT 100
    ''', (dataset_id,))
    
    metrics = [dict(row) for row in cursor.fetchall()]
    
    result = dict(dataset)
    result['recent_metrics'] = metrics
    
    conn.close()
    return jsonify(result)

@app.route('/api/datasets', methods=['POST'])
@token_required
def create_dataset(current_user):
    data = request.json
    
    if not data or not data.get('name'):
        return jsonify({'message': 'Name is required'}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            'INSERT INTO datasets (name, description, owner) VALUES (?, ?, ?)',
            (data['name'], data.get('description', ''), current_user)
        )
        conn.commit()
        
        # Get the created dataset
        dataset_id = cursor.lastrowid
        cursor.execute('SELECT * FROM datasets WHERE id = ?', (dataset_id,))
        dataset = dict(cursor.fetchone())
        
        conn.close()
        logger.info(f"Dataset created: {dataset['name']} (ID: {dataset_id})")
        return jsonify(dataset), 201
    
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'message': 'Dataset name already exists'}), 400
    
    except Exception as e:
        conn.close()
        logger.error(f"Error creating dataset: {str(e)}")
        return jsonify({'message': str(e)}), 500

# Quality metrics endpoints
@app.route('/api/datasets/<int:dataset_id>/metrics', methods=['POST'])
@token_required
def add_metrics(current_user, dataset_id):
    data = request.json
    
    if not data or not isinstance(data, list):
        return jsonify({'message': 'Expected array of metrics'}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if dataset exists
    cursor.execute('SELECT id FROM datasets WHERE id = ?', (dataset_id,))
    if not cursor.fetchone():
        conn.close()
        return jsonify({'message': 'Dataset not found'}), 404
    
    # Insert metrics
    timestamp = datetime.datetime.now().isoformat()
    inserted_count = 0
    
    try:
        for metric in data:
            if 'name' not in metric or 'value' not in metric:
                continue
            
            details = json.dumps(metric.get('details', {}))
            
            cursor.execute(
                'INSERT INTO quality_metrics (dataset_id, timestamp, metric_name, metric_value, details) VALUES (?, ?, ?, ?, ?)',
                (dataset_id, timestamp, metric['name'], metric['value'], details)
            )
            inserted_count += 1
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added {inserted_count} metrics for dataset {dataset_id}")
        return jsonify({
            'message': f'Added {inserted_count} metrics',
            'timestamp': timestamp
        })
    
    except Exception as e:
        conn.rollback()
        conn.close()
        logger.error(f"Error adding metrics: {str(e)}")
        return jsonify({'message': str(e)}), 500

@app.route('/api/datasets/<int:dataset_id>/metrics/<string:metric_name>', methods=['GET'])
@token_required
def get_metric_history(current_user, dataset_id, metric_name):
    # Get query parameters
    days = request.args.get('days', default=30, type=int)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get historical metrics
    cursor.execute('''
    SELECT timestamp, metric_value, details
    FROM quality_metrics
    WHERE dataset_id = ? AND metric_name = ? AND timestamp >= datetime('now', '-? days')
    ORDER BY timestamp
    ''', (dataset_id, metric_name, days))
    
    metrics = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    if not metrics:
        return jsonify({
            'message': f'No metrics found for {metric_name} in the last {days} days'
        }), 404
    
    # Calculate statistics
    values = [m['metric_value'] for m in metrics]
    stats = {
        'count': len(values),
        'min': min(values),
        'max': max(values),
        'avg': sum(values) / len(values),
        'median': statistics.median(values) if values else None,
        'latest': values[-1] if values else None
    }
    
    return jsonify({
        'metric_name': metric_name,
        'dataset_id': dataset_id,
        'days': days,
        'history': metrics,
        'stats': stats
    })

# Data quality calculation endpoints
@app.route('/api/analyze', methods=['POST'])
@token_required
def analyze_data_quality(current_user):
    if not request.json:
        return jsonify({'message': 'No data provided'}), 400
    
    data = request.json
    
    if 'csv' not in data and 'json' not in data:
        return jsonify({'message': 'Either CSV or JSON data is required'}), 400
    
    try:
        # Parse the data into a DataFrame
        if 'csv' in data:
            import io
            df = pd.read_csv(io.StringIO(data['csv']))
        else:
            df = pd.DataFrame(data['json'])
        
        # Calculate quality metrics
        metrics = calculate_data_quality(df)
        
        # If a dataset_id is provided, store the metrics
        dataset_id = data.get('dataset_id')
        if dataset_id:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Check if dataset exists
            cursor.execute('SELECT id FROM datasets WHERE id = ?', (dataset_id,))
            if cursor.fetchone():
                # Store metrics
                timestamp = datetime.datetime.now().isoformat()
                for metric_name, metric_data in metrics.items():
                    cursor.execute(
                        'INSERT INTO quality_metrics (dataset_id, timestamp, metric_name, metric_value, details) VALUES (?, ?, ?, ?, ?)',
                        (dataset_id, timestamp, metric_name, metric_data['value'], json.dumps(metric_data['details']))
                    )
                
                conn.commit()
                logger.info(f"Stored metrics for dataset {dataset_id}")
            
            conn.close()
        
        return jsonify({
            'metrics': metrics,
            'row_count': len(df),
            'column_count': len(df.columns)
        })
    
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        return jsonify({'message': str(e)}), 500

def calculate_data_quality(df):
    """Calculate data quality metrics for a DataFrame"""
    metrics = {}
    
    # 1. Completeness
    null_counts = df.isnull().sum()
    total_cells = df.shape[0] * df.shape[1]
    null_cells = null_counts.sum()
    
    completeness_value = ((total_cells - null_cells) / total_cells) * 100 if total_cells > 0 else 0
    
    metrics['completeness'] = {
        'value': round(completeness_value, 2),
        'details': {
            'total_cells': int(total_cells),
            'null_cells': int(null_cells),
            'columns': {col: int(count) for col, count in null_counts.items()}
        }
    }
    
    # 2. Uniqueness
    duplicate_rows = df.duplicated().sum()
    uniqueness_value = ((df.shape[0] - duplicate_rows) / df.shape[0]) * 100 if df.shape[0] > 0 else 0
    
    metrics['uniqueness'] = {
        'value': round(uniqueness_value, 2),
        'details': {
            'total_rows': df.shape[0],
            'duplicate_rows': int(duplicate_rows)
        }
    }
    
    # 3. Consistency (check data types)
    mixed_type_columns = []
    for col in df.columns:
        try:
            # Check if column has mixed numeric and string values
            numeric_count = df[col].apply(lambda x: isinstance(x, (int, float))).sum()
            string_count = df[col].apply(lambda x: isinstance(x, str)).sum()
            
            if numeric_count > 0 and string_count > 0:
                mixed_type_columns.append(col)
        except:
            continue
    
    consistency_value = ((len(df.columns) - len(mixed_type_columns)) / len(df.columns)) * 100 if len(df.columns) > 0 else 0
    
    metrics['consistency'] = {
        'value': round(consistency_value, 2),
        'details': {
            'total_columns': len(df.columns),
            'mixed_type_columns': mixed_type_columns
        }
    }
    
    # 4. Validity (for numeric columns)
    numeric_columns = df.select_dtypes(include=['number']).columns
    outlier_counts = {}
    
    for col in numeric_columns:
        # Simple outlier detection using IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
        outlier_counts[col] = int(outliers)
    
    total_numeric_values = df[numeric_columns].count().sum()
    total_outliers = sum(outlier_counts.values())
    
    validity_value = ((total_numeric_values - total_outliers) / total_numeric_values) * 100 if total_numeric_values > 0 else 0
    
    metrics['validity'] = {
        'value': round(validity_value, 2),
        'details': {
            'total_numeric_values': int(total_numeric_values),
            'total_outliers': int(total_outliers),
            'columns': outlier_counts
        }
    }
    
    # 5. Overall quality score (average of all metrics)
    overall_score = sum(m['value'] for m in metrics.values()) / len(metrics)
    
    metrics['overall_quality'] = {
        'value': round(overall_score, 2),
        'details': {
            'component_scores': {name: m['value'] for name, m in metrics.items() if name != 'overall_quality'}
        }
    }
    
    return metrics

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'message': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'message': 'Internal server error'}), 500

# Initialize the database before running the app
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
```

## Using the Data Quality API

Here's how you can use the data quality API in Python:

```python
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt

class DataQualityClient:
    """Client for interacting with the Data Quality API"""
    
    def __init__(self, base_url='http://localhost:5000/api'):
        self.base_url = base_url
        self.token = None
    
    def login(self, username, password):
        """Authenticate with the API"""
        response = requests.post(
            f"{self.base_url}/login",
            json={'username': username, 'password': password}
        )
        
        if response.status_code == 200:
            self.token = response.json()['token']
            return True
        return False
    
    def _get_headers(self):
        """Get headers with authentication token"""
        if not self.token:
            raise ValueError("Not authenticated. Call login() first.")
        
        return {'Authorization': f'Bearer {self.token}'}
    
    def get_datasets(self):
        """Get all datasets"""
        response = requests.get(
            f"{self.base_url}/datasets",
            headers=self._get_headers()
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    
    def get_dataset(self, dataset_id):
        """Get a specific dataset with recent metrics"""
        response = requests.get(
            f"{self.base_url}/datasets/{dataset_id}",
            headers=self._get_headers()
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    
    def create_dataset(self, name, description=""):
        """Create a new dataset"""
        response = requests.post(
            f"{self.base_url}/datasets",
            headers=self._get_
def create_dataset(self, name, description=""):
        """Create a new dataset"""
        response = requests.post(
            f"{self.base_url}/datasets",
            headers=self._get_headers(),
            json={'name': name, 'description': description}
        )
        
        if response.status_code == 201:
            return response.json()
        return None
    
    def add_metrics(self, dataset_id, metrics):
        """Add quality metrics to a dataset"""
        response = requests.post(
            f"{self.base_url}/datasets/{dataset_id}/metrics",
            headers=self._get_headers(),
            json=metrics
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    
    def get_metric_history(self, dataset_id, metric_name, days=30):
        """Get historical data for a specific metric"""
        response = requests.get(
            f"{self.base_url}/datasets/{dataset_id}/metrics/{metric_name}?days={days}",
            headers=self._get_headers()
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    
    def analyze_data(self, data, dataset_id=None):
        """Analyze data quality for a dataset"""
        payload = {}
        
        # Handle different input types
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to CSV string
            payload['csv'] = data.to_csv(index=False)
        elif isinstance(data, list):
            # Assume list of dictionaries (JSON)
            payload['json'] = data
        elif isinstance(data, str):
            # Assume CSV string
            payload['csv'] = data
        else:
            raise ValueError("Unsupported data type. Use DataFrame, list of dicts, or CSV string.")
        
        # Set dataset ID if provided
        if dataset_id:
            payload['dataset_id'] = dataset_id
        
        response = requests.post(
            f"{self.base_url}/analyze",
            headers=self._get_headers(),
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    
    def plot_metric_history(self, dataset_id, metric_name, days=30):
        """Plot the history of a specific metric"""
        history = self.get_metric_history(dataset_id, metric_name, days)
        
        if not history:
            print(f"No data available for {metric_name}")
            return
        
        # Extract data
        timestamps = [item['timestamp'] for item in history['history']]
        values = [item['metric_value'] for item in history['history']]
        
        # Convert timestamps to datetime
        import datetime
        timestamps = [datetime.datetime.fromisoformat(ts) for ts in timestamps]
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, values, marker='o')
        plt.title(f"{metric_name} History - Last {days} Days")
        plt.xlabel('Date')
        plt.ylabel('Value (%)')
        plt.grid(True)
        plt.ylim(0, 100)
        
        # Add statistics
        stats = history['stats']
        plt.axhline(y=stats['avg'], color='r', linestyle='--', label=f"Avg: {stats['avg']:.2f}%")
        plt.axhline(y=stats['max'], color='g', linestyle=':', label=f"Max: {stats['max']:.2f}%")
        plt.axhline(y=stats['min'], color='orange', linestyle=':', label=f"Min: {stats['min']:.2f}%")
        
        plt.legend()
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = DataQualityClient()
    
    # Login
    if client.login('admin', 'password'):
        print("Authentication successful")
        
        # Create a new dataset
        dataset = client.create_dataset(
            name="Customer Data",
            description="Monthly customer data import"
        )
        print(f"Created dataset: {dataset['name']} (ID: {dataset['id']})")
        
        # Load data from CSV
        df = pd.read_csv('customer_data.csv')
        
        # Analyze data quality
        analysis = client.analyze_data(df, dataset_id=dataset['id'])
        
        if analysis:
            print("\nData Quality Analysis:")
            for name, metric in analysis['metrics'].items():
                print(f"- {name}: {metric['value']}%")
            
            # Plot a metric over time
            client.plot_metric_history(dataset['id'], 'completeness')
    else:
        print("Authentication failed")
```

## Documenting APIs with OpenAPI/Swagger

Creating standardized API documentation:

```python
"""
Example OpenAPI/Swagger specification for the Data Quality API:

openapi: 3.0.0
info:
  title: Data Quality API
  description: API for tracking and monitoring data quality metrics
  version: 1.0.0
servers:
  - url: http://localhost:5000/api
    description: Development server
paths:
  /login:
    post:
      summary: Authenticate with the API
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                username:
                  type: string
                password:
                  type: string
              required:
                - username
                - password
      responses:
        '200':
          description: Authentication successful
          content:
            application/json:
              schema:
                type: object
                properties:
                  token:
                    type: string
        '401':
          description: Authentication failed
  /datasets:
    get:
      summary: Get all datasets
      security:
        - bearerAuth: []
      responses:
        '200':
          description: List of datasets
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Dataset'
        '401':
          description: Unauthorized
    post:
      summary: Create a new dataset
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                name:
                  type: string
                description:
                  type: string
              required:
                - name
      responses:
        '201':
          description: Dataset created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Dataset'
        '400':
          description: Invalid input
        '401':
          description: Unauthorized
  /datasets/{datasetId}:
    get:
      summary: Get a specific dataset with recent metrics
      security:
        - bearerAuth: []
      parameters:
        - name: datasetId
          in: path
          required: true
          schema:
            type: integer
      responses:
        '200':
          description: Dataset details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DatasetWithMetrics'
        '404':
          description: Dataset not found
        '401':
          description: Unauthorized
  /datasets/{datasetId}/metrics:
    post:
      summary: Add quality metrics to a dataset
      security:
        - bearerAuth: []
      parameters:
        - name: datasetId
          in: path
          required: true
          schema:
            type: integer
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: array
              items:
                $ref: '#/components/schemas/MetricInput'
      responses:
        '200':
          description: Metrics added
          content:
            application/json:
              schema:
                type: object
                properties:
                  message:
                    type: string
                  timestamp:
                    type: string
        '404':
          description: Dataset not found
        '401':
          description: Unauthorized
  /datasets/{datasetId}/metrics/{metricName}:
    get:
      summary: Get historical data for a specific metric
      security:
        - bearerAuth: []
      parameters:
        - name: datasetId
          in: path
          required: true
          schema:
            type: integer
        - name: metricName
          in: path
          required: true
          schema:
            type: string
        - name: days
          in: query
          required: false
          schema:
            type: integer
            default: 30
      responses:
        '200':
          description: Metric history
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/MetricHistory'
        '404':
          description: Metric not found
        '401':
          description: Unauthorized
  /analyze:
    post:
      summary: Analyze data quality
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                csv:
                  type: string
                json:
                  type: array
                  items:
                    type: object
                dataset_id:
                  type: integer
      responses:
        '200':
          description: Analysis results
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AnalysisResult'
        '400':
          description: Invalid input
        '401':
          description: Unauthorized
  /health:
    get:
      summary: API health check
      responses:
        '200':
          description: Health status
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                  timestamp:
                    type: string
components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
  schemas:
    Dataset:
      type: object
      properties:
        id:
          type: integer
        name:
          type: string
        description:
          type: string
        owner:
          type: string
        created_date:
          type: string
    DatasetWithMetrics:
      allOf:
        - $ref: '#/components/schemas/Dataset'
        - type: object
          properties:
            recent_metrics:
              type: array
              items:
                $ref: '#/components/schemas/Metric'
    Metric:
      type: object
      properties:
        id:
          type: integer
        dataset_id:
          type: integer
        timestamp:
          type: string
        metric_name:
          type: string
        metric_value:
          type: number
        details:
          type: string
    MetricInput:
      type: object
      properties:
        name:
          type: string
        value:
          type: number
        details:
          type: object
      required:
        - name
        - value
    MetricHistory:
      type: object
      properties:
        metric_name:
          type: string
        dataset_id:
          type: integer
        days:
          type: integer
        history:
          type: array
          items:
            type: object
            properties:
              timestamp:
                type: string
              metric_value:
                type: number
              details:
                type: string
        stats:
          type: object
          properties:
            count:
              type: integer
            min:
              type: number
            max:
              type: number
            avg:
              type: number
            median:
              type: number
            latest:
              type: number
    AnalysisResult:
      type: object
      properties:
        metrics:
          type: object
          additionalProperties:
            type: object
            properties:
              value:
                type: number
              details:
                type: object
        row_count:
          type: integer
        column_count:
          type: integer
"""
```

## Integrating with Third-Party APIs

Examples of working with common third-party APIs:

```python
# Using popular APIs for data enrichment

# 1. Census API for demographic data
def get_demographics_by_zipcode(zipcode, api_key):
    """Get demographic data from the Census API"""
    import requests
    
    base_url = "https://api.census.gov/data/2020/acs/acs5"
    
    # Specify variables to retrieve
    variables = [
        "NAME",           # Area name
        "B01003_001E",    # Total population
        "B19013_001E",    # Median household income
        "B01002_001E",    # Median age
        "B25077_001E"     # Median home value
    ]
    
    # Build the request
    params = {
        "get": ",".join(variables),
        "for": f"zip code tabulation area:{zipcode}",
        "key": api_key
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract header and values
        headers = data[0]
        values = data[1]
        
        # Create a dictionary with the results
        result = dict(zip(headers, values))
        
        # Convert numeric values
        for key in ["B01003_001E", "B19013_001E", "B01002_001E", "B25077_001E"]:
            if key in result and result[key] not in ["", "null"]:
                result[key] = int(result[key])
        
        # Rename keys for clarity
        return {
            "area_name": result["NAME"],
            "total_population": result.get("B01003_001E"),
            "median_household_income": result.get("B19013_001E"),
            "median_age": result.get("B01002_001E"),
            "median_home_value": result.get("B25077_001E"),
            "zipcode": result["zip code tabulation area"]
        }
    else:
        print(f"Error: {response.status_code}")
        return None

# 2. Data.gov API for government datasets
def search_government_datasets(query, limit=10):
    """Search for datasets on Data.gov"""
    import requests
    
    base_url = "https://catalog.data.gov/api/3/action/package_search"
    
    params = {
        "q": query,
        "rows": limit
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        if data["success"]:
            results = data["result"]["results"]
            
            # Extract relevant information
            datasets = []
            for result in results:
                dataset = {
                    "title": result["title"],
                    "description": result.get("notes", ""),
                    "organization": result.get("organization", {}).get("title", ""),
                    "url": result.get("url", ""),
                    "resources": []
                }
                
                # Add resources (data files, APIs, etc.)
                for resource in result.get("resources", []):
                    dataset["resources"].append({
                        "name": resource.get("name", ""),
                        "format": resource.get("format", ""),
                        "url": resource.get("url", "")
                    })
                
                datasets.append(dataset)
            
            return {
                "count": data["result"]["count"],
                "datasets": datasets
            }
        else:
            print("API request unsuccessful")
            return None
    else:
        print(f"Error: {response.status_code}")
        return None

# 3. Weather API for historical weather data
def get_historical_weather(location, start_date, end_date, api_key):
    """Get historical weather data from the Visual Crossing Weather API"""
    import requests
    from datetime import datetime
    
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    
    # Format dates
    if isinstance(start_date, datetime):
        start_date = start_date.strftime("%Y-%m-%d")
    
    if isinstance(end_date, datetime):
        end_date = end_date.strftime("%Y-%m-%d")
    
    # Build the URL
    url = f"{base_url}/{location}/{start_date}/{end_date}"
    
    params = {
        "unitGroup": "metric",
        "include": "days",
        "key": api_key,
        "contentType": "json"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract daily data
        daily_data = []
        for day in data.get("days", []):
            daily_data.append({
                "date": day.get("datetime"),
                "temp_max": day.get("tempmax"),
                "temp_min": day.get("tempmin"),
                "temp_avg": day.get("temp"),
                "precipitation": day.get("precip"),
                "humidity": day.get("humidity"),
                "wind_speed": day.get("windspeed"),
                "description": day.get("description")
            })
        
        return {
            "location": data.get("resolvedAddress"),
            "timezone": data.get("timezone"),
            "daily": daily_data
        }
    else:
        print(f"Error: {response.status_code}")
        if response.content:
            print(response.content)
        return None

# 4. Exchange Rate API for currency conversion
def get_exchange_rates(base_currency="USD", api_key=None):
    """Get current exchange rates from Exchange Rate API"""
    import requests
    
    # Free tier endpoint (requires API key for paid version)
    if api_key:
        # Paid API
        base_url = "https://v6.exchangerate-api.com/v6"
        url = f"{base_url}/{api_key}/latest/{base_currency}"
    else:
        # Free API without key
        url = f"https://open.er-api.com/v6/latest/{base_currency}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        
        if api_key:
            # Paid API response format
            if data.get("result") == "success":
                return {
                    "base_currency": data.get("base_code"),
                    "timestamp": data.get("time_last_update_unix"),
                    "rates": data.get("conversion_rates", {})
                }
        else:
            # Free API response format
            if data.get("result") == "success":
                return {
                    "base_currency": data.get("base_code"),
                    "timestamp": data.get("time_last_update_unix"),
                    "rates": data.get("rates", {})
                }
        
        print("API request unsuccessful")
        return None
    else:
        print(f"Error: {response.status_code}")
        return None

# 5. Geocoding API for location data
def geocode_address(address, api_key):
    """Convert an address to geographic coordinates using Google Maps API"""
    import requests
    
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    
    params = {
        "address": address,
        "key": api_key
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        if data["status"] == "OK" and data["results"]:
            result = data["results"][0]
            
            # Extract coordinates
            location = result["geometry"]["location"]
            
            # Extract address components
            address_components = {}
            for component in result["address_components"]:
                for component_type in component["types"]:
                    address_components[component_type] = component["long_name"]
            
            return {
                "formatted_address": result["formatted_address"],
                "latitude": location["lat"],
                "longitude": location["lng"],
                "place_id": result["place_id"],
                "address_components": address_components
            }
        else:
            print(f"Geocoding error: {data['status']}")
            return None
    else:
        print(f"Error: {response.status_code}")
        return None

# Data enrichment with multiple APIs
def enrich_customer_data(customer_df, census_api_key, geocoding_api_key):
    """Enrich customer data with demographic and location information"""
    import pandas as pd
    
    # Create a copy of the input dataframe
    enriched_df = customer_df.copy()
    
    # Add columns for enriched data
    enriched_df["latitude"] = None
    enriched_df["longitude"] = None
    enriched_df["median_income"] = None
    enriched_df["population"] = None
    enriched_df["median_age"] = None
    
    # Process each customer
    for i, row in enriched_df.iterrows():
        # Only process if we have an address and zipcode
        if pd.notna(row.get("address")) and pd.notna(row.get("zipcode")):
            # Get coordinates
            address = f"{row['address']}, {row.get('city', '')}, {row.get('state', '')} {row['zipcode']}"
            geo_data = geocode_address(address, geocoding_api_key)
            
            if geo_data:
                enriched_df.at[i, "latitude"] = geo_data["latitude"]
                enriched_df.at[i, "longitude"] = geo_data["longitude"]
            
            # Get demographic data
            demo_data = get_demographics_by_zipcode(row["zipcode"], census_api_key)
            
            if demo_data:
                enriched_df.at[i, "median_income"] = demo_data["median_household_income"]
                enriched_df.at[i, "population"] = demo_data["total_population"]
                enriched_df.at[i, "median_age"] = demo_data["median_age"]
    
    return enriched_df
```

## Next Steps

After mastering these API integration and development techniques, you'll be ready to:

1. Create more sophisticated data pipeline APIs
2. Build real-time data quality monitoring systems
3. Integrate with cloud services APIs (AWS, Azure, GCP)
4. Develop custom data enrichment services
5. Create APIs that enforce data governance rules

## Resources

- [Python Requests Documentation](https://requests.readthedocs.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Flask-RESTx](https://flask-restx.readthedocs.io/)
- [OpenAPI Specification](https://swagger.io/specification/)
- [JSON Web Tokens](https://jwt.io/)
- [SocketIO Documentation](https://python-socketio.readthedocs.io/)
- [API Design Best Practices](https://docs.microsoft.com/en-us/azure/architecture/best-practices/api-design)

## Exercises and Projects

For additional practice, try these exercises:

1. Create a simple REST API for managing a data dictionary
2. Build a client to interact with a public API (e.g., data.gov)
3. Implement a data quality scoring API for CSV files
4. Create a WebSocket server for real-time data quality monitoring
5. Develop an API client library for a data service you use

## Contributing

If you've found this guide helpful, consider contributing:
- Add examples for other API frameworks (FastAPI, Django REST Framework)
- Share code for interacting with additional public APIs
- Suggest improvements or corrections

Happy API development and integration!
