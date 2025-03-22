# Concurrency and Parallelism Examples

## Example 1: Thread-based Concurrency

```python
"""
Thread-based concurrency for I/O-bound operations.
This is ideal for operations that involve waiting for external resources
like network requests, file I/O, or database queries.
"""
import threading
import time
import requests
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(message)s'
)

def fetch_url(url: str) -> Dict:
    """Fetch data from a URL and return response info."""
    logging.info(f"Fetching {url}")
    start_time = time.time()
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        return {
            'url': url,
            'status': response.status_code,
            'content_length': len(response.content),
            'time': time.time() - start_time
        }
    except requests.RequestException as e:
        logging.error(f"Error fetching {url}: {str(e)}")
        return {
            'url': url,
            'status': 'error',
            'error': str(e),
            'time': time.time() - start_time
        }

def fetch_urls_sequential(urls: List[str]) -> List[Dict]:
    """Fetch a list of URLs sequentially."""
    logging.info("Starting sequential fetching")
    start_time = time.time()
    
    results = []
    for url in urls:
        results.append(fetch_url(url))
    
    logging.info(f"Sequential fetching completed in {time.time() - start_time:.2f} seconds")
    return results

def fetch_urls_threaded(urls: List[str], max_threads: int = 10) -> List[Dict]:
    """Fetch a list of URLs using multiple threads."""
    logging.info(f"Starting threaded fetching with {max_threads} threads")
    start_time = time.time()
    
    # Create a list to store results
    results = [{} for _ in range(len(urls))]
    
    def worker(idx: int, url: str):
        """Worker function for each thread."""
        results[idx] = fetch_url(url)
    
    # Create and start threads
    threads = []
    for i, url in enumerate(urls):
        thread = threading.Thread(target=worker, args=(i, url))
        threads.append(thread)
        thread.start()
        
        # Optional: Limit maximum concurrent threads
        if len(threads) >= max_threads:
            threads[0].join()
            threads.pop(0)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    logging.info(f"Threaded fetching completed in {time.time() - start_time:.2f} seconds")
    return results

def demonstrate_threading():
    """Demonstrate threading vs sequential execution."""
    urls = [
        "https://www.example.com",
        "https://www.python.org",
        "https://www.google.com",
        "https://www.github.com",
        "https://www.stackoverflow.com",
        "https://www.wikipedia.org",
        "https://www.reddit.com",
        "https://www.twitter.com",
        "https://www.facebook.com",
        "https://www.amazon.com"
    ]
    
    # Sequential fetching
    sequential_results = fetch_urls_sequential(urls)
    sequential_time = sum(result.get('time', 0) for result in sequential_results)
    
    # Threaded fetching
    threaded_results = fetch_urls_threaded(urls)
    threaded_time = sum(result.get('time', 0) for result in threaded_results)
    
    # Compare results
    logging.info(f"Sequential total processing time: {sequential_time:.2f} seconds")
    logging.info(f"Threaded total processing time: {threaded_time:.2f} seconds")
    logging.info(f"Speedup factor: {sequential_time / threaded_time:.2f}x")

if __name__ == "__main__":
    demonstrate_threading()
```

## Example 2: Process-based Parallelism

```python
"""
Process-based parallelism for CPU-bound operations.
This is ideal for computationally intensive tasks that can benefit
from running on multiple CPU cores simultaneously.
"""
import multiprocessing as mp
import time
import numpy as np
from typing import List, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(message)s'
)

def calculate_prime_factors(number: int) -> List[int]:
    """Calculate prime factors of a number."""
    logging.info(f"Calculating prime factors of {number}")
    start_time = time.time()
    
    factors = []
    divisor = 2
    
    while number > 1:
        while number % divisor == 0:
            factors.append(divisor)
            number //= divisor
        divisor += 1
        
        # Optional: add a performance safeguard
        if divisor * divisor > number:
            if number > 1:
                factors.append(number)
            break
    
    logging.info(f"Factors of original number: {factors}")
    logging.info(f"Calculation completed in {time.time() - start_time:.4f} seconds")
    
    return factors

def process_numbers_sequential(numbers: List[int]) -> List[Tuple[int, List[int]]]:
    """Process a list of numbers sequentially."""
    logging.info("Starting sequential processing")
    start_time = time.time()
    
    results = []
    for number in numbers:
        factors = calculate_prime_factors(number)
        results.append((number, factors))
    
    logging.info(f"Sequential processing completed in {time.time() - start_time:.2f} seconds")
    return results

def process_numbers_parallel(numbers: List[int], max_processes: int = None) -> List[Tuple[int, List[int]]]:
    """Process a list of numbers in parallel using multiple processes."""
    max_processes = max_processes or mp.cpu_count()
    logging.info(f"Starting parallel processing with {max_processes} processes")
    start_time = time.time()
    
    # Create a pool of worker processes
    with mp.Pool(processes=max_processes) as pool:
        # Map the function to the list of numbers
        results = pool.map(calculate_prime_factors, numbers)
    
    # Combine numbers with their factors
    result_with_numbers = list(zip(numbers, results))
    
    logging.info(f"Parallel processing completed in {time.time() - start_time:.2f} seconds")
    return result_with_numbers

def demonstrate_multiprocessing():
    """Demonstrate multiprocessing vs sequential execution."""
    # Generate large numbers likely to have complex factorizations
    np.random.seed(42)
    large_numbers = [
        np.random.randint(100000000, 900000000) for _ in range(8)
    ]
    
    # Sequential processing
    sequential_start = time.time()
    sequential_results = process_numbers_sequential(large_numbers)
    sequential_time = time.time() - sequential_start
    
    # Parallel processing
    parallel_start = time.time()
    parallel_results = process_numbers_parallel(large_numbers)
    parallel_time = time.time() - parallel_start
    
    # Compare results
    logging.info(f"Sequential execution time: {sequential_time:.2f} seconds")
    logging.info(f"Parallel execution time: {parallel_time:.2f} seconds")
    logging.info(f"Speedup factor: {sequential_time / parallel_time:.2f}x")
    
    # Verify results match
    seq_dict = dict(sequential_results)
    par_dict = dict(parallel_results)
    
    if seq_dict == par_dict:
        logging.info("Results match! 👍")
    else:
        logging.warning("Results don't match! ⚠️")

if __name__ == "__main__":
    demonstrate_multiprocessing()
```

## Example 3: Asynchronous Programming with asyncio

```python
"""
Asynchronous programming for concurrent I/O operations.
This is ideal for operations that involve waiting, like network requests,
allowing for efficient use of resources without threads.
"""
import asyncio
import aiohttp
import time
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s'
)
logger = logging.getLogger('asyncio_example')

async def fetch_url_async(session, url: str) -> Dict:
    """Fetch a URL asynchronously using aiohttp."""
    logger.info(f"Fetching {url}")
    start_time = time.time()
    
    try:
        async with session.get(url, timeout=10) as response:
            content = await response.read()
            return {
                'url': url,
                'status': response.status,
                'content_length': len(content),
                'time': time.time() - start_time
            }
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return {
            'url': url,
            'status': 'error',
            'error': str(e),
            'time': time.time() - start_time
        }

async def fetch_all_urls(urls: List[str], max_concurrent: int = 10) -> List[Dict]:
    """Fetch multiple URLs asynchronously with concurrency limit."""
    logger.info(f"Starting async fetching with max {max_concurrent} concurrent requests")
    start_time = time.time()
    
    # Create a client session
    async with aiohttp.ClientSession() as session:
        # Use a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(url):
            async with semaphore:
                return await fetch_url_async(session, url)
        
        # Create tasks for all URLs
        tasks = [fetch_with_semaphore(url) for url in urls]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
    logger.info(f"Async fetching completed in {time.time() - start_time:.2f} seconds")
    return results

def fetch_urls_sequential(urls: List[str]) -> List[Dict]:
    """Fetch a list of URLs sequentially for comparison."""
    logger.info("Starting sequential fetching")
    start_time = time.time()
    
    import requests
    results = []
    
    for url in urls:
        logger.info(f"Fetching {url}")
        url_start_time = time.time()
        
        try:
            response = requests.get(url, timeout=10)
            results.append({
                'url': url,
                'status': response.status_code,
                'content_length': len(response.content),
                'time': time.time() - url_start_time
            })
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            results.append({
                'url': url,
                'status': 'error',
                'error': str(e),
                'time': time.time() - url_start_time
            })
    
    logger.info(f"Sequential fetching completed in {time.time() - start_time:.2f} seconds")
    return results

async def demonstrate_asyncio():
    """Demonstrate asyncio vs sequential execution."""
    urls = [
        "https://www.example.com",
        "https://www.python.org",
        "https://www.google.com",
        "https://www.github.com",
        "https://www.stackoverflow.com",
        "https://www.wikipedia.org",
        "https://www.reddit.com",
        "https://www.twitter.com",
        "https://www.facebook.com",
        "https://www.amazon.com"
    ]
    
    # Async fetching
    async_start = time.time()
    async_results = await fetch_all_urls(urls)
    async_time = time.time() - async_start
    
    # Sequential fetching
    seq_start = time.time()
    sequential_results = fetch_urls_sequential(urls)
    seq_time = time.time() - seq_start
    
    # Compare results
    logger.info(f"Sequential execution time: {seq_time:.2f} seconds")
    logger.info(f"Async execution time: {async_time:.2f} seconds")
    logger.info(f"Speedup factor: {seq_time / async_time:.2f}x")

if __name__ == "__main__":
    asyncio.run(demonstrate_asyncio())
```

## Example 4: Concurrent.futures for Simplified Concurrency

```python
"""
Concurrent.futures provides a high-level interface for asynchronously
executing callables using threads or processes. This is often simpler
than directly working with threads or processes.
"""
import concurrent.futures
import time
import requests
import numpy as np
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s'
)
logger = logging.getLogger('concurrent_futures_example')

# I/O-bound task for ThreadPoolExecutor
def fetch_url(url: str) -> Dict[str, Any]:
    """Fetch a URL and return response info."""
    logger.info(f"Fetching {url}")
    start_time = time.time()
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        return {
            'url': url,
            'status': response.status_code,
            'content_length': len(response.content),
            'time': time.time() - start_time
        }
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return {
            'url': url,
            'status': 'error',
            'error': str(e),
            'time': time.time() - start_time
        }

# CPU-bound task for ProcessPoolExecutor
def complex_calculation(params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform a CPU-intensive calculation."""
    n = params.get('n', 1000000)
    logger.info(f"Starting calculation with n={n}")
    start_time = time.time()
    
    # Simulate complex calculation (e.g., Monte Carlo estimation of π)
    inside_circle = 0
    np.random.seed(params.get('seed', 42))
    
    for _ in range(n):
        x, y = np.random.random(2)
        if x**2 + y**2 <= 1:
            inside_circle += 1
    
    pi_estimate = 4 * inside_circle / n
    
    result = {
        'task_id': params.get('task_id', 'unknown'),
        'n': n,
        'pi_estimate': pi_estimate,
        'time': time.time() - start_time
    }
    
    logger.info(f"Calculation completed in {result['time']:.2f} seconds")
    return result

def demonstrate_thread_pool():
    """Demonstrate ThreadPoolExecutor for I/O-bound tasks."""
    urls = [
        "https://www.example.com",
        "https://www.python.org",
        "https://www.google.com",
        "https://www.github.com",
        "https://www.stackoverflow.com",
        "https://www.wikipedia.org",
        "https://www.reddit.com",
        "https://www.twitter.com",
        "https://www.facebook.com",
        "https://www.amazon.com"
    ]
    
    logger.info("Starting ThreadPoolExecutor demo")
    start_time = time.time()
    
    # Use ThreadPoolExecutor for I/O-bound operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks and get futures
        future_to_url = {executor.submit(fetch_url, url): url for url in urls}
        
        # Process results as they complete
        results = []
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                results.append(data)
                logger.info(f"Completed: {url} - Status: {data.get('status')}")
            except Exception as exc:
                logger.error(f"{url} generated an exception: {exc}")
    
    logger.info(f"All URLs fetched in {time.time() - start_time:.2f} seconds")
    return results

def demonstrate_process_pool():
    """Demonstrate ProcessPoolExecutor for CPU-bound tasks."""
    # Create a list of calculation parameters
    tasks = [
        {'task_id': f'task_{i}', 'n': 5000000, 'seed': i} 
        for i in range(8)
    ]
    
    logger.info("Starting ProcessPoolExecutor demo")
    start_time = time.time()
    
    # Use ProcessPoolExecutor for CPU-bound operations
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map function to tasks
        results = list(executor.map(complex_calculation, tasks))
    
    logger.info(f"All calculations completed in {time.time() - start_time:.2f} seconds")
    
    # Show average π estimate
    pi_estimates = [result['pi_estimate'] for result in results]
    logger.info(f"Average π estimate: {np.mean(pi_estimates)}")
    logger.info(f"True π value:       {np.pi}")
    
    return results

def run_demos():
    """Run both ThreadPoolExecutor and ProcessPoolExecutor demos."""
    # Run I/O-bound demo with thread pool
    logger.info("\n=== Thread Pool for I/O-bound Tasks ===\n")
    thread_results = demonstrate_thread_pool()
    
    # Run CPU-bound demo with process pool
    logger.info("\n=== Process Pool for CPU-bound Tasks ===\n")
    process_results = demonstrate_process_pool()
    
    return thread_results, process_results

if __name__ == "__main__":
    run_demos()
```

## Example 5: Combining Techniques with JobLib

```python
"""
Joblib provides a simple API for parallel computing that wraps multiprocessing
and threadpoolexecutor with additional optimizations for NumPy arrays and caching.
"""
import numpy as np
import time
import logging
from typing import List, Dict, Any
from joblib import Parallel, delayed, Memory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s'
)
logger = logging.getLogger('joblib_example')

# Set up cache directory for Memory
cachedir = './joblib_cache'
memory = Memory(cachedir, verbose=0)

# CPU-bound function that we'll parallelize
def process_chunk(chunk: np.ndarray, operation: str) -> Dict[str, Any]:
    """Process a chunk of data with specified operation."""
    chunk_id = hash(chunk.tobytes()) % 10000
    logger.info(f"Processing chunk {chunk_id} with {operation}")
    start_time = time.time()
    
    result = None
    if operation == 'mean':
        result = np.mean(chunk, axis=0)
    elif operation == 'std':
        result = np.std(chunk, axis=0)
    elif operation == 'pca':
        # Simple PCA implementation
        centered = chunk - np.mean(chunk, axis=0)
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Return top 2 components
        result = {
            'eigenvalues': eigenvalues[:2],
            'eigenvectors': eigenvectors[:, :2]
        }
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    processing_time = time.time() - start_time
    return {
        'chunk_id': chunk_id,
        'operation': operation,
        'result': result,
        'time': processing_time
    }

# Version that can be cached by joblib.Memory
@memory.cache
def cached_process_chunk(chunk_id: int, chunk: np.ndarray, operation: str) -> Dict[str, Any]:
    """Cacheable version of process_chunk."""
    logger.info(f"Processing chunk {chunk_id} with {operation} (cacheable)")
    start_time = time.time()
    
    result = None
    if operation == 'mean':
        result = np.mean(chunk, axis=0)
    elif operation == 'std':
        result = np.std(chunk, axis=0)
    elif operation == 'pca':
        centered = chunk - np.mean(chunk, axis=0)
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        result = {
            'eigenvalues': eigenvalues[:2],
            'eigenvectors': eigenvectors[:, :2]
        }
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    processing_time = time.time() - start_time
    return {
        'chunk_id': chunk_id,
        'operation': operation,
        'result': result,
        'time': processing_time
    }

def parallel_process_data(data: np.ndarray, operation: str, n_jobs: int = -1, 
                         chunk_size: int = 1000, use_cache: bool = False) -> List[Dict[str, Any]]:
    """Process data in parallel using joblib."""
    n_samples = data.shape[0]
    
    # Split data into chunks
    chunks = []
def parallel_process_data(data: np.ndarray, operation: str, n_jobs: int = -1, 
                         chunk_size: int = 1000, use_cache: bool = False) -> List[Dict[str, Any]]:
    """Process data in parallel using joblib."""
    n_samples = data.shape[0]
    
    # Split data into chunks
    chunks = []
    chunk_ids = []
    for i in range(0, n_samples, chunk_size):
        end = min(i + chunk_size, n_samples)
        chunks.append(data[i:end])
        chunk_ids.append(i // chunk_size)
    
    logger.info(f"Split data into {len(chunks)} chunks of size {chunk_size}")
    start_time = time.time()
    
    # Process chunks in parallel
    if use_cache:
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(cached_process_chunk)(chunk_id, chunk, operation) 
            for chunk_id, chunk in zip(chunk_ids, chunks)
        )
    else:
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_chunk)(chunk, operation) 
            for chunk in chunks
        )
    
    logger.info(f"Processed {len(chunks)} chunks in {time.time() - start_time:.2f} seconds")
    return results

def demonstrate_joblib():
    """Demonstrate parallel processing with joblib."""
    # Generate random data
    np.random.seed(42)
    n_samples, n_features = 50000, 20
    data = np.random.randn(n_samples, n_features)
    
    # Sequential processing for comparison
    logger.info("Starting sequential processing")
    sequential_start = time.time()
    sequential_results = [process_chunk(chunk, 'pca') 
                          for chunk in np.array_split(data, 8)]
    sequential_time = time.time() - sequential_start
    
    # Parallel processing without caching
    logger.info("Starting parallel processing (no cache)")
    parallel_start = time.time()
    parallel_results = parallel_process_data(
        data, 'pca', n_jobs=4, chunk_size=n_samples//8
    )
    parallel_time = time.time() - parallel_start
    
    # Parallel processing with caching (first run)
    logger.info("Starting parallel processing with cache (first run)")
    cache_start1 = time.time()
    cache_results1 = parallel_process_data(
        data, 'pca', n_jobs=4, chunk_size=n_samples//8, use_cache=True
    )
    cache_time1 = time.time() - cache_start1
    
    # Parallel processing with caching (second run - should use cache)
    logger.info("Starting parallel processing with cache (second run)")
    cache_start2 = time.time()
    cache_results2 = parallel_process_data(
        data, 'pca', n_jobs=4, chunk_size=n_samples//8, use_cache=True
    )
    cache_time2 = time.time() - cache_start2
    
    # Compare results
    logger.info(f"Sequential execution time: {sequential_time:.2f} seconds")
    logger.info(f"Parallel execution time: {parallel_time:.2f} seconds")
    logger.info(f"Parallel cached (first run): {cache_time1:.2f} seconds")
    logger.info(f"Parallel cached (second run): {cache_time2:.2f} seconds")
    
    logger.info(f"Speedup (parallel vs sequential): {sequential_time / parallel_time:.2f}x")
    logger.info(f"Speedup (cached second run vs first run): {cache_time1 / cache_time2:.2f}x")
    
    return {
        'sequential': sequential_results,
        'parallel': parallel_results,
        'cached_first': cache_results1,
        'cached_second': cache_results2,
        'times': {
            'sequential': sequential_time,
            'parallel': parallel_time,
            'cached_first': cache_time1,
            'cached_second': cache_time2
        }
    }

if __name__ == "__main__":
    demonstrate_joblib()
```

## Example 6: Combining asyncio with multiprocessing

```python
"""
This example demonstrates how to combine asyncio (for I/O-bound tasks)
with multiprocessing (for CPU-bound tasks) to maximize performance
by utilizing both I/O concurrency and CPU parallelism.
"""
import asyncio
import aiohttp
import multiprocessing as mp
import time
import numpy as np
from typing import List, Dict, Any
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hybrid_async_mp')

# I/O-bound task: Fetch data from URLs asynchronously
async def fetch_url_async(session, url: str) -> Dict[str, Any]:
    """Fetch a URL asynchronously using aiohttp."""
    logger.info(f"Fetching {url}")
    start_time = time.time()
    
    try:
        async with session.get(url, timeout=10) as response:
            content = await response.read()
            return {
                'url': url,
                'status': response.status,
                'content_length': len(content),
                'content': content.decode('utf-8', errors='ignore')[:100],  # First 100 chars
                'time': time.time() - start_time
            }
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return {
            'url': url,
            'status': 'error',
            'error': str(e),
            'time': time.time() - start_time
        }

# CPU-bound task: Process the fetched data in a separate process
def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the data fetched from a URL.
    This is a CPU-bound task that will be run in a separate process.
    """
    url = data.get('url', 'unknown')
    logger.info(f"Processing data from {url}")
    start_time = time.time()
    
    result = {
        'url': url,
        'processing_time': 0,
        'word_count': 0,
        'stats': {}
    }
    
    try:
        content = data.get('content', '')
        
        # Simple text analysis
        words = content.lower().split()
        word_count = len(words)
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            # Clean the word
            word = word.strip('.,!?():;"\'')
            if word and len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Calculate some statistics
        stats = {
            'word_count': word_count,
            'unique_words': len(word_freq),
            'top_words': dict(top_words),
            'average_word_length': np.mean([len(w) for w in words]) if words else 0
        }
        
        result['word_count'] = word_count
        result['stats'] = stats
        result['processing_time'] = time.time() - start_time
    
    except Exception as e:
        logger.error(f"Error processing data from {url}: {str(e)}")
        result['error'] = str(e)
    
    return result

# Function to be run in each worker process
def worker_process(data_queue, result_queue):
    """Worker process function to process data from the queue."""
    logger.info(f"Worker process started (pid={mp.current_process().pid})")
    
    while True:
        # Get data from the queue
        data = data_queue.get()
        if data is None:  # Poison pill to stop the worker
            logger.info(f"Worker process shutting down (pid={mp.current_process().pid})")
            break
        
        # Process the data
        result = process_data(data)
        
        # Put the result in the result queue
        result_queue.put(result)

async def main(urls: List[str], num_workers: int = 4):
    """
    Main function to fetch data asynchronously and process it in parallel.
    
    Args:
        urls: List of URLs to fetch
        num_workers: Number of worker processes for CPU-bound tasks
    """
    logger.info(f"Starting with {len(urls)} URLs and {num_workers} workers")
    overall_start = time.time()
    
    # Create queues for inter-process communication
    data_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Start worker processes
    workers = []
    for _ in range(num_workers):
        p = mp.Process(target=worker_process, args=(data_queue, result_queue))
        p.start()
        workers.append(p)
        logger.info(f"Started worker process with pid {p.pid}")
    
    # Fetch data asynchronously
    fetch_start = time.time()
    async with aiohttp.ClientSession() as session:
        # Create tasks for all URLs
        fetch_tasks = [fetch_url_async(session, url) for url in urls]
        
        # Process URLs as they complete
        for task in asyncio.as_completed(fetch_tasks):
            result = await task
            if result.get('status') != 'error':
                # Put the result in the queue for processing
                data_queue.put(result)
                logger.info(f"Queued {result['url']} for processing")
    
    fetch_time = time.time() - fetch_start
    logger.info(f"All URLs fetched in {fetch_time:.2f} seconds")
    
    # Send poison pills to stop workers
    for _ in range(num_workers):
        data_queue.put(None)
    
    # Wait for all workers to finish
    for p in workers:
        p.join()
    
    # Collect all results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    overall_time = time.time() - overall_start
    logger.info(f"All processing completed in {overall_time:.2f} seconds")
    
    # Print summary
    logger.info(f"Processed {len(results)} URLs")
    
    # Calculate some statistics
    total_words = sum(r.get('word_count', 0) for r in results)
    logger.info(f"Total words processed: {total_words}")
    
    for result in results:
        url = result.get('url', 'unknown')
        word_count = result.get('word_count', 0)
        process_time = result.get('processing_time', 0)
        logger.info(f"URL: {url} - Words: {word_count} - Processing time: {process_time:.2f}s")
    
    return {
        'fetch_time': fetch_time,
        'overall_time': overall_time,
        'results': results
    }

if __name__ == "__main__":
    # List of URLs to fetch and process
    urls = [
        "https://www.example.com",
        "https://www.python.org",
        "https://www.wikipedia.org",
        "https://docs.python.org/3/library/asyncio.html",
        "https://docs.python.org/3/library/multiprocessing.html",
        "https://www.google.com",
        "https://www.github.com",
        "https://www.stackoverflow.com"
    ]
    
    # Run the hybrid async+multiprocessing example
    asyncio.run(main(urls))
```
# Machine Learning with scikit-learn Examples

## Example 1: Basic Classification Pipeline

```python
"""
Basic classification pipeline with scikit-learn.
This example demonstrates:
1. Data loading and exploration
2. Data preprocessing
3. Model training and evaluation
4. Model persistence
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import joblib
import logging
from typing import Dict, Tuple, Any, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sklearn_example')

def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a file (CSV, Excel, etc.)."""
    logger.info(f"Loading data from {filepath}")
    
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def explore_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform basic exploratory data analysis."""
    logger.info("Exploring data")
    
    # Get basic info
    result = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percent': (df.isnull().sum() / len(df) * 100).to_dict(),
        'summary': df.describe().to_dict()
    }
    
    # Check for categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    result['categorical_columns'] = categorical_cols
    
    # Check for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    result['numeric_columns'] = numeric_cols
    
    # Basic checks
    logger.info(f"Shape: {result['shape']}")
    logger.info(f"Categorical columns: {categorical_cols}")
    logger.info(f"Numeric columns: {numeric_cols}")
    
    # Check for missing values
    missing_cols = [col for col, count in result['missing_values'].items() if count > 0]
    if missing_cols:
        logger.warning(f"Columns with missing values: {missing_cols}")
    
    return result

def preprocess_data(
    df: pd.DataFrame, 
    target_column: str,
    categorical_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Preprocess the data for machine learning.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        categorical_columns: List of categorical column names
        numeric_columns: List of numeric column names
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing preprocessed data and metadata
    """
    logger.info("Preprocessing data")
    
    # Identify feature and target columns
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    # If columns not provided, infer them
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target from feature columns
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    
    # Extract features and target
    X = df[categorical_columns + numeric_columns]
    y = df[target_column]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(f"Target distribution in train set: {y_train.value_counts(normalize=True)}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'categorical_columns': categorical_columns,
        'numeric_columns': numeric_columns,
        'target_column': target_column
    }

def build_pipeline(
    categorical_columns: List[str],
    numeric_columns: List[str],
    random_state: int = 42
) -> Pipeline:
    """
    Build a scikit-learn pipeline for classification.
    
    Args:
        categorical_columns: List of categorical column names
        numeric_columns: List of numeric column names
        random_state: Random seed for reproducibility
        
    Returns:
        Scikit-learn pipeline
    """
    logger.info("Building pipeline")
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        # You could add OrdinalEncoder or OneHotEncoder here
    ])
    
    # Preprocessing for numerical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_columns),
            ('num', numeric_transformer, numeric_columns)
        ]
    )
    
    # Build pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_classif, k=min(10, len(categorical_columns) + len(numeric_columns)))),
        ('classifier', RandomForestClassifier(random_state=random_state))
    ])
    
    return pipeline

def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train a model using the provided pipeline.
    
    Args:
        pipeline: Scikit-learn pipeline
        X_train: Training features
        y_train: Training targets
        cv: Number of cross-validation folds
        
    Returns:
        Trained pipeline and training metrics
    """
    logger.info("Training model")
    start_time = pd.Timestamp.now()
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Cross-validation scores
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    
    training_time = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    metrics = {
        'cross_val_scores': cv_scores,
        'mean_cv_accuracy': cv_scores.mean(),
        'std_cv_accuracy': cv_scores.std(),
        'training_time': training_time
    }
    
    return pipeline, metrics

def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluate the trained model on test data.
    
    Args:
        pipeline: Trained scikit-learn pipeline
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model")
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # For ROC and precision-recall curves
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)
    else:
        # Some models don't have predict_proba
        y_prob = None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    # ROC curve for binary classification
    if y_prob is not None and len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        metrics['roc_auc'] = auc(fpr, tpr)
        metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    
    return metrics

def save_model(pipeline: Pipeline, filepath: str) -> None:
    """
    Save the trained model to a file.
    
    Args:
        pipeline: Trained scikit-learn pipeline
        filepath: Path to save the model
    """
    logger.info(f"Saving model to {filepath}")
    joblib.dump(pipeline, filepath)

def load_saved_model(filepath: str) -> Pipeline:
    """
    Load a saved model from a file.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded scikit-learn pipeline
    """
    logger.info(f"Loading model from {filepath}")
    return joblib.load(filepath)

def run_classification_example(data_file: str, target_column: str) -> Dict[str, Any]:
    """
    Run the full classification example.
    
    Args:
        data_file: Path to the data file
        target_column: Name of the target column
        
    Returns:
        Dictionary with all results
    """
    # Load and explore data
    df = load_data(data_file)
    exploration = explore_data(df)
    
    # Preprocess data
    data = preprocess_data(df, target_column)
    
    # Build and train model
    pipeline = build_pipeline(
        data['categorical_columns'],
        data['numeric_columns']
    )
    
    trained_pipeline, training_metrics = train_model(
        pipeline, data['X_train'], data['y_train']
    )
    
    # Evaluate model
    evaluation_metrics = evaluate_model(
        trained_pipeline, data['X_test'], data['y_test']
    )
    
    # Save model
    model_file = f"{target_column}_classifier.joblib"
    save_model(trained_pipeline, model_file)
    
    # Return all results
    return {
        'exploration': exploration,
        'data': data,
        'training_metrics': training_metrics,
        'evaluation_metrics': evaluation_metrics,
        'model_file': model_file
    }

# If run directly, use this example
if __name__ == "__main__":
    # For this example, we'll use the Iris dataset
    from sklearn.datasets import load_iris
    
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(
        data=np.c_[iris['data'], iris['target']],
        columns=iris['feature_names'] + ['target']
    )
    
    # Convert target to integers for simplicity
    df['target'] = df['target'].astype(int)
    
    # Save to a temporary CSV file
    temp_file = "iris_data.csv"
    df.to_csv(temp_file, index=False)
    
    # Run the example
    results = run_classification_example(temp_file, "target")
    
    print("\nClassification Report:")
    print(pd.DataFrame(results['evaluation_metrics']['classification_report']).T)
```

## Example 2: Clustering and Dimensionality Reduction

```python
"""
Unsupervised learning with clustering and dimensionality reduction.
This example demonstrates:
1. Data preprocessing for unsupervised learning
2. Dimensionality reduction with PCA
3. K-means clustering
4. Visualization of clusters
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline
import logging
from typing import Dict, Tuple, Any, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('unsupervised_learning')

def preprocess_for_clustering(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    scale: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Preprocess data for clustering.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric column names to use
        scale: Whether to standardize the data
        
    Returns:
        Preprocessed data array and preprocessing info
    """
    logger.info("Preprocessing data for clustering")
    
    # Select numeric columns if not specified
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    logger.info(f"Using {len(numeric_columns)} numeric features")
    
    # Extract numeric features
    X = df[numeric_columns].values
    
    # Scale the data
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("Data standardized")
    else:
        X_scaled = X
        scaler = None
    
    preprocessing_info = {
        'numeric_columns': numeric_columns,
        'scaler': scaler,
        'original_shape': X.shape
    }
    
    return X_scaled, preprocessing_info

def reduce_dimensions(
    X: np.ndarray,
    n_components: int = 2,
    method: str = 'pca'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reduce dimensionality of the data.
    
    Args:
        X: Input data array
        n_components: Number of components to reduce to
        method: Dimensionality reduction method ('pca' supported)
        
    Returns:
        Reduced data array and reduction info
    """
    logger.info(f"Reducing dimensions to {n_components} components using {method}")
    
    reduction_info = {}
    
    if method == 'pca':
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        logger.info(f"Explained variance by {n_components} components: {cumulative_variance[-1]:.4f}")
        
        reduction_info = {
            'model': pca,
            'explained_variance_ratio': explained_variance,
            'cumulative_variance': cumulative_variance,
            'n_components': n_components
        }
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")
    
    return X_reduced, reduction_info

def perform_clustering(
    X: np.ndarray,
    method: str = 'kmeans',
    n_clusters: int = 3,
    random_state: int = 42,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Perform clustering on the data.
    
    Args:
        X: Input data array
        method: Clustering method ('kmeans' or 'dbscan')
        n_clusters: Number of clusters for k-means
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments for clustering algorithms
        
    Returns:
        Cluster labels and clustering info
    """
    logger.info(f"Performing clustering using {method}")
    
    clustering_info = {}
    
    if method == 'kmeans':
        # Try different numbers of clusters to find optimal
        if 'find_optimal' in kwargs and kwargs['find_optimal']:
            max_clusters = kwargs.get('max_clusters', 10)
            silhouette_scores = []
            ch_scores = []
            
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=random_state)
                labels = kmeans.fit_predict(X)
                
                if len(np.unique(labels)) > 1:  # Silhouette requires at least 2 clusters
                    silhouette_scores.append(silhouette_score(X, labels))
                    ch_scores.append(calinski_harabasz_score
silhouette_scores.append(silhouette_score(X, labels))
                ch_scores.append(calinski_harabasz_score(X, labels))
            
            # Find optimal k based on silhouette score
            optimal_k_silhouette = np.argmax(silhouette_scores) + 2
            optimal_k_ch = np.argmax(ch_scores) + 2
            
            logger.info(f"Optimal number of clusters (silhouette): {optimal_k_silhouette}")
            logger.info(f"Optimal number of clusters (Calinski-Harabasz): {optimal_k_ch}")
            
            clustering_info['silhouette_scores'] = silhouette_scores
            clustering_info['ch_scores'] = ch_scores
            clustering_info['optimal_k_silhouette'] = optimal_k_silhouette
            clustering_info['optimal_k_ch'] = optimal_k_ch
            
            # Use optimal k from silhouette score
            n_clusters = optimal_k_silhouette
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(X)
        
        # Calculate clustering metrics
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            ch_score = calinski_harabasz_score(X, labels)
            logger.info(f"Silhouette score: {silhouette:.4f}")
            logger.info(f"Calinski-Harabasz score: {ch_score:.4f}")
        else:
            silhouette = 0
            ch_score = 0
            logger.warning("Only one cluster found, scores set to 0")
        
        clustering_info.update({
            'model': kmeans,
            'n_clusters': n_clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': ch_score,
            'inertia': kmeans.inertia_
        })
        
    elif method == 'dbscan':
        # Set default parameters
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Count number of clusters (excluding noise points with label -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        logger.info(f"Number of clusters found: {n_clusters}")
        logger.info(f"Number of noise points: {list(labels).count(-1)}")
        
        # Calculate clustering metrics if more than one cluster
        if n_clusters > 1:
            # Filter out noise points for score calculation
            if -1 in labels:
                mask = labels != -1
                silhouette = silhouette_score(X[mask], labels[mask])
                ch_score = calinski_harabasz_score(X[mask], labels[mask])
            else:
                silhouette = silhouette_score(X, labels)
                ch_score = calinski_harabasz_score(X, labels)
                
            logger.info(f"Silhouette score: {silhouette:.4f}")
            logger.info(f"Calinski-Harabasz score: {ch_score:.4f}")
        else:
            silhouette = 0
            ch_score = 0
            logger.warning("Less than two non-noise clusters found, scores set to 0")
        
        clustering_info = {
            'model': dbscan,
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': ch_score
        }
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    return labels, clustering_info

def visualize_clusters(
    X_reduced: np.ndarray,
    labels: np.ndarray,
    reduction_info: Dict[str, Any],
    clustering_info: Dict[str, Any],
    title: str = "Cluster Visualization",
    show_centers: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize clusters in 2D after dimensionality reduction.
    
    Args:
        X_reduced: Reduced data (2D)
        labels: Cluster labels
        reduction_info: Dimensionality reduction info
        clustering_info: Clustering info
        title: Plot title
        show_centers: Whether to show cluster centers
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info("Visualizing clusters")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique cluster labels
    unique_labels = np.unique(labels)
    
    # Generate colors for clusters
    cmap = plt.cm.get_cmap('tab10', len(unique_labels))
    
    # Plot data points colored by cluster
    for i, cluster in enumerate(unique_labels):
        mask = labels == cluster
        if cluster == -1:
            # Plot noise points as black X's
            ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], marker='x', s=50, 
                       color='black', label='Noise')
        else:
            # Plot cluster points
            ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], s=50, 
                       color=cmap(i), alpha=0.7, 
                       label=f'Cluster {cluster}')
    
    # Plot cluster centers for K-means
    if show_centers and 'cluster_centers' in clustering_info:
        centers = clustering_info['cluster_centers']
        pca = reduction_info['model']
        
        # Transform centers to 2D
        centers_2d = pca.transform(centers)
        
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], s=200, marker='*', 
                   color='black', alpha=0.8, label='Centroids')
    
    # Set plot labels and title
    if reduction_info.get('model').__class__.__name__ == 'PCA':
        explained_var = reduction_info['explained_variance_ratio']
        ax.set_xlabel(f'Principal Component 1 ({explained_var[0]:.2%} variance)')
        ax.set_ylabel(f'Principal Component 2 ({explained_var[1]:.2%} variance)')
    else:
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
    
    # Add clustering method info to title
    if 'model' in clustering_info:
        model_name = clustering_info['model'].__class__.__name__
        title = f"{title} - {model_name}"
    
    ax.set_title(title)
    ax.legend()
    
    # Add text with metrics
    if 'silhouette_score' in clustering_info:
        silhouette = clustering_info['silhouette_score']
        n_clusters = clustering_info.get('n_clusters', 'N/A')
        
        metric_text = (
            f"Clusters: {n_clusters}\n"
            f"Silhouette Score: {silhouette:.4f}"
        )
        
        if 'inertia' in clustering_info:
            inertia = clustering_info['inertia']
            metric_text += f"\nInertia: {inertia:.2f}"
        
        plt.figtext(0.02, 0.02, metric_text, fontsize=10, bbox=dict(
            facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.tight_layout()
    return fig

def run_clustering_example(data_file: str) -> Dict[str, Any]:
    """
    Run the full clustering example.
    
    Args:
        data_file: Path to the data file
        
    Returns:
        Dictionary with all results
    """
    # Load data
    df = load_data(data_file)
    
    # Preprocess data
    X_scaled, preprocessing_info = preprocess_for_clustering(df)
    
    # Reduce dimensions for visualization
    X_reduced, reduction_info = reduce_dimensions(X_scaled, n_components=2)
    
    # Find optimal number of clusters
    _, kmeans_info = perform_clustering(
        X_scaled, method='kmeans', find_optimal=True, max_clusters=10
    )
    
    # Perform K-means clustering with optimal k
    optimal_k = kmeans_info['optimal_k_silhouette']
    kmeans_labels, kmeans_info = perform_clustering(
        X_scaled, method='kmeans', n_clusters=optimal_k
    )
    
    # Perform DBSCAN clustering
    dbscan_labels, dbscan_info = perform_clustering(
        X_scaled, method='dbscan', eps=0.5, min_samples=5
    )
    
    # Visualize K-means clusters
    kmeans_fig = visualize_clusters(
        X_reduced, kmeans_labels, reduction_info, kmeans_info,
        title="K-means Clustering", save_path="kmeans_clusters.png"
    )
    
    # Visualize DBSCAN clusters
    dbscan_fig = visualize_clusters(
        X_reduced, dbscan_labels, reduction_info, dbscan_info,
        title="DBSCAN Clustering", show_centers=False, 
        save_path="dbscan_clusters.png"
    )
    
    # Return all results
    return {
        'preprocessing_info': preprocessing_info,
        'reduction_info': reduction_info,
        'kmeans_info': kmeans_info,
        'dbscan_info': dbscan_info,
        'kmeans_labels': kmeans_labels,
        'dbscan_labels': dbscan_labels,
        'X_scaled': X_scaled,
        'X_reduced': X_reduced,
        'kmeans_fig': kmeans_fig,
        'dbscan_fig': dbscan_fig
    }

# Helper function to load data
def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a file (CSV, Excel, etc.)."""
    logger.info(f"Loading data from {filepath}")
    
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

# If run directly, use this example
if __name__ == "__main__":
    # For this example, we'll use the Iris dataset
    from sklearn.datasets import load_iris
    
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(
        data=iris['data'],
        columns=iris['feature_names']
    )
    
    # Save to a temporary CSV file
    temp_file = "iris_data.csv"
    df.to_csv(temp_file, index=False)
    
    # Run the example
    results = run_clustering_example(temp_file)
    
    # Show plots
    plt.show()
```

## Example 3: Model Evaluation and Hyperparameter Tuning

```python
"""
Model evaluation and hyperparameter tuning with scikit-learn.
This example demonstrates:
1. Cross-validation
2. Grid search for hyperparameter tuning
3. Learning curves
4. Feature importance
5. Model comparison
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import logging
from typing import Dict, Tuple, Any, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_evaluation')

def create_model_pipeline(model_name: str, random_state: int = 42) -> Tuple[Pipeline, Dict[str, List]]:
    """
    Create a pipeline with a specific model and its parameter grid for tuning.
    
    Args:
        model_name: Name of the model to use
        random_state: Random seed for reproducibility
        
    Returns:
        Pipeline and parameter grid dictionary
    """
    # Create base pipeline with scaling
    pipeline_steps = [
        ('scaler', StandardScaler())
    ]
    
    # Add model and parameter grid based on model_name
    if model_name == 'random_forest':
        model = RandomForestClassifier(random_state=random_state)
        pipeline_steps.append(('model', model))
        
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    
    elif model_name == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=random_state)
        pipeline_steps.append(('model', model))
        
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
            'model__min_samples_split': [2, 5],
            'model__subsample': [0.8, 1.0]
        }
    
    elif model_name == 'logistic_regression':
        model = LogisticRegression(random_state=random_state, max_iter=1000)
        pipeline_steps.append(('model', model))
        
        param_grid = {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'model__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }
        
        # Remove invalid combinations
        # 'l1' penalty only works with 'liblinear' and 'saga' solvers
        # 'elasticnet' only works with 'saga'
        # 'none' doesn't work with 'liblinear'
        valid_param_grid = []
        for params in _get_param_combinations(param_grid):
            penalty = params['model__penalty']
            solver = params['model__solver']
            
            if (penalty == 'l1' and solver not in ['liblinear', 'saga']) or \
               (penalty == 'elasticnet' and solver != 'saga') or \
               (penalty == 'none' and solver == 'liblinear'):
                continue
            
            valid_param_grid.append(params)
        
        # Convert back to grid format
        param_grid = {k: sorted(list(set(p[k] for p in valid_param_grid))) 
                     for k in valid_param_grid[0].keys()}
    
    elif model_name == 'svm':
        model = SVC(probability=True, random_state=random_state)
        pipeline_steps.append(('model', model))
        
        param_grid = {
            'model__C': [0.1, 1, 10, 100],
            'model__gamma': ['scale', 'auto', 0.1, 0.01],
            'model__kernel': ['rbf', 'linear', 'poly']
        }
    
    elif model_name == 'decision_tree':
        model = DecisionTreeClassifier(random_state=random_state)
        pipeline_steps.append(('model', model))
        
        param_grid = {
            'model__max_depth': [None, 5, 10, 15, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__criterion': ['gini', 'entropy']
        }
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    pipeline = Pipeline(pipeline_steps)
    
    return pipeline, param_grid

def _get_param_combinations(param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """
    Get all combinations of parameters from a parameter grid.
    Helper function for valid parameter combinations.
    """
    import itertools
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    return [dict(zip(keys, combo)) for combo in combinations]

def tune_hyperparameters(
    pipeline: Pipeline,
    param_grid: Dict[str, List],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    n_jobs: int = -1,
    method: str = 'grid',
    n_iter: int = 20,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Tune hyperparameters using grid search or randomized search.
    
    Args:
        pipeline: Scikit-learn pipeline
        param_grid: Parameter grid
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs
        method: 'grid' for grid search or 'random' for randomized search
        n_iter: Number of iterations for randomized search
        random_state: Random seed for reproducibility
        
    Returns:
        Tuned pipeline and search results
    """
    logger.info(f"Tuning hyperparameters using {method} search")
    
    if method == 'grid':
        search = GridSearchCV(
            pipeline, param_grid, cv=cv, n_jobs=n_jobs,
            scoring='accuracy', return_train_score=True,
            verbose=1
        )
    elif method == 'random':
        search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=n_iter, cv=cv, n_jobs=n_jobs,
            scoring='accuracy', return_train_score=True,
            random_state=random_state, verbose=1
        )
    else:
        raise ValueError(f"Unsupported search method: {method}")
    
    # Fit the search
    logger.info("Starting search...")
    start_time = pd.Timestamp.now()
    search.fit(X_train, y_train)
    search_time = (pd.Timestamp.now() - start_time).total_seconds()
    
    # Log results
    logger.info(f"Search completed in {search_time:.2f} seconds")
    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best cross-validation score: {search.best_score_:.4f}")
    
    # Create results dictionary
    search_results = {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': search.cv_results_,
        'search_time': search_time
    }
    
    return search.best_estimator_, search_results

def plot_learning_curve(
    estimator: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    n_jobs: int = -1,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
    title: str = "Learning Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a learning curve for the estimator.
    
    Args:
        estimator: Trained scikit-learn pipeline
        X: Features
        y: Labels
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs
        train_sizes: Array of training set sizes to try
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info("Generating learning curve")
    
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs,
        train_sizes=train_sizes, scoring='accuracy',
        shuffle=True, random_state=42
    )
    
    # Calculate mean and std for train and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot learning curve
    ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    
    ax.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-Validation Score')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Accuracy Score')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True)
    
    # Add text with final scores
    final_train_score = train_mean[-1]
    final_test_score = test_mean[-1]
    score_text = (
        f"Final Training Score: {final_train_score:.4f}\n"
        f"Final Test Score: {final_test_score:.4f}\n"
        f"Gap: {final_train_score - final_test_score:.4f}"
    )
    
    plt.figtext(0.02, 0.02, score_text, fontsize=10, bbox=dict(
        facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Learning curve saved to {save_path}")
    
    plt.tight_layout()
    return fig

def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    title: str = "Feature Importance",
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model (must have feature_importances_ attribute)
        feature_names: List of feature names
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure or None if model doesn't support feature importance
    """
    logger.info("Generating feature importance plot")
    
    # Check if model supports feature importance
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model doesn't have feature_importances_ attribute")
        return None
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot feature importances
    ax.bar(range(len(sorted_importances)), sorted_importances, align='center')
    ax.set_xticks(range(len(sorted_importances)))
    ax.set_xticklabels(sorted_feature_names, rotation=90)
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title(title)
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.tight_layout()
    return fig

def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_names: List[str],
    cv: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Compare multiple models on the same dataset.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_names: List of model names to compare
        cv: Number of cross-validation folds
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Comparing {len(model_names)} models: {model_names}")
    
    results = {
        'pipelines': {},
        'metrics': {},
        'cv_scores': {},
        'train_times': {},
        'predict_times': {}
    }
    
    for model_name in model_names:
        logger.info(f"Training {model_name}")
        
        # Create model pipeline
        pipeline, _ = create_model_pipeline(model_name, random_state)
        
        # Fit the model
        start_time = pd.Timestamp.now()
        pipeline.fit(X_train, y_train)
        train_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Make predictions
        start_time = pd.Timestamp.now()
        y_pred = pipeline.predict(X_test)
        predict_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Cross-validation scores
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f}")
        
        # Store results
        results['pipelines'][model_name] = pipeline
        results['metrics'][model_name] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': report
        }
        results['cv_scores'][model_name] = cv_scores
        results['train_times'][model_name] = train_time
        results['predict_times'][model_name] = predict_time
    
    return results

def plot_model_comparison(
    comparison_results: Dict[str, Any],
    title: str = "Model Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot model comparison results.
    
    Args:
        comparison_results: Results from compare_models function
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info("Generating model comparison plot")
    
    # Extract data
    model_names = list(comparison_results['metrics'].keys())
    accuracies = [comparison_results['metrics'][name]['accuracy'] for name in model_names]
    cv_means = [comparison_results['cv_scores'][name].mean() for name in model_names]
    cv_stds = [comparison_results['cv_scores'][name].std() for name in model_names]
    train_times = [comparison_results['train_times'][name] for name in model_names]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot accuracy comparison
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1.bar(x - width/2, cv_means, width, yerr=cv_stds, label='Cross-Validation', color='skyblue')
    ax1.bar(x + width/2, accuracies, width, label='Test Set', color='lightcoral')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy Score')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot training time comparison
    ax2.bar(model_names, train_times, color='lightgreen')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Model Training Time Comparison')
    ax2.set_xticklabels(model_names, rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    return fig

def run_model_evaluation_example(data_file: str, target
def run_model_evaluation_example(data_file: str, target_column: str) -> Dict[str, Any]:
    """
    Run the full model evaluation and tuning example.
    
    Args:
        data_file: Path to the data file
        target_column: Name of the target column
        
    Returns:
        Dictionary with all results
    """
    # Load data
    df = load_data(data_file)
    
    # Preprocess data
    data = preprocess_data(df, target_column)
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    # Create and tune Random Forest model
    rf_pipeline, rf_param_grid = create_model_pipeline('random_forest')
    tuned_rf, rf_tuning_results = tune_hyperparameters(
        rf_pipeline, rf_param_grid, X_train, y_train, method='grid'
    )
    
    # Generate learning curve
    rf_learning_curve = plot_learning_curve(
        tuned_rf, X_train, y_train, 
        title="Random Forest Learning Curve",
        save_path="rf_learning_curve.png"
    )
    
    # Get feature importance
    feature_names = data['categorical_columns'] + data['numeric_columns']
    rf_feature_importance = plot_feature_importance(
        tuned_rf.named_steps['model'], feature_names,
        title="Random Forest Feature Importance",
        save_path="rf_feature_importance.png"
    )
    
    # Compare multiple models
    model_comparison = compare_models(
        X_train, y_train, X_test, y_test,
        model_names=['random_forest', 'gradient_boosting', 'logistic_regression', 'decision_tree']
    )
    
    # Plot model comparison
    comparison_plot = plot_model_comparison(
        model_comparison, title="Model Comparison",
        save_path="model_comparison.png"
    )
    
    # Return all results
    return {
        'data': data,
        'rf_tuning_results': rf_tuning_results,
        'tuned_rf': tuned_rf,
        'model_comparison': model_comparison,
        'figures': {
            'rf_learning_curve': rf_learning_curve,
            'rf_feature_importance': rf_feature_importance,
            'comparison_plot': comparison_plot
        }
    }

# Helper function for data preprocessing
def preprocess_data(
    df: pd.DataFrame, 
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Preprocess the data for machine learning.
    
    This function is simplified for this example.
    """
    # Identify feature and target columns
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    # Identify column types
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target from feature columns
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    
    # Extract features and target
    X = df[categorical_columns + numeric_columns].values
    y = df[target_column].values
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'categorical_columns': categorical_columns,
        'numeric_columns': numeric_columns,
        'target_column': target_column
    }

# Helper function to load data
def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a file (CSV, Excel, etc.)."""
    logger.info(f"Loading data from {filepath}")
    
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

# If run directly, use this example
if __name__ == "__main__":
    # For this example, we'll use the Iris dataset
    from sklearn.datasets import load_iris
    
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(
        data=np.c_[iris['data'], iris['target']],
        columns=iris['feature_names'] + ['target']
    )
    
    # Convert target to integers for simplicity
    df['target'] = df['target'].astype(int)
    
    # Save to a temporary CSV file
    temp_file = "iris_data.csv"
    df.to_csv(temp_file, index=False)
    
    # Run the example
    results = run_model_evaluation_example(temp_file, "target")
    
    # Show learning curve and feature importance
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(results['figures']['rf_learning_curve'])
    plt.subplot(2, 1, 2)
    plt.imshow(results['figures']['rf_feature_importance'])
    plt.show()
```

## Example 4: Time Series Forecasting with Prophet

```python
"""
Time series forecasting using Facebook Prophet.
This example demonstrates:
1. Time series data preprocessing
2. Model training with Prophet
3. Forecasting future values
4. Model evaluation
5. Handling seasonality and holidays
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import logging
from typing import Dict, Tuple, Any, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('time_series_forecasting')

def prepare_time_series_data(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare time series data for Prophet.
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column
        value_column: Name of the value column
        min_date: Optional minimum date to include
        max_date: Optional maximum date to include
        
    Returns:
        DataFrame formatted for Prophet
    """
    logger.info("Preparing time series data for Prophet")
    
    # Check if required columns exist
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found")
    
    if value_column not in df.columns:
        raise ValueError(f"Value column '{value_column}' not found")
    
    # Convert to Prophet format (ds, y)
    prophet_df = df[[date_column, value_column]].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Ensure date column is datetime type
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # Filter by date range if specified
    if min_date:
        prophet_df = prophet_df[prophet_df['ds'] >= pd.to_datetime(min_date)]
    
    if max_date:
        prophet_df = prophet_df[prophet_df['ds'] <= pd.to_datetime(max_date)]
    
    # Sort by date
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    logger.info(f"Prepared data with {len(prophet_df)} rows from "
                f"{prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
    
    return prophet_df

def add_holidays_to_model(model: Prophet, country: str = 'US') -> Prophet:
    """
    Add country-specific holidays to the Prophet model.
    
    Args:
        model: Prophet model
        country: Country code for holidays
        
    Returns:
        Prophet model with holidays
    """
    logger.info(f"Adding {country} holidays to model")
    
    # Get holiday definitions
    from prophet.holidays import get_holiday_names, get_holiday_dates
    
    # Add country holidays
    model.add_country_holidays(country_name=country)
    
    return model

def train_prophet_model(
    df: pd.DataFrame,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
    add_holidays: bool = True,
    holidays_country: str = 'US',
    cap: Optional[float] = None,
    floor: Optional[float] = None,
    growth: str = 'linear',
    changepoints: Optional[List[str]] = None,
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    holidays_prior_scale: float = 10.0
) -> Tuple[Prophet, Dict[str, Any]]:
    """
    Train a Prophet model for time series forecasting.
    
    Args:
        df: DataFrame with 'ds' and 'y' columns
        yearly_seasonality: Whether to include yearly seasonality
        weekly_seasonality: Whether to include weekly seasonality
        daily_seasonality: Whether to include daily seasonality
        add_holidays: Whether to add holidays
        holidays_country: Country code for holidays
        cap: Maximum value for logistic growth
        floor: Minimum value for logistic growth
        growth: Growth model ('linear' or 'logistic')
        changepoints: List of changepoint dates
        changepoint_prior_scale: Flexibility of the changepoints
        seasonality_prior_scale: Flexibility of the seasonality
        holidays_prior_scale: Flexibility of the holidays
        
    Returns:
        Trained Prophet model and training info
    """
    logger.info("Training Prophet model")
    
    # Create model with specified parameters
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale,
        growth=growth
    )
    
    # Add holidays if requested
    if add_holidays:
        model = add_holidays_to_model(model, holidays_country)
    
    # Add capacity for logistic growth
    if growth == 'logistic':
        if cap is None:
            raise ValueError("Cap must be specified for logistic growth")
        
        df = df.copy()
        df['cap'] = cap
        
        if floor is not None:
            df['floor'] = floor
    
    # Add custom changepoints if provided
    if changepoints:
        model.add_changepoints(pd.to_datetime(changepoints))
    
    # Fit the model
    start_time = pd.Timestamp.now()
    model.fit(df)
    training_time = (pd.Timestamp.now() - start_time).total_seconds()
    
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    
    # Create training info
    training_info = {
        'start_date': df['ds'].min(),
        'end_date': df['ds'].max(),
        'data_points': len(df),
        'training_time': training_time,
        'model_params': {
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'daily_seasonality': daily_seasonality,
            'add_holidays': add_holidays,
            'holidays_country': holidays_country if add_holidays else None,
            'growth': growth,
            'cap': cap,
            'floor': floor,
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale,
            'holidays_prior_scale': holidays_prior_scale
        }
    }
    
    return model, training_info

def make_forecast(
    model: Prophet,
    periods: int = 30,
    freq: str = 'D',
    include_history: bool = True
) -> pd.DataFrame:
    """
    Generate a forecast using the trained Prophet model.
    
    Args:
        model: Trained Prophet model
        periods: Number of periods to forecast
        freq: Frequency of forecast (D for daily, W for weekly, etc.)
        include_history: Whether to include historical data
        
    Returns:
        DataFrame with forecast
    """
    logger.info(f"Generating forecast for {periods} {freq} periods")
    
    # Create future dataframe
    future = model.make_future_dataframe(
        periods=periods, 
        freq=freq, 
        include_history=include_history
    )
    
    # Add capacity for logistic growth if needed
    if model.growth == 'logistic':
        if 'cap' in model.history.columns:
            future['cap'] = model.history['cap'].max()
        
        if 'floor' in model.history.columns:
            future['floor'] = model.history['floor'].min()
    
    # Make predictions
    forecast = model.predict(future)
    
    logger.info(f"Generated forecast from {forecast['ds'].min()} to {forecast['ds'].max()}")
    
    return forecast

def evaluate_forecast(
    model: Prophet,
    horizon: str = '30 days',
    period: str = '180 days',
    initial: str = '365 days'
) -> Dict[str, Any]:
    """
    Evaluate forecast accuracy using cross-validation.
    
    Args:
        model: Trained Prophet model
        horizon: Forecast horizon
        period: Period between cutoffs
        initial: Initial training period
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating forecast (horizon: {horizon}, period: {period}, initial: {initial})")
    
    # Perform cross-validation
    start_time = pd.Timestamp.now()
    cv_results = cross_validation(
        model, horizon=horizon, period=period, initial=initial
    )
    cv_time = (pd.Timestamp.now() - start_time).total_seconds()
    
    # Calculate performance metrics
    metrics = performance_metrics(cv_results)
    
    logger.info(f"Cross-validation completed in {cv_time:.2f} seconds")
    logger.info(f"RMSE: {metrics['rmse'].mean():.4f}, MAE: {metrics['mae'].mean():.4f}")
    
    # Create evaluation results
    evaluation = {
        'cv_results': cv_results,
        'metrics': metrics,
        'cv_time': cv_time,
        'horizon': horizon,
        'period': period,
        'initial': initial
    }
    
    return evaluation

def plot_forecast(
    model: Prophet,
    forecast: pd.DataFrame,
    history_df: pd.DataFrame,
    uncertainty: bool = True,
    components: bool = True,
    title: str = "Time Series Forecast",
    save_path: Optional[str] = None
) -> List[plt.Figure]:
    """
    Plot forecast results.
    
    Args:
        model: Trained Prophet model
        forecast: Forecast DataFrame
        history_df: Historical data DataFrame
        uncertainty: Whether to show uncertainty intervals
        components: Whether to plot forecast components
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        List of Matplotlib figures
    """
    logger.info("Generating forecast plots")
    
    figures = []
    
    # Plot forecast
    fig1 = model.plot(forecast, uncertainty=uncertainty)
    ax1 = fig1.axes[0]
    ax1.set_title(title)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    
    # Add actual historical data points
    ax1.scatter(history_df['ds'], history_df['y'], color='black', s=10, alpha=0.5, label='Actual')
    ax1.legend()
    
    figures.append(fig1)
    
    # Plot forecast components if requested
    if components:
        fig2 = model.plot_components(forecast)
        figures.append(fig2)
    
    # Save plots if path provided
    if save_path:
        base_path = save_path.rsplit('.', 1)[0]
        
        fig1.savefig(f"{base_path}_forecast.png", dpi=300, bbox_inches='tight')
        logger.info(f"Forecast plot saved to {base_path}_forecast.png")
        
        if components:
            fig2.savefig(f"{base_path}_components.png", dpi=300, bbox_inches='tight')
            logger.info(f"Components plot saved to {base_path}_components.png")
    
    return figures

def plot_cross_validation(
    evaluation: Dict[str, Any],
    metric: str = 'mape',
    title: str = "Cross-Validation Performance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot cross-validation results.
    
    Args:
        evaluation: Evaluation results from evaluate_forecast
        metric: Metric to plot ('mse', 'rmse', 'mae', 'mape', or 'mdape')
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Generating cross-validation plot for {metric}")
    
    cv_results = evaluation['cv_results']
    
    # Create figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Plot cross-validation metric
    plot_cross_validation_metric(cv_results, metric=metric, ax=ax)
    
    ax.set_title(title)
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Cross-validation plot saved to {save_path}")
    
    return fig

def run_time_series_example(
    data_file: str,
    date_column: str,
    value_column: str,
    forecast_periods: int = 90
) -> Dict[str, Any]:
    """
    Run the full time series forecasting example.
    
    Args:
        data_file: Path to the data file
        date_column: Name of the date column
        value_column: Name of the value column
        forecast_periods: Number of periods to forecast
        
    Returns:
        Dictionary with all results
    """
    # Load data
    df = load_data(data_file)
    
    # Prepare data for Prophet
    prophet_df = prepare_time_series_data(df, date_column, value_column)
    
    # Train model
    model, training_info = train_prophet_model(
        prophet_df,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        add_holidays=True
    )
    
    # Generate forecast
    forecast = make_forecast(model, periods=forecast_periods)
    
    # Evaluate model
    evaluation = evaluate_forecast(model)
    
    # Create plots
    forecast_plots = plot_forecast(
        model, forecast, prophet_df,
        title="Time Series Forecast",
        save_path="time_series_forecast.png"
    )
    
    cv_plot = plot_cross_validation(
        evaluation, metric='rmse',
        title="Forecast Error (RMSE)",
        save_path="time_series_cv.png"
    )
    
    # Return all results
    return {
        'data': prophet_df,
        'model': model,
        'training_info': training_info,
        'forecast': forecast,
        'evaluation': evaluation,
        'figures': {
            'forecast_plots': forecast_plots,
            'cv_plot': cv_plot
        }
    }

# Helper function to load data
def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a file (CSV, Excel, etc.)."""
    logger.info(f"Loading data from {filepath}")
    
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

# If run directly, use this example
if __name__ == "__main__":
    # Generate sample time series data
    # (In a real application, you would load this from a file)
    dates = pd.date_range(start='2020-01-01', end='2022-12-31')
    
    # Create synthetic data with trend and seasonality
    np.random.seed(42)
    trend = np.linspace(0, 100, len(dates))
    yearly_seasonality = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    weekly_seasonality = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    noise = 10 * np.random.randn(len(dates))
    
    values = trend + yearly_seasonality + weekly_seasonality + noise
    
    # Create DataFrame
    sample_df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    # Save to a temporary CSV file
    temp_file = "sample_time_series.csv"
    sample_df.to_csv(temp_file, index=False)
    
    # Run the example
    results = run_time_series_example(temp_file, 'date', 'value')
    
    # Show plots
    for fig in results['figures']['forecast_plots']:
        plt.figure(figsize=(12, 8))
        plt.imshow(fig)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(results['figures']['cv_plot'])
    plt.show()
```
# Web Scraping with Beautiful Soup Examples

## Example 1: Basic Web Scraping

```python
"""
Basic web scraping with Beautiful Soup.
This example demonstrates:
1. Making HTTP requests
2. Parsing HTML with Beautiful Soup
3. Extracting structured data
4. Handling common web scraping issues
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('web_scraper')

class WebScraper:
    """A basic web scraper using Beautiful Soup."""
    
    def __init__(
        self,
        base_url: str,
        user_agent: Optional[str] = None,
        timeout: int = 10,
        retry_count: int = 3,
        delay_range: tuple = (1, 3)
    ):
        """
        Initialize the web scraper.
        
        Args:
            base_url: Base URL for the website
            user_agent: User agent string to use for requests
            timeout: Request timeout in seconds
            retry_count: Number of retry attempts
            delay_range: Range of seconds to delay between requests
        """
        self.base_url = base_url
        self.session = requests.Session()
        
        # Set default user agent if not provided
        if user_agent is None:
            user_agent = (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
            )
        
        # Set up session headers
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache'
        })
        
        self.timeout = timeout
        self.retry_count = retry_count
        self.delay_range = delay_range
    
    def get_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Get a web page and parse it with Beautiful Soup.
        
        Args:
            url: URL to request
            
        Returns:
            BeautifulSoup object or None if request failed
        """
        # Resolve relative URLs
        if not url.startswith('http'):
            url = urljoin(self.base_url, url)
        
        logger.info(f"Fetching: {url}")
        
        # Implement retry logic
        for attempt in range(self.retry_count):
            try:
                # Add a random delay to reduce server load and avoid throttling
                if attempt > 0:
                    delay = random.uniform(*self.delay_range)
                    logger.info(f"Retry {attempt+1}/{self.retry_count}. Waiting {delay:.2f}s...")
                    time.sleep(delay)
                
                # Make the request
                response = self.session.get(url, timeout=self.timeout)
                
                # Check for successful response
                if response.status_code == 200:
                    # Parse the HTML
                    soup = BeautifulSoup(response.text, 'html.parser')
                    return soup
                else:
                    logger.warning(f"HTTP Error: {response.status_code} for {url}")
            
            except (requests.RequestException, Exception) as e:
                logger.error(f"Request failed: {str(e)}")
        
        logger.error(f"Failed to retrieve {url} after {self.retry_count} attempts")
        return None
    
    def extract_links(self, soup: BeautifulSoup, css_selector: str) -> List[str]:
        """
        Extract links from a Beautiful Soup object.
        
        Args:
            soup: BeautifulSoup object
            css_selector: CSS selector for the links
            
        Returns:
            List of extracted link URLs
        """
        links = []
        
        try:
            for link in soup.select(css_selector):
                href = link.get('href')
                if href:
                    # Resolve relative URLs
                    absolute_url = urljoin(self.base_url, href)
                    links.append(absolute_url)
        
        except Exception as e:
            logger.error(f"Error extracting links: {str(e)}")
        
        logger.info(f"Extracted {len(links)} links")
        return links
    
    def extract_text(self, soup: BeautifulSoup, css_selector: str) -> str:
        """
        Extract text from a Beautiful Soup object.
        
        Args:
            soup: BeautifulSoup object
            css_selector: CSS selector for the element
            
        Returns:
            Extracted text
        """
        try:
            element = soup.select_one(css_selector)
            if element:
                # Get text and normalize whitespace
                return ' '.join(element.get_text(strip=True).split())
            else:
                logger.warning(f"No element found for selector: {css_selector}")
                return ""
        
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return ""
    
    def extract_data_from_page(
        self, 
        soup: BeautifulSoup, 
        selectors: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Extract data from a page using multiple selectors.
        
        Args:
            soup: BeautifulSoup object
            selectors: Dictionary mapping field names to CSS selectors
            
        Returns:
            Dictionary of extracted data
        """
        data = {}
        
        for field, selector in selectors.items():
            data[field] = self.extract_text(soup, selector)
        
        return data
    
    def scrape_pages(
        self, 
        urls: List[str], 
        selectors: Dict[str, str],
        max_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Scrape multiple pages and extract data.
        
        Args:
            urls: List of URLs to scrape
            selectors: Dictionary mapping field names to CSS selectors
            max_pages: Maximum number of pages to scrape (None for all)
            
        Returns:
            List of dictionaries containing scraped data
        """
        results = []
        
        # Limit the number of pages if specified
        if max_pages:
            urls = urls[:max_pages]
        
        for url in urls:
            # Add a delay between requests
            delay = random.uniform(*self.delay_range)
            time.sleep(delay)
            
            # Get and parse the page
            soup = self.get_page(url)
            
            if soup:
                # Extract data from the page
                data = self.extract_data_from_page(soup, selectors)
                
                # Add the URL to the data
                data['url'] = url
                
                # Add the data to results
                results.append(data)
        
        logger.info(f"Scraped {len(results)} pages")
        return results
    
    def save_results(self, results: List[
def save_results(self, results: List[Dict[str, Any]], output_file: str) -> None:
        """
        Save scraped results to a file.
        
        Args:
            results: List of dictionaries containing scraped data
            output_file: Path to output file (CSV or JSON)
        """
        if not results:
            logger.warning("No results to save")
            return
        
        logger.info(f"Saving {len(results)} results to {output_file}")
        
        try:
            # Determine file format based on extension
            if output_file.endswith('.csv'):
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False)
                
            elif output_file.endswith('.json'):
                import json
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                    
            else:
                logger.error(f"Unsupported output format: {output_file}")
                return
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

# Example usage
def scrape_example_website():
    """Example of scraping a simple website."""
    # We'll use Python's official jobs board for this example
    # This is a public website that explicitly allows scraping
    base_url = "https://python.org"
    scraper = WebScraper(base_url)
    
    # Get the jobs page
    jobs_url = "https://www.python.org/jobs/"
    jobs_page = scraper.get_page(jobs_url)
    
    if not jobs_page:
        logger.error("Failed to retrieve jobs page")
        return
    
    # Extract job listing links
    job_links = scraper.extract_links(jobs_page, "h2.listing-company a")
    
    if not job_links:
        logger.error("No job links found")
        return
    
    # Define selectors for job details
    job_selectors = {
        'title': 'h1',
        'company': '.company-name',
        'location': '.listing-location',
        'description': '.job-description',
        'posted_date': '.listing-posted',
        'requirements': '.job-requirements'
    }
    
    # Scrape job details (limit to 5 for the example)
    job_details = scraper.scrape_pages(job_links, job_selectors, max_pages=5)
    
    # Save results
    scraper.save_results(job_details, "python_jobs.csv")
    scraper.save_results(job_details, "python_jobs.json")
    
    return job_details

if __name__ == "__main__":
    scrape_example_website()
```

## Example 2: Advanced Scraping with Pagination and AJAX Content

```python
"""
Advanced web scraping with pagination and AJAX content.
This example demonstrates:
1. Handling pagination
2. Working with dynamic content loaded via AJAX
3. Implementing proper delays and rate limiting
4. Managing sessions and cookies
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import json
import logging
from typing import List, Dict, Any, Optional, Iterator
from urllib.parse import urljoin, urlparse, parse_qs

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('advanced_scraper')

class AdvancedScraper:
    """Advanced web scraper with pagination and AJAX handling."""
    
    def __init__(
        self,
        base_url: str,
        user_agent: Optional[str] = None,
        timeout: int = 30,
        retry_count: int = 3,
        delay_range: tuple = (2, 5),
        max_requests_per_minute: int = 20
    ):
        """
        Initialize the advanced scraper.
        
        Args:
            base_url: Base URL for the website
            user_agent: User agent string to use for requests
            timeout: Request timeout in seconds
            retry_count: Number of retry attempts
            delay_range: Range of seconds to delay between requests
            max_requests_per_minute: Maximum number of requests per minute
        """
        self.base_url = base_url
        self.session = requests.Session()
        
        # Set default user agent if not provided
        if user_agent is None:
            user_agent = (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
            )
        
        # Set up session headers
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
            'X-Requested-With': 'XMLHttpRequest'  # Needed for some AJAX requests
        })
        
        self.timeout = timeout
        self.retry_count = retry_count
        self.delay_range = delay_range
        
        # Rate limiting
        self.min_request_interval = 60.0 / max_requests_per_minute
        self.last_request_time = 0
    
    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_html(self, url: str, params: Optional[Dict] = None) -> Optional[str]:
        """
        Get HTML content from a URL.
        
        Args:
            url: URL to request
            params: Optional query parameters
            
        Returns:
            HTML content as string or None if request failed
        """
        # Resolve relative URLs
        if not url.startswith('http'):
            url = urljoin(self.base_url, url)
        
        logger.info(f"Fetching HTML: {url}")
        
        # Implement retry logic
        for attempt in range(self.retry_count):
            try:
                # Apply rate limiting
                self._wait_for_rate_limit()
                
                # Add a random delay to reduce server load and avoid throttling
                if attempt > 0:
                    delay = random.uniform(*self.delay_range)
                    logger.info(f"Retry {attempt+1}/{self.retry_count}. Waiting {delay:.2f}s...")
                    time.sleep(delay)
                
                # Make the request
                response = self.session.get(url, params=params, timeout=self.timeout)
                
                # Check for successful response
                if response.status_code == 200:
                    return response.text
                else:
                    logger.warning(f"HTTP Error: {response.status_code} for {url}")
            
            except (requests.RequestException, Exception) as e:
                logger.error(f"Request failed: {str(e)}")
        
        logger.error(f"Failed to retrieve {url} after {self.retry_count} attempts")
        return None
    
    def get_json(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Get JSON content from a URL (for AJAX requests).
        
        Args:
            url: URL to request
            params: Optional query parameters
            
        Returns:
            JSON content as dictionary or None if request failed
        """
        # Resolve relative URLs
        if not url.startswith('http'):
            url = urljoin(self.base_url, url)
        
        logger.info(f"Fetching JSON: {url}")
        
        # Update headers for JSON request
        original_accept = self.session.headers.get('Accept')
        self.session.headers['Accept'] = 'application/json, text/javascript, */*; q=0.01'
        
        # Implement retry logic
        for attempt in range(self.retry_count):
            try:
                # Apply rate limiting
                self._wait_for_rate_limit()
                
                # Add a random delay to reduce server load and avoid throttling
                if attempt > 0:
                    delay = random.uniform(*self.delay_range)
                    logger.info(f"Retry {attempt+1}/{self.retry_count}. Waiting {delay:.2f}s...")
                    time.sleep(delay)
                
                # Make the request
                response = self.session.get(url, params=params, timeout=self.timeout)
                
                # Check for successful response
                if response.status_code == 200:
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON response from {url}")
                        return None
                else:
                    logger.warning(f"HTTP Error: {response.status_code} for {url}")
            
            except (requests.RequestException, Exception) as e:
                logger.error(f"Request failed: {str(e)}")
        
        # Restore original Accept header
        if original_accept:
            self.session.headers['Accept'] = original_accept
        
        logger.error(f"Failed to retrieve JSON from {url} after {self.retry_count} attempts")
        return None
    
    def parse_html(self, html: str) -> Optional[BeautifulSoup]:
        """
        Parse HTML content with Beautiful Soup.
        
        Args:
            html: HTML content as string
            
        Returns:
            BeautifulSoup object or None if parsing failed
        """
        try:
            return BeautifulSoup(html, 'html.parser')
        except Exception as e:
            logger.error(f"Error parsing HTML: {str(e)}")
            return None
    
    def get_pagination_links(
        self, 
        soup: BeautifulSoup, 
        pagination_selector: str,
        base_url: Optional[str] = None
    ) -> List[str]:
        """
        Extract pagination links from a page.
        
        Args:
            soup: BeautifulSoup object
            pagination_selector: CSS selector for pagination links
            base_url: Base URL for resolving relative links
            
        Returns:
            List of pagination URLs
        """
        pagination_links = []
        
        if base_url is None:
            base_url = self.base_url
        
        try:
            for link in soup.select(pagination_selector):
                href = link.get('href')
                if href:
                    # Skip links that are just anchors
                    if href.startswith('#'):
                        continue
                    
                    # Resolve relative URLs
                    absolute_url = urljoin(base_url, href)
                    
                    # Avoid duplicates
                    if absolute_url not in pagination_links:
                        pagination_links.append(absolute_url)
        
        except Exception as e:
            logger.error(f"Error extracting pagination links: {str(e)}")
        
        logger.info(f"Extracted {len(pagination_links)} pagination links")
        return pagination_links
    
    def scrape_paginated_content(
        self,
        start_url: str,
        item_selector: str,
        data_selectors: Dict[str, str],
        pagination_selector: str,
        max_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Scrape content across multiple paginated pages.
        
        Args:
            start_url: URL of the first page
            item_selector: CSS selector for items on each page
            data_selectors: Dictionary mapping field names to CSS selectors
            pagination_selector: CSS selector for pagination links
            max_pages: Maximum number of pages to scrape (None for all)
            
        Returns:
            List of dictionaries containing scraped data
        """
        results = []
        pages_scraped = 0
        next_page_url = start_url
        
        while next_page_url and (max_pages is None or pages_scraped < max_pages):
            # Get page content
            html = self.get_html(next_page_url)
            
            if not html:
                break
            
            # Parse page
            soup = self.parse_html(html)
            
            if not soup:
                break
            
            # Process items on the current page
            items = soup.select(item_selector)
            logger.info(f"Found {len(items)} items on page {pages_scraped + 1}")
            
            for item in items:
                data = {}
                
                # Extract data using selectors
                for field, selector in data_selectors.items():
                    elements = item.select(selector)
                    
                    if elements:
                        # Get text and normalize whitespace
                        data[field] = ' '.join(elements[0].get_text(strip=True).split())
                    else:
                        data[field] = ""
                
                # Add the data to results
                results.append(data)
            
            # Get next page URL
            pages_scraped += 1
            
            # Check if we've reached the maximum number of pages
            if max_pages is not None and pages_scraped >= max_pages:
                break
            
            # Get pagination links
            pagination_links = self.get_pagination_links(soup, pagination_selector, next_page_url)
            
            # Find the "next" link (usually the last one in pagination)
            next_page_url = pagination_links[-1] if pagination_links else None
            
            if next_page_url:
                logger.info(f"Next page: {next_page_url}")
                
                # Add a delay between pages
                delay = random.uniform(*self.delay_range)
                logger.info(f"Waiting {delay:.2f}s before next page...")
                time.sleep(delay)
        
        logger.info(f"Scraped {len(results)} items from {pages_scraped} pages")
        return results
    
    def scrape_ajax_data(
        self, 
        api_url: str,
        params_template: Dict[str, Any],
        page_param_name: str = 'page',
        data_path: Optional[List[str]] = None,
        max_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Scrape data from an AJAX API with pagination.
        
        Args:
            api_url: URL of the AJAX API
            params_template: Template for query parameters
            page_param_name: Name of the page parameter
            data_path: List of keys to access the data in the JSON response
            max_pages: Maximum number of pages to scrape (None for all)
            
        Returns:
            List of dictionaries containing scraped data
        """
        results = []
        current_page = 1
        
        while max_pages is None or current_page <= max_pages:
            # Update page parameter
            params = params_template.copy()
            params[page_param_name] = current_page
            
            # Get JSON data
            data = self.get_json(api_url, params)
            
            if not data:
                break
            
            # Navigate to the actual data in the JSON response
            if data_path:
                try:
                    for key in data_path:
                        data = data[key]
                except (KeyError, TypeError):
                    logger.error(f"Error accessing data path: {data_path}")
                    break
            
            # Check if we have a list of items
            if not isinstance(data, list):
                logger.error("Data is not a list")
                break
            
            # Add items to results
            results.extend(data)
            logger.info(f"Retrieved {len(data)} items from page {current_page}")
            
            # Check if we've reached the end of data
            if len(data) == 0:
                logger.info("No more data available")
                break
            
            # Move to next page
            current_page += 1
            
            # Add a delay between requests
            delay = random.uniform(*self.delay_range)
            logger.info(f"Waiting {delay:.2f}s before next request...")
            time.sleep(delay)
        
        logger.info(f"Scraped {len(results)} items from AJAX API")
        return results

# Example usage
def scrape_github_trending():
    """Example of scraping GitHub trending repositories."""
    base_url = "https://github.com"
    scraper = AdvancedScraper(base_url)
    
    # Starting URL for trending repositories
    trending_url = "https://github.com/trending"
    
    # Define selectors
    item_selector = "article.Box-row"
    data_selectors = {
        'repository': 'h2 a',
        'description': 'p',
        'language': 'span[itemprop="programmingLanguage"]',
        'stars': 'a.Link--muted:nth-child(1)',
        'forks': 'a.Link--muted:nth-child(2)'
    }
    pagination_selector = "a.BtnGroup-item"
    
    # Scrape trending repositories (GitHub trending has only one page, 
    # but this demonstrates the pagination functionality)
    trending_repos = scraper.scrape_paginated_content(
        trending_url,
        item_selector,
        data_selectors,
        pagination_selector,
        max_pages=1
    )
    
    # Save results
    if trending_repos:
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(trending_repos)
        
        # Clean up the data
        df['repository'] = df['repository'].str.strip().str.replace('\n', ' ').str.strip()
        df['stars'] = df['stars'].str.strip().str.replace(',', '').astype(int, errors='ignore')
        df['forks'] = df['forks'].str.strip().str.replace(',', '').astype(int, errors='ignore')
        
        # Save to CSV
        df.to_csv("github_trending.csv", index=False)
        logger.info("Results saved to github_trending.csv")
    
    return trending_repos

if __name__ == "__main__":
    scrape_github_trending()
```

## Example 3: Scraping with Selenium for JavaScript-Heavy Sites

```python
"""
Web scraping with Selenium for JavaScript-heavy sites.
This example demonstrates:
1. Using Selenium to automate browser interaction
2. Handling dynamic content loaded with JavaScript
3. Waiting for elements to load
4. Extracting data from complex pages
"""
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('selenium_scraper')

class SeleniumScraper:
    """Web scraper using Selenium for JavaScript-heavy sites."""
    
    def __init__(
        self,
        chromedriver_path: Optional[str] = None,
        headless: bool = True,
        timeout: int = 10,
        delay_range: Tuple[float, float] = (1, 3),
        user_agent: Optional[str] = None
    ):
        """
        Initialize the Selenium scraper.
        
        Args:
            chromedriver_path: Path to ChromeDriver (if None, use PATH)
            headless: Whether to run Chrome in headless mode
            timeout: Timeout for waiting for elements in seconds
            delay_range: Range of seconds to delay between actions
            user_agent: User agent string to use
        """
        self.timeout = timeout
        self.delay_range = delay_range
        
        # Set up Chrome options
        chrome_options = Options()
        
        if headless:
            chrome_options.add_argument("--headless")
        
        # Add other useful options
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Set user agent if provided
        if user_agent:
            chrome_options.add_argument(f"--user-agent={user_agent}")
        
        # Set up Chrome service
        service = Service(executable_path=chromedriver_path) if chromedriver_path else Service()
        
        # Initialize the WebDriver
        try:
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(timeout)
            logger.info("Chrome WebDriver initialized")
        except Exception as e:
            logger.error(f"Error initializing Chrome WebDriver: {str(e)}")
            raise
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'driver'):
            try:
                self.driver.quit()
                logger.info("Chrome WebDriver closed")
            except Exception as e:
                logger.error(f"Error closing WebDriver: {str(e)}")
    
    def navigate_to(self, url: str) -> bool:
        """
        Navigate to a URL.
        
        Args:
            url: URL to navigate to
            
        Returns:
            True if navigation was successful, False otherwise
        """
        logger.info(f"Navigating to: {url}")
        
        try:
            self.driver.get(url)
            return True
        except Exception as e:
            logger.error(f"Error navigating to {url}: {str(e)}")
            return False
    
    def wait_for_element(
        self, 
        selector: str, 
        by: By = By.CSS_SELECTOR, 
        timeout: Optional[int] = None
    ) -> Optional[Any]:
        """
        Wait for an element to be present on the page.
        
        Args:
            selector: Element selector
            by: Type of selector (By.CSS_SELECTOR, By.XPATH, etc.)
            timeout: Custom timeout in seconds
            
        Returns:
            WebElement if found, None otherwise
        """
        if timeout is None:
            timeout = self.timeout
        
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
            return element
        except TimeoutException:
            logger.warning(f"Timeout waiting for element: {selector}")
            return None
        except Exception as e:
            logger.error(f"Error waiting for element {selector}: {str(e)}")
            return None
    
    def wait_for_elements(
        self, 
        selector: str, 
        by: By = By.CSS_SELECTOR, 
        timeout: Optional[int] = None
    ) -> List[Any]:
        """
        Wait for elements to be present on the page.
        
        Args:
            selector: Elements selector
            by: Type of selector (By.CSS_SELECTOR, By.XPATH, etc.)
            timeout: Custom timeout in seconds
            
        Returns:
            List of WebElements if found, empty list otherwise
        """
        if timeout is None:
            timeout = self.timeout
        
        try:
            elements = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_all_elements_located((by, selector))
            )
            return elements
        except TimeoutException:
            logger.warning(f"Timeout waiting for elements: {selector}")
            return []
        except Exception as e:
            logger.error(f"Error waiting for elements {selector}: {str(e)}")
            return []
    
    def click_element(
        self, 
        selector: str, 
        by: By = By.CSS_SELECTOR,
        wait_time: Optional[int] = None
    ) -> bool:
        """
        Click on an element.
        
        Args:
            selector: Element selector
            by: Type of selector (By.CSS_SELECTOR, By.XPATH, etc.)
            wait_time: Optional time to wait after clicking
            
        Returns:
            True if click was successful, False otherwise
        """
        element = self.wait_for_element(selector, by)
        
        if element:
            try:
                element.click()
                logger.info(f"Clicked element: {selector}")
                
                # Wait if specified
                if wait_time:
                    time.sleep(wait_time)
                else:
                    # Random delay
                    delay = random.uniform(*self.delay_range)
                    time.sleep(delay)
                
                return True
            except Exception as e:
                logger.error(f"Error clicking element {selector}: {str(e)}")
                return False
        
        return False
    
    def get_element_text(
        self, 
        selector: str, 
        by: By = By.CSS_SELECTOR
    ) -> Optional[str]:
        """
        Get the text content of an element.
        
        Args:
            selector: Element selector
            by: Type of selector (By.CSS_SELECTOR, By.XPATH, etc.)
            
        Returns:
            Element text if found, None otherwise
        """
        element = self.wait_for_element(selector, by)
        
        if element:
            try:
                text = element.text.strip()
                return text
            except Exception as e:
                logger.error(f"Error getting text from element {selector}: {str(e)}")
                return None
        
        return None
    
    def get_element_attribute(
        self, 
        selector: str, 
        attribute: str, 
        by: By = By.CSS_SELECTOR
    ) -> Optional[str]:
        """
        Get an attribute value from an element.
        
        Args:
            selector: Element selector
            attribute: Attribute name
            by: Type of selector (By.CSS_SELECTOR, By.XPATH, etc.)
            
        Returns:
            Attribute value if found, None otherwise
        """
        element = self.wait_for_element(selector, by)
        
        if element:
            try:
                value = element.get_attribute(attribute)
                return value
            except Exception as e:
                logger.error(f"Error getting attribute {attribute} from element {selector}: {str(e)}")
                return None
        
        return None
    
    def scroll_to_bottom(self, scroll_pause_time: float = 1.0) -> None:
        """
        Scroll to the bottom of the page incrementally.
        
        Args:
            scroll_pause_time: Time to pause between scrolls
        """
        logger.info("Scrolling to bottom of page")
        
        # Get scroll height
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        while True:
            # Scroll down to bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # Wait to load page
            time.sleep(scroll_pause_time)
            
            # Calculate new scroll height and compare with last scroll height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        logger.info("Reached bottom of page")
    
    def get_page_source(self) -> str:
        """
        Get the current page source.
        
        Returns:
            Page source HTML
        """
        return self.driver.page_source
    
    def get_soup(self) -> BeautifulSoup:
        """
        Get a BeautifulSoup object for the current page.
        
        Returns:
            BeautifulSoup object
        """
        return BeautifulSoup(self.get_page_source(), 'html.parser')
    
    def take_screenshot(self, filename: str) -> bool:
        """
        Take a screenshot of the current page.
        
        Args:
            filename: Path to save the screenshot
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.driver.save_screenshot(filename)
            logger.info(f"Screenshot saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error taking screenshot: {str(e)}")
            return False
    
    def extract_data_with_selectors(
        self, 
        selectors: Dict[str, Tuple[By, str]]
    ) -> Dict[str, Any]:
        """
        Extract data from the current page using multiple selectors.
        
        Args:
            selectors: Dictionary mapping field names to (By, selector) tuples
            
        Returns:
            Dictionary of extracted data
        """
        data = {}
        
        for field, (by, selector) in selectors.items():
            try:
                element = self.wait_for_element(selector, by)
                
                if element:
                    data[field] = element.text.strip()
                else:
                    data[field] = None
            except Exception as e:
                logger.error(f"Error extracting {field}: {str(e)}")
                data[field] = None
        
        return data
    
    def scrape_dynamic_paginated_content(
        self,
        url: str,
        item_selector: str,
        data_selectors: Dict[str, Tuple[By, str]],
        next_button_selector: str,
        max_pages: Optional[int] = None,
        wait_for_page_load: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Scrape content from a dynamic paginated website.
        
        Args:
            url: Starting URL
            item_selector: CSS selector for items on each page
            data_selectors: Dict mapping field names to (By, selector) tuples
            next_button_selector: CSS selector for the next page button
            max_pages: Maximum number of pages to scrape
            wait_for_page_load: Time to wait after clicking next page
            
        Returns:
            List of dictionaries containing scraped data
        """
        results = []
        pages_scraped = 0
        
        # Navigate to the starting URL
        if not self.navigate_to(url):
            return results
        
        # Process pages
        while True:
            # Wait for page to load
