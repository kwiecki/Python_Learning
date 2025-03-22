# Debugging and Performance Profiling Examples

# Example 1: Using Python's built-in debugger (pdb)
"""
The Python debugger (pdb) allows you to step through code execution,
inspect variables, and track down issues.

Basic pdb commands:
- n (next): Execute the current line and move to the next one
- s (step): Step into a function call
- c (continue): Continue execution until the next breakpoint
- q (quit): Quit the debugger
- p expression: Print the value of an expression
- l (list): Show the current line and surrounding code
- h (help): Show help for debugger commands
"""

def buggy_function(data):
    """A function with a bug for debugging demonstration"""
    result = []
    for i, value in enumerate(data):
        # Bug: using i + 1 instead of i as index
        new_value = value * (i + 1)
        result.append(new_value)
    
    # Bug: summing the list incorrectly
    total = 0
    for i in range(1, len(result)):  # Should start from 0, not 1
        total += result[i]
    
    return result, total


# Method 1: Hard-coded breakpoint
def debug_with_breakpoint():
    data = [1, 2, 3, 4, 5]
    
    # Add a breakpoint
    import pdb; pdb.set_trace()
    
    # Or in Python 3.7+, you can use the built-in breakpoint() function
    # breakpoint()
    
    result, total = buggy_function(data)
    print(f"Result: {result}")
    print(f"Total: {total}")


# Method 2: Using try/except with post-mortem debugging
def debug_with_postmortem():
    data = [1, 2, 3, 4, 5]
    
    try:
        result, total = buggy_function(data)
        print(f"Result: {result}")
        print(f"Total: {total}")
        
        # Verify the result
        assert total == sum(result), "Total doesn't match sum of results"
        
    except Exception as e:
        import pdb
        print(f"Error occurred: {e}")
        pdb.post_mortem()


# Method 3: Using a context manager for debugging
def debug_with_context_manager():
    import contextlib
    
    @contextlib.contextmanager
    def debug_context():
        try:
            yield
        except Exception as e:
            import pdb
            print(f"Error occurred: {e}")
            pdb.post_mortem()
    
    data = [1, 2, 3, 4, 5]
    
    with debug_context():
        result, total = buggy_function(data)
        print(f"Result: {result}")
        print(f"Total: {total}")
        
        # Verify the result
        assert total == sum(result), "Total doesn't match sum of results"


# Example 2: Using print debugging
def debug_with_print():
    data = [1, 2, 3, 4, 5]
    print(f"Input data: {data}")
    
    result = []
    for i, value in enumerate(data):
        new_value = value * (i + 1)
        print(f"Processing index {i}, value {value}, new_value {new_value}")
        result.append(new_value)
    
    print(f"Intermediate result: {result}")
    
    total = 0
    for i in range(1, len(result)):
        print(f"Adding result[{i}] = {result[i]} to total")
        total += result[i]
    
    print(f"Final result: {result}, total: {total}")
    print(f"Sum of result: {sum(result)}")
    
    return result, total


# Example 3: Using logging for debugging
import logging

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )

def debug_with_logging():
    setup_logging()
    logger = logging.getLogger("buggy_function")
    
    data = [1, 2, 3, 4, 5]
    logger.info(f"Input data: {data}")
    
    result = []
    for i, value in enumerate(data):
        new_value = value * (i + 1)
        logger.debug(f"Processing index {i}, value {value}, new_value {new_value}")
        result.append(new_value)
    
    logger.debug(f"Intermediate result: {result}")
    
    total = 0
    for i in range(1, len(result)):
        logger.debug(f"Adding result[{i}] = {result[i]} to total")
        total += result[i]
    
    logger.info(f"Final result: {result}, total: {total}")
    logger.debug(f"Sum of result: {sum(result)}")
    
    return result, total


# Example 4: CPU profiling with cProfile
import cProfile
import pstats
import io

def profile_function():
    """Profile a function to identify performance bottlenecks"""
    
    def process_large_dataset(size=10000):
        """A function with performance issues to profile"""
        data = list(range(size))
        
        # Inefficient way to process data
        result = []
        for i in range(size):
            # Inefficient: Appending to a list in a loop
            result.append(data[i] ** 2)
        
        # Inefficient way to filter data
        filtered = []
        for item in result:
            if item % 3 == 0:
                filtered.append(item)
        
        # Inefficient way to sum data
        total = 0
        for item in filtered:
            total += item
        
        return total
    
    # Profile the function
    pr = cProfile.Profile()
    pr.enable()
    
    # Run the function
    result = process_large_dataset()
    
    pr.disable()
    
    # Print the results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Print top 20 functions
    print(s.getvalue())
    
    return result


# Example 5: Line profiling (requires line_profiler package)
"""
Line profiling with line_profiler package:

1. Install the package:
   pip install line_profiler

2. Add @profile decorator to the function you want to profile.
   The @profile decorator is automatically recognized by line_profiler.

3. Run the kernprof tool:
   kernprof -l -v your_script.py
"""

# @profile  # Uncomment when using kernprof
def slow_function(size=10000):
    """A function to profile with line_profiler"""
    data = list(range(size))
    
    # Inefficient: Appending to a list in a loop
    result = []
    for i in range(size):
        result.append(data[i] ** 2)
    
    # Inefficient: Filtering with a loop
    filtered = []
    for item in result:
        if item % 3 == 0:
            filtered.append(item)
    
    # Inefficient: Summing with a loop
    total = 0
    for item in filtered:
        total += item
    
    return total


# Example 6: Memory profiling (requires memory_profiler package)
"""
Memory profiling with memory_profiler package:

1. Install the package:
   pip install memory_profiler

2. Add @profile decorator to the function you want to profile.
   The @profile decorator comes from memory_profiler.

3. Run the script with memory profiling:
   python -m memory_profiler your_script.py
"""

# Uncomment when using memory_profiler
# from memory_profiler import profile

# @profile
def memory_intensive_function(size=10000):
    """A function with memory issues to profile"""
    # Create a large list
    data = list(range(size))
    
    # Create multiple copies of the data
    copies = [data.copy() for _ in range(10)]
    
    # Create a large dictionary
    large_dict = {i: i**2 for i in range(size)}
    
    # Process the data
    result = []
    for d in copies:
        for item in d:
            if item in large_dict:
                result.append(large_dict[item])
    
    return sum(result)


# Example 7: Performance Optimization
import numpy as np
import time

def compare_performance():
    """Compare performance of different implementations"""
    size = 10000000  # 10 million elements
    
    # Initialize data
    data = list(range(size))
    
    # Method 1: Inefficient list operations
    start_time = time.time()
    
    result = []
    for i in range(len(data)):
        result.append(data[i] ** 2)
    
    filtered = []
    for item in result:
        if item % 3 == 0:
            filtered.append(item)
        total =
        def compare_performance():
    """Compare performance of different implementations"""
    size = 10000000  # 10 million elements
    
    # Initialize data
    data = list(range(size))
    
    # Method 1: Inefficient list operations
    start_time = time.time()
    
    result = []
    for i in range(len(data)):
        result.append(data[i] ** 2)
    
    filtered = []
    for item in result:
        if item % 3 == 0:
            filtered.append(item)
    
    total = sum(filtered)
    
    method1_time = time.time() - start_time
    print(f"Method 1 (inefficient loops): {method1_time:.2f} seconds")
    
    # Method 2: List comprehensions
    start_time = time.time()
    
    result = [x ** 2 for x in data]
    filtered = [x for x in result if x % 3 == 0]
    total = sum(filtered)
    
    method2_time = time.time() - start_time
    print(f"Method 2 (list comprehensions): {method2_time:.2f} seconds")
    
    # Method 3: NumPy vectorization
    start_time = time.time()
    
    data_array = np.array(data)
    result = data_array ** 2
    filtered = result[result % 3 == 0]
    total = np.sum(filtered)
    
    method3_time = time.time() - start_time
    print(f"Method 3 (NumPy vectorization): {method3_time:.2f} seconds")
    
    # Method 4: Combined optimizations
    start_time = time.time()
    
    # Calculate, filter, and sum in one step
    total = sum(x ** 2 for x in range(size) if (x ** 2) % 3 == 0)
    
    method4_time = time.time() - start_time
    print(f"Method 4 (generator expression): {method4_time:.2f} seconds")
    
    # Compare the methods
    print("\nPerformance comparison:")
    print(f"Method 2 is {method1_time / method2_time:.1f}x faster than Method 1")
    print(f"Method 3 is {method1_time / method3_time:.1f}x faster than Method 1")
    print(f"Method 4 is {method1_time / method4_time:.1f}x faster than Method 1")
    
    return {
        "inefficient_loops": method1_time,
        "list_comprehensions": method2_time,
        "numpy_vectorization": method3_time,
        "generator_expression": method4_time
    }


# Example 8: Benchmarking with pytest-benchmark
"""
Benchmarking with pytest-benchmark:

1. Install the package:
   pip install pytest-benchmark

2. Create benchmark tests
"""

# Example of a benchmark test (test_benchmarks.py)
def test_square_list_methods(benchmark):
    """Benchmark different methods of squaring a list of numbers"""
    data = list(range(10000))
    
    # Method 1: For loop with append
    def method1():
        result = []
        for x in data:
            result.append(x ** 2)
        return result
    
    # Method 2: List comprehension
    def method2():
        return [x ** 2 for x in data]
    
    # Method 3: Map with lambda
    def method3():
        return list(map(lambda x: x ** 2, data))
    
    # Method 4: NumPy vectorization
    def method4():
        return (np.array(data) ** 2).tolist()
    
    # Run the benchmark on one of the methods
    result = benchmark(method2)
    
    # You can also compare different methods
    # To run this test with comparison:
    # pytest test_benchmarks.py --benchmark-compare


# Example 9: Performance optimized code
def optimize_data_processing(data, threshold):
    """
    Process data efficiently
    
    Before optimization:
    - Used loops and conditionals inefficiently
    - Created intermediate data structures
    - Performed redundant operations
    
    After optimization:
    - Used NumPy for vectorized operations
    - Avoided unnecessary data copying
    - Eliminated redundant calculations
    """
    # Convert to NumPy array for vectorized operations
    data_array = np.array(data)
    
    # Calculate squares in one vectorized operation
    squares = data_array ** 2
    
    # Filter in one vectorized operation
    mask = squares > threshold
    filtered = squares[mask]
    
    # Calculate statistics efficiently
    total = np.sum(filtered)
    mean = np.mean(filtered)
    std = np.std(filtered)
    
    return {
        'filtered_data': filtered,
        'count': len(filtered),
        'total': total,
        'mean': mean,
        'std': std
    }


# Example 10: Using time decorators for profiling
def timing_decorator(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


@timing_decorator
def slow_calculation(size):
    """Example function to profile with timing decorator"""
    result = 0
    for i in range(size):
        result += i ** 2
    return result


# Example 11: Benchmarking different algorithms
def benchmark_sorting_algorithms():
    """Benchmark different sorting algorithms"""
    import random
    
    # Generate random data
    data_small = [random.randint(0, 1000) for _ in range(1000)]
    data_medium = [random.randint(0, 10000) for _ in range(10000)]
    data_large = [random.randint(0, 100000) for _ in range(100000)]
    
    datasets = {
        "small": data_small.copy(),
        "medium": data_medium.copy(),
        "large": data_large.copy()
    }
    
    # Sort in-place algorithms
    def quick_sort(data):
        data_copy = data.copy()
        data_copy.sort()
        return data_copy
    
    def merge_sort(data):
        import heapq
        data_copy = data.copy()
        return list(heapq.merge(data_copy[:len(data_copy)//2], data_copy[len(data_copy)//2:]))
    
    def bubble_sort(data):
        data_copy = data.copy()
        n = len(data_copy)
        for i in range(n):
            for j in range(0, n - i - 1):
                if data_copy[j] > data_copy[j + 1]:
                    data_copy[j], data_copy[j + 1] = data_copy[j + 1], data_copy[j]
        return data_copy
    
    # Only use bubble sort for small dataset due to O(n²) complexity
    algorithms = {
        "quick_sort": quick_sort,
        "merge_sort": merge_sort
    }
    
    results = {}
    
    for name, dataset in datasets.items():
        results[name] = {}
        
        for algo_name, algorithm in algorithms.items():
            start_time = time.time()
            algorithm(dataset)
            elapsed = time.time() - start_time
            results[name][algo_name] = elapsed
        
        # Only run bubble sort on small dataset
        if name == "small":
            start_time = time.time()
            bubble_sort(dataset)
            elapsed = time.time() - start_time
            results[name]["bubble_sort"] = elapsed
    
    # Print results
    print("Sorting Algorithm Benchmarks (time in seconds):")
    for dataset_name, algos in results.items():
        print(f"\nDataset: {dataset_name}")
        for algo_name, elapsed in algos.items():
            print(f"  {algo_name}: {elapsed:.6f}")
    
    return results


# Example 12: Memory profiling visualization with memory_profiler
"""
To visualize memory usage over time:

1. Install memory_profiler and matplotlib:
   pip install memory_profiler matplotlib

2. Run the script with memory profiling:
   python -m memory_profiler your_script.py

3. Generate a plot:
   mprof run --python your_script.py
   mprof plot

This will create a plot showing memory usage over time.
"""

# Example 13: Using %timeit in IPython/Jupyter
"""
In IPython or Jupyter Notebook, you can use %timeit to quickly benchmark code:

%timeit [x**2 for x in range(1000)]
%timeit list(map(lambda x: x**2, range(1000)))
%timeit np.array(range(1000))**2

This will run the code multiple times and report the average execution time.
"""

# Run the examples if script is executed directly
if __name__ == "__main__":
    print("\n--- Debugging Examples ---")
    print("\nExample 1: Using breakpoints")
    # Uncomment to run:
    # debug_with_breakpoint()
    
    print("\nExample 2: Post-mortem debugging")
    # Uncomment to run:
    # debug_with_postmortem()
    
    print("\nExample 3: Using a debug context manager")
    # Uncomment to run:
    # debug_with_context_manager()
    
    print("\nExample 4: Using print debugging")
    # debug_with_print()
    
    print("\nExample 5: Using logging for debugging")
    # debug_with_logging()
    
    print("\n--- Profiling Examples ---")
    print("\nExample 6: CPU profiling with cProfile")
    # profile_function()
    
    print("\nExample 7: Performance optimization comparison")
    compare_performance()
    
    print("\nExample 10: Using timing decorators")
    slow_calculation(10000)
    
    print("\nExample 11: Benchmarking sorting algorithms")
    # benchmark_sorting_algorithms()
