# data_processor.py
"""
Example data processing functions for unit testing demonstration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by converting to lowercase, replacing spaces with underscores,
    and removing special characters.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned column names
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


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Optional list of columns to consider when identifying duplicates
        
    Returns:
        DataFrame with duplicates removed
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Remove duplicates
    result = result.drop_duplicates(subset=subset)
    
    return result


def fill_missing_values(
    df: pd.DataFrame, 
    strategies: Dict[str, str]
) -> pd.DataFrame:
    """
    Fill missing values in specified columns using different strategies.
    
    Args:
        df: Input DataFrame
        strategies: Dictionary mapping column names to filling strategies
                    ('mean', 'median', 'mode', 'zero', 'empty_string', or a specific value)
        
    Returns:
        DataFrame with missing values filled
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if not isinstance(strategies, dict):
        raise TypeError("Strategies must be a dictionary")
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    for column, strategy in strategies.items():
        if column not in result.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if strategy == 'mean':
            if not pd.api.types.is_numeric_dtype(result[column]):
                raise ValueError(f"Column '{column}' must be numeric for 'mean' strategy")
            result[column] = result[column].fillna(result[column].mean())
            
        elif strategy == 'median':
            if not pd.api.types.is_numeric_dtype(result[column]):
                raise ValueError(f"Column '{column}' must be numeric for 'median' strategy")
            result[column] = result[column].fillna(result[column].median())
            
        elif strategy == 'mode':
            mode_value = result[column].mode()[0]
            result[column] = result[column].fillna(mode_value)
            
        elif strategy == 'zero':
            result[column] = result[column].fillna(0)
            
        elif strategy == 'empty_string':
            result[column] = result[column].fillna('')
            
        else:
            # Use the provided value directly
            result[column] = result[column].fillna(strategy)
    
    return result


def calculate_statistics(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """
    Calculate various statistics for a numeric column.
    
    Args:
        df: Input DataFrame
        column: Name of the column to analyze
        
    Returns:
        Dictionary containing statistics
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    # Handle empty DataFrames
    if len(df) == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'count': 0,
            'missing': 0
        }
    
    # Calculate statistics
    stats = {
        'mean': float(df[column].mean()),
        'median': float(df[column].median()),
        'std': float(df[column].std()),
        'min': float(df[column].min()),
        'max': float(df[column].max()),
        'count': int(df[column].count()),
        'missing': int(df[column].isna().sum())
    }
    
    return stats


def bin_numeric_column(
    df: pd.DataFrame, 
    column: str, 
    bins: Union[int, List], 
    labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Bin a numeric column into categories.
    
    Args:
        df: Input DataFrame
        column: Name of the column to bin
        bins: Number of bins or list of bin edges
        labels: Optional labels for the bins
        
    Returns:
        DataFrame with added binned column
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Create binned column name
    binned_column = f"{column}_binned"
    
    # Create bins
    result[binned_column] = pd.cut(
        result[column], 
        bins=bins, 
        labels=labels, 
        include_lowest=True
    )
    
    return result


def identify_outliers(
    df: pd.DataFrame, 
    column: str, 
    method: str = 'iqr', 
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Identify outliers in a numeric column.
    
    Args:
        df: Input DataFrame
        column: Name of the column to analyze
        method: Method to use for outlier detection ('iqr' or 'zscore')
        threshold: Threshold for outlier detection (1.5 for IQR, 3 for zscore)
        
    Returns:
        DataFrame with added boolean column indicating outliers
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Create outlier column name
    outlier_column = f"{column}_outlier"
    
    # Identify outliers
    if method == 'iqr':
        q1 = result[column].quantile(0.25)
        q3 = result[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        result[outlier_column] = (result[column] < lower_bound) | (result[column] > upper_bound)
        
    elif method == 'zscore':
        mean = result[column].mean()
        std = result[column].std()
        result[outlier_column] = (abs(result[column] - mean) / std) > threshold
        
    else:
        raise ValueError(f"Invalid method: {method}. Use 'iqr' or 'zscore'.")
    
    return result


# tests/test_data_processor.py
"""
Unit tests for data_processor.py functions
"""

import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from data_processor import (
    clean_column_names,
    remove_duplicates,
    fill_missing_values,
    calculate_statistics,
    bin_numeric_column,
    identify_outliers
)


class TestCleanColumnNames:
    """Tests for the clean_column_names function"""
    
    def test_lowercase_conversion(self):
        """Test that column names are converted to lowercase"""
        # Arrange
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        
        # Act
        result = clean_column_names(df)
        
        # Assert
        assert list(result.columns) == ['a', 'b']
    
    def test_space_replacement(self):
        """Test that spaces in column names are replaced with underscores"""
        # Arrange
        df = pd.DataFrame({'Column A': [1, 2], 'Column B': [3, 4]})
        
        # Act
        result = clean_column_names(df)
        
        # Assert
        assert list(result.columns) == ['column_a', 'column_b']
    
    def test_special_character_removal(self):
        """Test that special characters are handled correctly"""
        # Arrange
        df = pd.DataFrame({
            'Column-A': [1, 2],
            'Column.B': [3, 4],
            'Column/C': [5, 6],
            'Column(D)': [7, 8]
        })
        
        # Act
        result = clean_column_names(df)
        
        # Assert
        assert list(result.columns) == ['column_a', 'column_b', 'column_c', 'columnd']
    
    def test_non_dataframe_input(self):
        """Test that TypeError is raised for non-DataFrame input"""
        # Arrange
        not_a_df = [1, 2, 3]
        
        # Act & Assert
        with pytest.raises(TypeError):
            clean_column_names(not_a_df)
    
    def test_numeric_column_names(self):
        """Test that numeric column names are handled correctly"""
        # Arrange
        df = pd.DataFrame({0: [1, 2], 1: [3, 4]})
        
        # Act
        result = clean_column_names(df)
        
        # Assert
        assert list(result.columns) == ['0', '1']
    
    def test_original_df_unchanged(self):
        """Test that the original DataFrame is not modified"""
        # Arrange
        df = pd.DataFrame({'A B': [1, 2], 'C-D': [3, 4]})
        original_columns = list(df.columns)
        
        # Act
        result = clean_column_names(df)
        
        # Assert
        assert list(df.columns) == original_columns
        assert list(result.columns) == ['a_b', 'c_d']


class TestRemoveDuplicates:
    """Tests for the remove_duplicates function"""
    
    def test_remove_exact_duplicates(self):
        """Test removing exact duplicate rows"""
        # Arrange
        df = pd.DataFrame({
            'a': [1, 2, 1, 3],
            'b': [4, 5, 4, 6]
        })
        
        # Act
        result = remove_duplicates(df)
        
        # Assert
        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        assert_frame_equal(result, expected)
    
    def test_subset_duplicates(self):
        """Test removing duplicates based on a subset of columns"""
        # Arrange
        df = pd.DataFrame({
            'a': [1, 1, 2, 2],
            'b': [4, 5, 6, 7]
        })
        
        # Act
        result = remove_duplicates(df, subset=['a'])
        
        # Assert
        expected = pd.DataFrame({
            'a': [1, 2],
            'b': [4, 6]
        }, index=[0, 2])  # Note: Original indices are preserved
        assert_frame_equal(result, expected)
    
    def test_no_duplicates(self):
        """Test when there are no duplicates"""
        # Arrange
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        
        # Act
        result = remove_duplicates(df)
        
        # Assert
        assert_frame_equal(result, df)
    
    def test_empty_dataframe(self):
        """Test with an empty DataFrame"""
        # Arrange
        df = pd.DataFrame({
            'a': [],
            'b': []
        })
        
        # Act
        result = remove_duplicates(df)
        
        # Assert
        assert_frame_equal(result, df)
    
    def test_non_dataframe_input(self):
        """Test that TypeError is raised for non-DataFrame input"""
        # Arrange
        not_a_df = [1, 2, 3]
        
        # Act & Assert
        with pytest.raises(TypeError):
            remove_duplicates(not_a_df)


class TestFillMissingValues:
    """Tests for the fill_missing_values function"""
    
    def test_fill_with_mean(self):
        """Test filling missing values with the mean"""
        # Arrange
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [5, np.nan, 7, 8]
        })
        strategies = {'a': 'mean', 'b': 'mean'}
        
        # Act
        result = fill_missing_values(df, strategies)
        
        # Assert
        expected = pd.DataFrame({
            'a': [1.0, 2.0, 2.0, 4.0],  # mean of [1, 2, 4] is 2.33... rounded to 2.0 for assertion
            'b': [5.0, 6.0, 7.0, 8.0]   # mean of [5, 7, 8] is 6.67... rounded to 6.0 for assertion
        })
        pd.testing.assert_frame_equal(result, expected, check_dtype=True, rtol=1e-2)
    
    def test_fill_with_median(self):
        """Test filling missing values with the median"""
        # Arrange
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 10],  # median is 2
            'b': [5, np.nan, 7, 15]   # median is 7
        })
        strategies = {'a': 'median', 'b': 'median'}
        
        # Act
        result = fill_missing_values(df, strategies)
        
        # Assert
        expected = pd.DataFrame({
            'a': [1.0, 2.0, 2.0, 10.0],
            'b': [5.0, 7.0, 7.0, 15.0]
        })
        assert_frame_equal(result, expected)
    
    def test_fill_with_mode(self):
        """Test filling missing values with the mode"""
        # Arrange
        df = pd.DataFrame({
            'a': [1, 2, 2, np.nan],  # mode is 2
            'b': ['x', 'y', 'y', np.nan]  # mode is 'y'
        })
        strategies = {'a': 'mode', 'b': 'mode'}
        
        # Act
        result = fill_missing_values(df, strategies)
        
        # Assert
        expected = pd.DataFrame({
            'a': [1.0, 2.0, 2.0, 2.0],
            'b': ['x', 'y', 'y', 'y']
        })
        assert_frame_equal(result, expected)
    
    def test_fill_with_zero(self):
        """Test filling missing values with zero"""
        # Arrange
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': ['x', 'y', np.nan, 'z']
        })
        strategies = {'a': 'zero', 'b': 'zero'}
        
        # Act
        result = fill_missing_values(df, strategies)
        
        # Assert
        expected = pd.DataFrame({
            'a': [1.0, 2.0, 0.0, 4.0],
            'b': ['x', 'y', 0, 'z']
        })
        assert_frame_equal(result, expected)
    
    def test_fill_with_empty_string(self):
        """Test filling missing values with an empty string"""
        # Arrange
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': ['x', 'y', np.nan, 'z']
        })
        strategies = {'a': 'empty_string', 'b': 'empty_string'}
        
        # Act
        result = fill_missing_values(df, strategies)
        
        # Assert
        expected = pd.DataFrame({
            'a': [1.0, 2.0, '', 4.0],
            'b': ['x', 'y', '', 'z']
        })
        assert_frame_equal(result, expected)
    
    def test_fill_with_specific_value(self):
        """Test filling missing values with a specific value"""
        # Arrange
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': ['x', 'y', np.nan, 'z']
        })
        strategies = {'a': 999, 'b': 'MISSING'}
        
        # Act
        result = fill_missing_values(df, strategies)
        
        # Assert
        expected = pd.DataFrame({
            'a': [1.0, 2.0, 999.0, 4.0],
            'b': ['x', 'y', 'MISSING', 'z']
        })
        assert_frame_equal(result, expected)
    
    def test_column_not_found(self):
        """Test that ValueError is raised when a column is not found"""
        # Arrange
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [5, np.nan, 7, 8]
        })
        strategies = {'c': 'mean'}
        
        # Act & Assert
        with pytest.raises(ValueError):
            fill_missing_values(df, strategies)
    
    def test_non_numeric_with_numeric_strategy(self):
        """Test that ValueError is raised when using a numeric strategy on a non-numeric column"""
        # Arrange
        df = pd.DataFrame({
            'a': ['x', 'y', None, 'z']
        })
        strategies = {'a': 'mean'}
        
        # Act & Assert
        with pytest.raises(ValueError):
            fill_missing_values(df, strategies)
    
    def test_invalid_input_types(self):
        """Test that TypeError is raised for invalid input types"""
        # Arrange
        df = pd.DataFrame({'a': [1, 2, np.nan, 4]})
        not_a_dict = [1, 2, 3]
        
        # Act & Assert
        with pytest.raises(TypeError):
            fill_missing_values(df, not_a_dict)
        
        with pytest.raises(TypeError):
            fill_missing_values([1, 2, 3], {'a': 'mean'})


class TestCalculateStatistics:
    """Tests for the calculate_statistics function"""
    
    def test_basic_statistics(self):
        """Test calculation of basic statistics for a column"""
        # Arrange
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5]
        })
        
        # Act
        result = calculate_statistics(df, 'a')
        
        # Assert
        assert result['mean'] == 3.0
        assert result['median'] == 3.0
        assert result['min'] == 1.0
        assert result['max'] == 5.0
        assert result['count'] == 5
        assert result['missing'] == 0
    
    def test_with_missing_values(self):
        """Test statistics calculation with missing values"""
        # Arrange
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4, 5]
        })
        
        # Act
        result = calculate_statistics(df, 'a')
        
        # Assert
        assert result['mean'] == 3.0
        assert result['median'] == 3.0
        assert result['min'] == 1.0
        assert result['max'] == 5.0
        assert result['count'] == 4
        assert result['missing'] == 1
    
    def test_empty_dataframe(self):
        """Test statistics calculation with an empty DataFrame"""
        # Arrange
        df = pd.DataFrame({'a': []})
        
        # Act
        result = calculate_statistics(df, 'a')
        
        # Assert
        assert pd.isna(result['mean'])
        assert pd.isna(result['median'])
        assert pd.isna(result['std'])
        assert pd.isna(result['min'])
        assert pd.isna(result['max'])
        assert result['count'] == 0
        assert result['missing'] == 0
    
    def test_column_not_found(self):
        """Test that ValueError is raised when a column is not found"""
        # Arrange
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        # Act & Assert
        with pytest.raises(ValueError):
            calculate_statistics(df, 'b')
    
    def test_non_numeric_column(self):
        """Test that ValueError is raised for a non-numeric column"""
        # Arrange
        df = pd.DataFrame({'a': ['x', 'y', 'z']})
        
        # Act & Assert
        with pytest.raises(ValueError):
            calculate_statistics(df, 'a')
    
    def test_non_dataframe_input(self):
        """Test that TypeError is raised for non-DataFrame input"""
        # Arrange
        not_a_df = [1, 2, 3]
        
        # Act & Assert
        with pytest.raises(TypeError):
            calculate_statistics(not_a_df, 'a')


class TestBinNumericColumn:
    """Tests for the bin_numeric_column function"""
    
    def test_bin_with_number_of_bins(self):
        """Test binning with a specified number of bins"""
        # Arrange
        df = pd.DataFrame({
            'a': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        # Act
        result = bin_numeric_column(df, 'a', bins=5)
        
        # Assert
        # With 5 bins, the bin edges would be [10, 28, 46, 64, 82, 100]
        expected_bins = pd.cut(df['a'], bins=5, include_lowest=True)
        assert_series_equal(result['a_binned'], expected_bins)
    
    def test_bin_with_custom_edges(self):
        """Test binning with custom bin edges"""
        # Arrange
        df = pd.DataFrame({
            'a': [10,
            def test_bin_with_custom_edges(self):
        """Test binning with custom bin edges"""
        # Arrange
        df = pd.DataFrame({
            'a': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        custom_bins = [0, 25, 50, 75, 100]
        
        # Act
        result = bin_numeric_column(df, 'a', bins=custom_bins)
        
        # Assert
        expected_bins = pd.cut(df['a'], bins=custom_bins, include_lowest=True)
        assert_series_equal(result['a_binned'], expected_bins)
    
    def test_bin_with_labels(self):
        """Test binning with custom labels"""
        # Arrange
        df = pd.DataFrame({
            'a': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        custom_bins = [0, 25, 50, 75, 100]
        labels = ['Low', 'Medium', 'High', 'Very High']
        
        # Act
        result = bin_numeric_column(df, 'a', bins=custom_bins, labels=labels)
        
        # Assert
        expected_bins = pd.cut(df['a'], bins=custom_bins, labels=labels, include_lowest=True)
        assert_series_equal(result['a_binned'], expected_bins)
    
    def test_column_not_found(self):
        """Test that ValueError is raised when a column is not found"""
        # Arrange
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        # Act & Assert
        with pytest.raises(ValueError):
            bin_numeric_column(df, 'b', bins=3)
    
    def test_non_numeric_column(self):
        """Test that ValueError is raised for a non-numeric column"""
        # Arrange
        df = pd.DataFrame({'a': ['x', 'y', 'z']})
        
        # Act & Assert
        with pytest.raises(ValueError):
            bin_numeric_column(df, 'a', bins=3)
    
    def test_original_df_unchanged(self):
        """Test that the original DataFrame is not modified"""
        # Arrange
        df = pd.DataFrame({
            'a': [10, 20, 30, 40, 50]
        })
        original_columns = list(df.columns)
        
        # Act
        result = bin_numeric_column(df, 'a', bins=3)
        
        # Assert
        assert list(df.columns) == original_columns
        assert 'a_binned' in result.columns
        assert 'a_binned' not in df.columns


class TestIdentifyOutliers:
    """Tests for the identify_outliers function"""
    
    def test_outliers_iqr_method(self):
        """Test identifying outliers using the IQR method"""
        # Arrange
        df = pd.DataFrame({
            'a': [10, 20, 30, 40, 100]  # 100 is an outlier
        })
        
        # Act
        result = identify_outliers(df, 'a', method='iqr', threshold=1.5)
        
        # Assert
        expected_outliers = pd.Series([False, False, False, False, True], name='a_outlier')
        assert_series_equal(result['a_outlier'], expected_outliers)
    
    def test_outliers_zscore_method(self):
        """Test identifying outliers using the z-score method"""
        # Arrange
        df = pd.DataFrame({
            'a': [10, 20, 30, 40, 100]  # 100 is an outlier
        })
        
        # Act
        result = identify_outliers(df, 'a', method='zscore', threshold=2)
        
        # Assert
        expected_outliers = pd.Series([False, False, False, False, True], name='a_outlier')
        assert_series_equal(result['a_outlier'], expected_outliers)
    
    def test_no_outliers(self):
        """Test when there are no outliers"""
        # Arrange
        df = pd.DataFrame({
            'a': [10, 20, 30, 40, 50]  # No outliers
        })
        
        # Act
        result = identify_outliers(df, 'a', method='iqr')
        
        # Assert
        expected_outliers = pd.Series([False, False, False, False, False], name='a_outlier')
        assert_series_equal(result['a_outlier'], expected_outliers)
    
    def test_invalid_method(self):
        """Test that ValueError is raised for an invalid method"""
        # Arrange
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        
        # Act & Assert
        with pytest.raises(ValueError):
            identify_outliers(df, 'a', method='invalid_method')
    
    def test_column_not_found(self):
        """Test that ValueError is raised when a column is not found"""
        # Arrange
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        # Act & Assert
        with pytest.raises(ValueError):
            identify_outliers(df, 'b')
    
    def test_non_numeric_column(self):
        """Test that ValueError is raised for a non-numeric column"""
        # Arrange
        df = pd.DataFrame({'a': ['x', 'y', 'z']})
        
        # Act & Assert
        with pytest.raises(ValueError):
            identify_outliers(df, 'a')


# Example of Test-Driven Development: First, write failing tests

class TestNormalizeColumn:
    """Tests for a normalize_column function (not yet implemented)"""
    
    def test_min_max_normalization(self):
        """Test min-max normalization of a column"""
        # Arrange
        df = pd.DataFrame({
            'a': [10, 20, 30, 40, 50]
        })
        
        # Act
        result = normalize_column(df, 'a', method='min_max')
        
        # Assert
        expected = pd.DataFrame({
            'a': [10, 20, 30, 40, 50],
            'a_normalized': [0.0, 0.25, 0.5, 0.75, 1.0]
        })
        assert_frame_equal(result, expected)
    
    def test_zscore_normalization(self):
        """Test z-score normalization of a column"""
        # Arrange
        df = pd.DataFrame({
            'a': [10, 20, 30, 40, 50]
        })
        
        # Act
        result = normalize_column(df, 'a', method='zscore')
        
        # Assert
        # Mean = 30, Std = 15.81...
        normalized_values = (df['a'] - 30) / 15.81139
        expected = pd.DataFrame({
            'a': [10, 20, 30, 40, 50],
            'a_normalized': normalized_values
        })
        assert_frame_equal(result, expected, check_less_precise=3)
    
    def test_custom_range_normalization(self):
        """Test normalization to a custom range"""
        # Arrange
        df = pd.DataFrame({
            'a': [10, 20, 30, 40, 50]
        })
        
        # Act
        result = normalize_column(df, 'a', method='min_max', range_min=100, range_max=200)
        
        # Assert
        expected = pd.DataFrame({
            'a': [10, 20, 30, 40, 50],
            'a_normalized': [100, 125, 150, 175, 200]
        })
        assert_frame_equal(result, expected)
    
    def test_non_numeric_column(self):
        """Test that ValueError is raised for a non-numeric column"""
        # Arrange
        df = pd.DataFrame({'a': ['x', 'y', 'z']})
        
        # Act & Assert
        with pytest.raises(ValueError):
            normalize_column(df, 'a')


# Then, implement the function to make the tests pass

def normalize_column(
    df: pd.DataFrame, 
    column: str, 
    method: str = 'min_max',
    range_min: float = 0.0,
    range_max: float = 1.0
) -> pd.DataFrame:
    """
    Normalize a numeric column using different methods.
    
    Args:
        df: Input DataFrame
        column: Name of the column to normalize
        method: Normalization method ('min_max' or 'zscore')
        range_min: Minimum value for min-max normalization
        range_max: Maximum value for min-max normalization
        
    Returns:
        DataFrame with added normalized column
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Create normalized column name
    normalized_column = f"{column}_normalized"
    
    # Normalize based on method
    if method == 'min_max':
        min_val = df[column].min()
        max_val = df[column].max()
        
        if max_val == min_val:
            # Avoid division by zero
            result[normalized_column] = range_min
        else:
            # Min-max normalization formula
            normalized = (df[column] - min_val) / (max_val - min_val)
            # Scale to desired range
            result[normalized_column] = normalized * (range_max - range_min) + range_min
    
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        
        if std == 0:
            # Avoid division by zero
            result[normalized_column] = 0
        else:
            # Z-score normalization formula
            result[normalized_column] = (df[column] - mean) / std
    
    else:
        raise ValueError(f"Invalid method: {method}. Use 'min_max' or 'zscore'.")
    
    return result


# Example of using fixtures for test data

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        'numeric': [10, 20, 30, 40, 50],
        'categorical': ['A', 'B', 'C', 'A', 'B'],
        'with_nulls': [1.0, 2.0, np.nan, 4.0, 5.0],
        'constant': [7, 7, 7, 7, 7]
    })


def test_clean_column_names_with_fixture(sample_dataframe):
    """Test clean_column_names using a fixture"""
    # Act
    result = clean_column_names(sample_dataframe)
    
    # Assert
    assert list(result.columns) == ['numeric', 'categorical', 'with_nulls', 'constant']


def test_calculate_statistics_with_fixture(sample_dataframe):
    """Test calculate_statistics using a fixture"""
    # Act
    result = calculate_statistics(sample_dataframe, 'numeric')
    
    # Assert
    assert result['mean'] == 30.0
    assert result['median'] == 30.0
    assert result['min'] == 10.0
    assert result['max'] == 50.0


# Example of using mocks to test functions with external dependencies

def read_data_from_csv(file_path: str) -> pd.DataFrame:
    """Read data from a CSV file"""
    return pd.read_csv(file_path)


def process_customer_data(file_path: str) -> Dict[str, Any]:
    """
    Process customer data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dictionary containing processing results
    """
    # Read data
    df = read_data_from_csv(file_path)
    
    # Clean column names
    df = clean_column_names(df)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Fill missing values
    strategies = {
        'age': 'median',
        'income': 'mean',
        'email': 'empty_string'
    }
    df = fill_missing_values(df, strategies)
    
    # Calculate statistics
    age_stats = calculate_statistics(df, 'age')
    income_stats = calculate_statistics(df, 'income')
    
    # Create summary
    summary = {
        'record_count': len(df),
        'age_stats': age_stats,
        'income_stats': income_stats,
        'processed_data': df
    }
    
    return summary


def test_process_customer_data_with_mock():
    """Test the process_customer_data function using a mock for read_data_from_csv"""
    # Arrange
    # Create a mock DataFrame to be returned by the mocked function
    mock_df = pd.DataFrame({
        'Customer Id': [1, 2, 3, 3, 4],
        'Age': [25, 30, np.nan, 40, 50],
        'Income': [50000, np.nan, 70000, 80000, 90000],
        'Email': ['a@example.com', 'b@example.com', np.nan, 'c@example.com', 'd@example.com']
    })
    
    # Mock the read_data_from_csv function
    with pytest.monkeypatch.context() as monkeypatch:
        monkeypatch.setattr('__main__.read_data_from_csv', lambda file_path: mock_df)
        
        # Act
        result = process_customer_data('dummy_path.csv')
    
    # Assert
    assert result['record_count'] == 4  # After removing duplicates
    assert result['age_stats']['mean'] == pytest.approx(36.25)
    assert result['income_stats']['median'] == pytest.approx(75000.0)
    assert list(result['processed_data'].columns) == ['customer_id', 'age', 'income', 'email']


# Example of parameterized tests

@pytest.mark.parametrize(
    "input_df, column, method, threshold, expected_count",
    [
        # Test case 1: IQR method with clear outlier
        (pd.DataFrame({'a': [10, 20, 30, 40, 100]}), 'a', 'iqr', 1.5, 1),
        
        # Test case 2: Z-score method with clear outlier
        (pd.DataFrame({'a': [10, 20, 30, 40, 100]}), 'a', 'zscore', 2, 1),
        
        # Test case 3: No outliers with IQR method
        (pd.DataFrame({'a': [10, 20, 30, 40, 50]}), 'a', 'iqr', 1.5, 0),
        
        # Test case 4: No outliers with Z-score method
        (pd.DataFrame({'a': [10, 20, 30, 40, 50]}), 'a', 'zscore', 2, 0),
        
        # Test case 5: Multiple outliers
        (pd.DataFrame({'a': [10, 20, 30, 100, 200]}), 'a', 'iqr', 1.5, 2)
    ]
)
def test_identify_outliers_parameterized(input_df, column, method, threshold, expected_count):
    """Parameterized test for identify_outliers with various test cases"""
    # Act
    result = identify_outliers(input_df, column, method, threshold)
    
    # Assert
    assert result[f'{column}_outlier'].sum() == expected_count
