"""
Basic tests for NYC Motor Vehicle Collisions Analysis Dashboard.

This module contains simple tests to validate the application functionality.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    validate_coordinates, validate_date, clean_street_name,
    calculate_severity_score, get_time_period_label, get_season,
    format_number, get_data_summary_stats, filter_data_by_criteria
)
from config import DATA_CONFIG, VALIDATION_CONFIG

class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = pd.DataFrame({
            'date/time': pd.to_datetime(['2023-01-01 12:00:00', '2023-01-02 15:30:00']),
            'latitude': [40.7128, 40.7589],
            'longitude': [-74.0060, -73.9851],
            'on_street_name': ['Broadway', '5th Ave'],
            'injured_persons': [2, 1],
            'injured_pedestrians': [1, 0],
            'injured_cyclists': [0, 1],
            'injured_motorists': [1, 0],
            'killed_persons': [0, 0],
            'killed_pedestrians': [0, 0],
            'killed_cyclists': [0, 0],
            'killed_motorists': [0, 0],
            'total_injured': [2, 1],
            'total_killed': [0, 0],
            'hour': [12, 15],
            'day_of_week': ['Sunday', 'Monday']
        })
    
    def test_validate_coordinates(self):
        """Test coordinate validation."""
        # Valid NYC coordinates
        self.assertTrue(validate_coordinates(40.7128, -74.0060))
        self.assertTrue(validate_coordinates(40.7589, -73.9851))
        
        # Invalid coordinates
        self.assertFalse(validate_coordinates(0, 0))  # Outside NYC
        self.assertFalse(validate_coordinates(91, 0))  # Invalid latitude
        self.assertFalse(validate_coordinates(0, 181))  # Invalid longitude
        self.assertFalse(validate_coordinates(None, None))  # None values
    
    def test_validate_date(self):
        """Test date validation."""
        # Valid dates
        self.assertTrue(validate_date('2023-01-01'))
        self.assertTrue(validate_date(datetime(2023, 1, 1)))
        
        # Invalid dates
        self.assertFalse(validate_date('2010-01-01'))  # Too old
        self.assertFalse(validate_date('2035-01-01'))  # Too future
        self.assertFalse(validate_date(None))  # None value
        self.assertFalse(validate_date('invalid-date'))  # Invalid format
    
    def test_clean_street_name(self):
        """Test street name cleaning."""
        # Test basic cleaning
        self.assertEqual(clean_street_name('broadway'), 'Broadway')
        self.assertEqual(clean_street_name('5TH AVE'), '5th Avenue')
        self.assertEqual(clean_street_name('  main st  '), 'Main Street')
        
        # Test abbreviations
        self.assertEqual(clean_street_name('Park Ave'), 'Park Avenue')
        self.assertEqual(clean_street_name('Central Blvd'), 'Central Boulevard')
        
        # Test edge cases
        self.assertEqual(clean_street_name(None), 'Unknown')
        self.assertEqual(clean_street_name(''), 'Unknown')
        self.assertEqual(clean_street_name(123), 'Unknown')
    
    def test_calculate_severity_score(self):
        """Test severity score calculation."""
        row = self.sample_data.iloc[0]
        score = calculate_severity_score(row)
        
        # Should be 2 (2 injuries, 0 fatalities)
        self.assertEqual(score, 2.0)
        
        # Test with fatalities
        row_with_fatalities = row.copy()
        row_with_fatalities['killed_pedestrians'] = 1
        score_with_fatalities = calculate_severity_score(row_with_fatalities)
        
        # Should be 12 (2 injuries + 1 fatality * 10)
        self.assertEqual(score_with_fatalities, 12.0)
    
    def test_get_time_period_label(self):
        """Test time period labeling."""
        self.assertEqual(get_time_period_label(8), 'Morning')
        self.assertEqual(get_time_period_label(14), 'Afternoon')
        self.assertEqual(get_time_period_label(19), 'Evening')
        self.assertEqual(get_time_period_label(23), 'Night')
        self.assertEqual(get_time_period_label(2), 'Night')
    
    def test_get_season(self):
        """Test season determination."""
        self.assertEqual(get_season(datetime(2023, 1, 1)), 'Winter')
        self.assertEqual(get_season(datetime(2023, 4, 1)), 'Spring')
        self.assertEqual(get_season(datetime(2023, 7, 1)), 'Summer')
        self.assertEqual(get_season(datetime(2023, 10, 1)), 'Fall')
    
    def test_format_number(self):
        """Test number formatting."""
        self.assertEqual(format_number(1234), '1.2K')
        self.assertEqual(format_number(1234567), '1.2M')
        self.assertEqual(format_number(123), '123')
        self.assertEqual(format_number(1234.56, 1), '1.2K')
    
    def test_get_data_summary_stats(self):
        """Test summary statistics calculation."""
        stats = get_data_summary_stats(self.sample_data)
        
        self.assertEqual(stats['total_collisions'], 2)
        self.assertEqual(stats['total_injured'], 3)
        self.assertEqual(stats['total_fatalities'], 0)
        self.assertEqual(stats['unique_streets'], 2)
        self.assertEqual(stats['most_common_hour'], 12)
    
    def test_filter_data_by_criteria(self):
        """Test data filtering."""
        # Test date range filter
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 1)
        filtered = filter_data_by_criteria(
            self.sample_data, 
            date_range=(start_date, end_date)
        )
        self.assertEqual(len(filtered), 1)
        
        # Test injury filter
        filtered = filter_data_by_criteria(
            self.sample_data, 
            min_injuries=2
        )
        self.assertEqual(len(filtered), 1)
        
        # Test time period filter
        filtered = filter_data_by_criteria(
            self.sample_data, 
            time_period='Afternoon'
        )
        self.assertEqual(len(filtered), 1)

class TestConfig(unittest.TestCase):
    """Test cases for configuration."""
    
    def test_data_config(self):
        """Test data configuration."""
        self.assertIn('default_file', DATA_CONFIG)
        self.assertIn('max_rows', DATA_CONFIG)
        self.assertIn('required_columns', DATA_CONFIG)
    
    def test_validation_config(self):
        """Test validation configuration."""
        self.assertIn('latitude_range', VALIDATION_CONFIG)
        self.assertIn('longitude_range', VALIDATION_CONFIG)
        self.assertIn('nyc_bounds', VALIDATION_CONFIG)

def run_tests():
    """Run all tests."""
    print("Running NYC Collisions Analysis Dashboard tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestUtils))
    test_suite.addTest(unittest.makeSuite(TestConfig))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)