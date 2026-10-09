import unittest
import pandas as pd
import matplotlib.pyplot as plt
class TestCases(unittest.TestCase):
    def test_normal_dataframe(self):
        df = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'value': ['A', 'B', 'A', 'B', 'A', 'B']
        })
        ax = task_func(df)
        self.assertIsInstance(ax, plt.Axes, "Should return an Axes object")
        self.assertEqual(len(ax.patches), 2, "Should have 2 bars for values 'A' and 'B'")
        self.assertEqual(ax.get_title(), "Value Distribution", "Incorrect title")
        plt.close()
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=['id', 'value'])
        ax = task_func(df)
        self.assertIsInstance(ax, plt.Axes, "Should handle empty DataFrame")
        self.assertEqual(len(ax.patches), 0, "Should have no bars for an empty DataFrame")
        plt.close()
    def test_numeric_values(self):
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [100, 200, 300]
        })
        ax = task_func(df)
        self.assertIsInstance(ax, plt.Axes, "Should handle numeric values in 'value' column")
        plt.close()
    
    def test_plot_attributes(self):
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [100, 200, 300]
        })
        ax = task_func(df)
        self.assertEqual(ax.get_title(), 'Value Distribution')
        self.assertEqual(ax.get_xlabel(), 'Value')
        self.assertEqual(ax.get_ylabel(), 'Count')
        plt.close()
    
    def test_plot_point(self):
        df = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'value': ['A', 'B', 'A', 'B']
        })
        ax = task_func(df)
        # Get the actual value counts from the DataFrame
        actual_value_counts = df['value'].value_counts()
        # Get the patches from the bar plot
        patches = ax.patches
        # Ensure that each patch (bar) has the correct height (count)
        for i, patch in enumerate(patches):
            # The height of each bar should match the count of its corresponding value
            expected_height = actual_value_counts.iloc[i]
            self.assertAlmostEqual(patch.get_height(), expected_height, delta=0.1, msg=f"Bar {i+1} does not have the correct height")
        plt.close()