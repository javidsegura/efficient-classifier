import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
from library.phases.dataset.dataset import Dataset
from library.phases.data_preprocessing.uncomplete_data import UncompleteData
from library.phases.data_preprocessing.class_imbalance import ClassImbalance
from library.phases.data_preprocessing.feature_scaling import FeatureScaling
from library.phases.data_preprocessing.outliers_bounds import OutliersBounds
import random

class Preprocessing:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.uncomplete_data_obj = UncompleteData(dataset=self.dataset)
        self.class_imbalance_obj = ClassImbalance(dataset=self.dataset)
        self.feature_scaling_obj = FeatureScaling(dataset=self.dataset)
        self.outliers_bounds_obj = OutliersBounds(dataset=self.dataset)

    def delete_columns(self, columnsToDelete: list[str]) -> str:
      """ 
      Deletes the columns in the dataset

      Parameters:
      -----------
      columnsToDelete : list[str]
          The columns to delete

      Returns:
      --------
      str
          Message indicating the number of columns deleted
      """

      # Validate input type
      if not isinstance(columnsToDelete, list) or not all(isinstance(col, str) for col in columnsToDelete):
          raise TypeError("columnsToDelete must be a list of strings.")

      # Validate dataset attributes
      for attr in ['X_train', 'X_val', 'X_test']:
          if not hasattr(self.dataset, attr):
              raise AttributeError(f"The dataset is missing the attribute '{attr}'.")

      # Check that all columns exist in all datasets
      missing_cols = {
          attr: [col for col in columnsToDelete if col not in getattr(self.dataset, attr).columns]
          for attr in ['X_train', 'X_val', 'X_test']
      }
      errors = [f"{attr} is missing columns: {cols}" for attr, cols in missing_cols.items() if cols]
      if errors:
          raise ValueError("Some columns to delete are missing:\n" + "\n".join(errors))

      # Try deleting the columns
      try:
          self.dataset.X_train.drop(columns=columnsToDelete, inplace=True)
          self.dataset.X_val.drop(columns=columnsToDelete, inplace=True)
          self.dataset.X_test.drop(columns=columnsToDelete, inplace=True)
      except Exception as e:
          raise RuntimeError(f"An error occurred while deleting columns: {e}")

      return (f"Successfully deleted {len(columnsToDelete)} columns. "
              "To check the results, run: baseline_pipeline.dataset.X_train.head()")


  
