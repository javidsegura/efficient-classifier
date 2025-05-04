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
      self.dataset.X_train.drop(columns=columnsToDelete, inplace=True)
      self.dataset.X_val.drop(columns=columnsToDelete, inplace=True)
      self.dataset.X_test.drop(columns=columnsToDelete, inplace=True)
      return f"Succesfully deleted {len(columnsToDelete)} columns, to check the results run: \n baseline_pipeline.dataset.X_train.head()"
    
    

  
