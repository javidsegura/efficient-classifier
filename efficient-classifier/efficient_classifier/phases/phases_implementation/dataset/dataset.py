import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates


import seaborn as sns
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from imblearn.over_sampling import SMOTENC
from boruta import BorutaPy
from statsmodels.stats.outliers_influence import variance_inflation_factor


from efficient_classifier.phases.phases_implementation.dataset.split.strategies.noTimeSeries import NoTimeSeries
from efficient_classifier.phases.phases_implementation.dataset.split.strategies.timeSeries import TimeSeries


# Global variables
RANDOM_STATE = 88

class Dataset:
    """ Created dataframe, provides info, splits and encodes"""
    def __init__(self, dataset_path: str, model_task: str, variables: dict, random_state: int = RANDOM_STATE) -> None:
      """
      Creates a dataframe from a csv file

      Parameters
      ----------
      path : str
          The path to the dataframe
      problem_type : str
          The type of problem to solve (e.g: classification, regression)
      random_state : int
          The random state to use
      """ 
      assert model_task in ["classification_timeSeries", "regression_timeSeries", "classification", "regression"], "The model task must be either classification or regression"
      self.df = pd.read_csv(dataset_path)
      self.variables = variables
      self.random_state = random_state

      splitted_type = model_task.split("_")
      self.modelTask = splitted_type[0]
      self.isTimeSeries = len(splitted_type) > 1 and (splitted_type[-1] == "timeSeries")

      self.split = create_split_strategy(self, self.isTimeSeries)

    def eliminate_variables(self, variables_to_eliminate: list[str], after_split: bool = False):
      if after_split:
        self.X_train.drop(columns=variables_to_eliminate, inplace=True)
        self.X_val.drop(columns=variables_to_eliminate, inplace=True)
        self.X_test.drop(columns=variables_to_eliminate, inplace=True)
      else:
        self.df.drop(columns=variables_to_eliminate, inplace=True)


def create_split_strategy(dataset, is_time_series: bool = False):
    """
    Factory method to create the appropriate split strategy based on the dataset type.
    
    Parameters
    ----------
    dataset : Dataset
        The dataset to split
    is_time_series : bool
        Whether the dataset is a time series
        
    Returns
    -------
    Split
        The appropriate split strategy
    """
    if is_time_series:
        return TimeSeries(dataset)
    return NoTimeSeries(dataset) 