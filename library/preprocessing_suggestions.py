import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from library.dataset import Dataset

class Preprocessing:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
    
    def get_missing_values(self):
        """
        Gets the missing values in the dataset and returns rows with missing values if any
        """
        missing_values_sum = self.dataset.df.isnull().sum().sum()
        
        if missing_values_sum > 0:
            print(f"Dataset contains {missing_values_sum} missing values")
            
            # Get rows that have at least one missing value
            rows_with_missing = self.dataset.df[self.dataset.df.isnull().any(axis=1)]
            print(f"Rows with missing values:\n{rows_with_missing}")
            
            return rows_with_missing
        else:
            print("No missing values found in the dataset")
            return None
        
    def get_outliers_df(self, plot: bool = False, threshold: float = 1.5, columnsToCheck: list[str] = []):
      outlier_df = pd.DataFrame(columns=["feature", "outlierCount","percentageOfOutliers", "descriptiveStatistics"])
      only_numerical_features = self.dataset.X_train.select_dtypes(include=["number"]).columns
      outliers = {}

      for feature in only_numerical_features if not columnsToCheck else columnsToCheck:
            original_values = self.dataset.X_train[feature].copy()
            original_values_size = len(original_values)
            IQR = self.dataset.X_train[feature].quantile(0.75) - self.dataset.X_train[feature].quantile(0.25)
            lower_bound = self.dataset.X_train[feature].quantile(0.25) - threshold * IQR
            upper_bound = self.dataset.X_train[feature].quantile(0.75) + threshold * IQR
            outliersDataset = self.dataset.X_train[feature][(self.dataset.X_train[feature] < lower_bound) | (self.dataset.X_train[feature] > upper_bound)]
            outliers_count = len(outliersDataset)
            if (outliers_count > 0):
                  outliers[feature] = outliersDataset
                  outlier_df = pd.concat([outlier_df, 
                                          pd.DataFrame({"feature": feature, 
                                                        "outlierCount": len(outliersDataset), 
                                                        "percentageOfOutliers": len(outliersDataset) / original_values_size * 100, 
                                                        "descriptiveStatistics": [self.dataset.X_train[feature].describe()],
                                                        "outliersValues": [outliersDataset.values]
                                                    })])
                  # Print distribution of feature
                  if plot:
                        plt.title(f"Distribution of '{feature}'")
                        sns.histplot(self.dataset.X_train[feature], kde=True)
                        plt.show()
      print(f"There are {len(outlier_df)} features with outliers out of {len(only_numerical_features)} numerical features ({len(outlier_df) / len(only_numerical_features) * 100}%)")
      return outlier_df, outliers

    