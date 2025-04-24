import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE


from library.phases.dataset.dataset import Dataset

class DataPreprocessing:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
    
    def remove_duplicates(self):
        """
        Removes duplicates from the dataset
        """
        duplicates = self.dataset.df.duplicated()
        duplicates_sum = duplicates.sum()
        if duplicates_sum > 0:
            print(f"Dataset duplicates: \n {self.df[duplicates]}")
            print(f"There are {duplicates_sum} duplicates in the dataset")
            self.dataset.df.drop_duplicates(inplace=True)
            print(f"Succesfully removed duplicates from the dataset")
        else:
            print("No duplicates found in the dataset")
    
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
    
    def scale_features(self, scaler: str = "minmax", columnsToScale: list[str] = []):
      assert len(columnsToScale) > 0, "Columns to scale must be provided"
      if scaler == "minmax":
        scaler = MinMaxScaler()
      elif scaler == "robust":
        scaler = RobustScaler()
      elif scaler == "standard":
        scaler = StandardScaler()
      else:
        raise ValueError(f"Invalid scaler: {scaler}")
      
      self.dataset.X_train[columnsToScale] = scaler.fit_transform(self.dataset.X_train[columnsToScale])
      self.dataset.X_val[columnsToScale] = scaler.transform(self.dataset.X_val[columnsToScale])
      self.dataset.X_test[columnsToScale] = scaler.transform(self.dataset.X_test[columnsToScale])

    def class_imbalance(self):
      """
      Checks if the dataset is imbalanced and returns the imbalance ratio
      
      Returns:
      --------
      str
        Message indicating the number of columns deleted
      """
      self.imbalance_ratio = self.dataset.y_train.value_counts().min() / self.dataset.y_train.value_counts().max()
      
      smote = SMOTE(random_state=42)
      self.dataset.X_train, self.dataset.y_train = smote.fit_resample(self.dataset.X_train, self.dataset.y_train)
      return f"Succesfully balanced the classes in the dataset. There was a {self.imbalance_ratio} to 1 ratio before balancing, now it is 1 to 1"

