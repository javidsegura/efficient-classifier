import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from library.phases.dataset.dataset import Dataset

class Preprocessing:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
    
    def analyze_duplicates(self, plot: bool = False):
        """
        Analyzes the duplicates in the dataset
        
        Parameters:
        -----------
        plot : bool
            Whether to plot the duplicates
            
        Returns:
        --------
        None
        """
        duplicates = self.dataset.df.duplicated()
        duplicates_sum = duplicates.sum()
        if duplicates_sum > 0:
            if plot:
                # Count duplicates per column
                duplicates_by_column = self.dataset.df[duplicates].count()
            
                # Create generic feature names
                feature_names = [f'{i+1}' for i in range(len(duplicates_by_column))]
                
                # Create barplot
                plt.figure(figsize=(15, 4))
                sns.barplot(x=feature_names, y=duplicates_by_column.values)
                plt.title("Number of Duplicates by Column")
                plt.xlabel("Features")
                plt.ylabel("Number of Duplicates")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()
        else:
            if plot:
                print("No duplicates found in the dataset, no need to plot")
            else:
                print("No duplicates found in the dataset")
        return f"There are {duplicates_sum} duplicates in the dataset"
       
    def remove_duplicates(self):
        """
        Removes duplicates from the dataset
        
        Returns:
        --------
        str
            Message indicating the number of duplicates removed
        """
        duplicates = self.dataset.df.duplicated()
        duplicates_sum = duplicates.sum()
        if duplicates_sum > 0:
            print(f"Dataset duplicates: \n {self.df[duplicates]}")
            print(f"There are {duplicates_sum} duplicates in the dataset")
            self.dataset.df.drop_duplicates(inplace=True)
            return f"Succesfully removed duplicates from the dataset"
        else:
            return "No duplicates found in the dataset"
    
    def get_missing_values(self, placeholders: list, plot: bool = False):
        """
        Gets the missing values in the dataset and returns rows with missing values if any
        
        Parameters:
        -----------
        plot : bool
            Whether to plot the missing values
            
        Returns:
        --------
        None or pd.DataFrame with the rows with missing values
        """
        missing_values_sum = self.dataset.df.isnull().sum().sum()
        
        if placeholders:
          for placeholder in placeholders:
            missing_values_sum += (self.dataset.df == placeholder).sum().sum()
        
        if missing_values_sum > 0:
            print(f"Dataset contains {missing_values_sum} missing values")
            
            # Get rows that have at least one missing value
            rows_with_missing = self.dataset.df[self.dataset.df.isnull().any(axis=1)]
            print(f"Rows with missing values:\n{rows_with_missing}")
            
            if plot:
                # Plot missing values
                missing_values_by_column = self.dataset.df.isnull().sum()
                feature_names = [f'{i+1}' for i in range(len(missing_values_by_column))]
                plt.figure(figsize=(15, 4))
                sns.barplot(x=feature_names, y=missing_values_by_column.values)
                plt.title("Missing Values by Column")
                plt.xlabel("Features")
                plt.ylabel("Number of Missing Values")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()
            return rows_with_missing
        else:
            if plot:
                print("No missing values found in the dataset, no need to plot")
            else:
                print("No missing values found in the dataset")
            return None
          
    def bound_checking(self, columnsToCheck: list[str] = [], bounds: list[tuple] = []):
      """
      Checks if the values are within the bounds of the dataset and removes them if less than 0.5% of total data.
      
      Parameters:
      -----------
      columnsToCheck : list[str]
          List of column names to check bounds for
      bounds : list[tuple]
          List of (min, max) tuples corresponding to each column in columnsToCheck
          
      Returns:
      --------
      dict
          Dictionary with column names as keys and DataFrames of out-of-bounds values as values
      """
      assert len(columnsToCheck) > 0, "Columns to check must be provided"
      assert len(bounds) > 0, "Bounds must be provided"
      assert len(columnsToCheck) == len(bounds), "Number of columns and bounds must match"
      
      out_of_bounds = {}
      
      for i, column in enumerate(columnsToCheck):
          print(f"\n--- {i + 1}. Checking column {column}")
          min_val, max_val = bounds[i]
          
          # Check if column exists in the dataset
          if column not in self.dataset.df.columns:
              print(f"Warning: Column '{column}' not found in dataset")
              continue
          
          # Find values outside the bounds
          out_of_range = self.dataset.df[(self.dataset.df[column] < min_val) | 
                                        (self.dataset.df[column] > max_val)]
          
          if len(out_of_range) > 0:
              percentage = len(out_of_range) / len(self.dataset.df) * 100
              out_of_bounds[column] = out_of_range
              print(f"Found {len(out_of_range)} values outside bounds [{min_val}, {max_val}]")
              print(f"Percentage: {percentage:.4f}% of data")
              
              if percentage < 0.5:
                  print(f"→ Less than 0.5%. Deleting these rows...")
                  self.dataset.df = self.dataset.df.drop(out_of_range.index)
                  self.dataset.df.reset_index(drop=True, inplace=True)
              else:
                  print(f"→ More than 0.5%. Keeping them for manual review.")
          else:
              print(f"All values in column '{column}' are within bounds [{min_val}, {max_val}]")
      
      return out_of_bounds
 
    def get_outliers_df(self, pipeline: str = "iqr", plot: bool = False, threshold: float = 1.5, columnsToCheck: list[str] = []):
        """
        Detects outliers, removes them from X_train, and returns a DataFrame with outlier statistics.

        Parameters:
        -----------
        pipeline: str
            To determine how to treat the outliers based on the pipeline
        plot : bool
            Whether to plot the outliers
        threshold : float
            Multiplier for the IQR to determine outliers
        columnsToCheck : list[str]
            Specific columns to check for outliers. If empty, all numerical columns are used.

        Returns:
        --------
        str
            Summary of the outlier detection operation 
        """
        outlier_rows = []
        only_numerical_features = self.dataset.X_train.select_dtypes(include=["number"]).columns
        outliers = {}

        for feature in only_numerical_features if not columnsToCheck else columnsToCheck:
            original_values = self.dataset.X_train[feature]
            original_values_size = len(original_values)
            IQR = original_values.quantile(0.75) - original_values.quantile(0.25)
            lower_bound = original_values.quantile(0.25) - threshold * IQR
            upper_bound = original_values.quantile(0.75) + threshold * IQR
            outlier_mask = (original_values < lower_bound) | (original_values > upper_bound)
            outliersDataset = original_values[outlier_mask]
            outliers_count = outlier_mask.sum()

            if outliers_count > 0:
                outliers[feature] = outliersDataset
                outlier_rows.append({
                    "feature": feature,
                    "outlierCount": outliers_count,
                    "percentageOfOutliers": outliers_count / original_values_size * 100,
                    "descriptiveStatistics": original_values.describe(),
                    "outliersValues": outliersDataset.values
                })

                if plot:
                    plt.title(f"Distribution of '{feature}'")
                    sns.histplot(original_values, kde=True)
                    plt.show()

                if pipeline == "iqr":
                  # Remove outliers from X_train
                  self.dataset.X_train = self.dataset.X_train[~outlier_mask]
                elif pipeline == "percentile":
                  # Clip outliers on 1st and 99th percentile
                  p1 = original_values.quantile(0.01)
                  p99 = original_values.quantile(0.99)

                  self.dataset.X_train[feature] = original_values.clip(lower=p1, upper=p99)
                else:
                  assert("Error: You must introduce a correct value for pipeline. Only 'iqr' and 'percentile' are accepted.")

        # Reset index after removing rows
        self.dataset.X_train.reset_index(drop=True, inplace=True)

        outlier_df = pd.DataFrame(outlier_rows)

        return f"There are {len(outlier_df)} features with outliers out of {len(only_numerical_features)} numerical features ({len(outlier_df) / len(only_numerical_features) * 100:.2f}%)"
   
    def scale_features(self, scaler: str, columnsToScale: list[str] = []):
      """
      Scales the features in the dataset
      
      Parameters:
      -----------
      scaler : str
        The scaler to use
      columnsToScale : list[str]
        The columns to scale
        
      Returns:
      --------
      str
        Message indicating the number of features scaled  
      """
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
      
      return f"Succesfully scaled {len(columnsToScale)} features, to check the results run: \n baseline_pipeline.dataset.X_train.head()"

    def delete_columns(self, columnsToDelete: list[str]):
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
    
    def class_imbalance(self, plot: bool = False):
        """
        Balances classes via SMOTE and optionally plots the distributions
        before and after resampling.

        Parameters:
        -----------
        plot : bool
            Whether to show barplots of class counts before/after SMOTE

        Returns:
        --------
        str
            Summary of the balancing operation
        """
        # 1. Record original counts
        counts_before = self.dataset.y_train.value_counts().sort_index()
        self.imbalance_ratio = counts_before.min() / counts_before.max()

        # 2. Optionally plot before
        if plot:
            plt.figure(figsize=(6, 4))
            sns.barplot(
                x=counts_before.index.astype(str),
                y=counts_before.values
            )
            plt.title(f"Before SMOTE (imbalance ratio {self.imbalance_ratio:.2f}:1)")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

        # 3. Apply SMOTE
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(
            self.dataset.X_train, 
            self.dataset.y_train
        )
        self.dataset.X_train, self.dataset.y_train = X_res, y_res

        # 4. Record new counts and plot
        counts_after = self.dataset.y_train.value_counts().sort_index()
        if plot:
            plt.figure(figsize=(6, 4))
            sns.barplot(
                x=counts_after.index.astype(str),
                y=counts_after.values
            )
            plt.title("After SMOTE (balanced 1:1)")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

        return (
            f"Successfully balanced classes via SMOTE. "
            f"Started with a {self.imbalance_ratio:.2f}:1 ratio; now 1:1."
        )
        
  