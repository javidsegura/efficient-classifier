import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
from library.phases.dataset.dataset import Dataset

class UncompleteData:
    def __init__(self, dataset: Dataset) -> None:
      self.dataset = dataset
    
    def analyze_duplicates(self, plot: bool = False):
        """Report and optionally visualise duplicate rows.

        Parameters
        ----------
        plot : bool, default=False
            If *True* a barplot of duplicate counts per column is displayed.

        Returns
        -------
        str
            Diagnostic string with the number of duplicate rows found.
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
    
    def get_missing_values(self, placeholders: list[str] | None = None, *, plot: bool = False):
        """Return the subset of rows that contain *any* missing value.

        Parameters
        ----------
        placeholders : list[str] | None
            Additional strings that should be considered *NA* (for example,
            "N/A", "missing", "-1", …).
        plot : bool, default=False
            When *True* show a barplot of missing counts per column.

        Returns
        -------
        pandas.DataFrame | None
            Rows that include at least one missing value or *None* if the
            dataset is complete.
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
      
   