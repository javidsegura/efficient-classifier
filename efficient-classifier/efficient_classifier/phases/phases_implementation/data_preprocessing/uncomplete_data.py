import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
from efficient_classifier.phases.phases_implementation.dataset.dataset import Dataset
from efficient_classifier.utils.miscellaneous.save_or_store_plot import save_or_store_plot


class UncompleteData:
    def __init__(self, dataset: Dataset) -> None:
      self.dataset = dataset
    
    def analyze_duplicates(self, save_plots: bool = False, save_path: str = None) -> str:
        """Report and optionally visualise duplicate rows.

        Parameters
        ----------
        save_plots : bool, default=False
            If *True*, a barplot of duplicate counts per column is displayed.
        save_path : str
            The path to save the plot.

        Returns
        -------
        str
            Diagnostic string with the number of duplicate rows found.
        """

        # --- Input validation ---
        if not isinstance(save_plots, bool):
            raise TypeError("Parameter 'save_plots' must be a boolean.")

        # --- Dataset structure check ---
        if not hasattr(self.dataset, "df"):
            raise AttributeError("The dataset does not contain an attribute named 'df'.")
        if not hasattr(self.dataset.df, "duplicated"):
            raise TypeError("self.dataset.df must be a pandas DataFrame.")

        # --- Duplicate analysis ---
        try:
            duplicates = self.dataset.df.duplicated()
            duplicates_sum = duplicates.sum()
        except Exception as e:
            raise RuntimeError(f"Error checking for duplicates: {e}")

        # --- Plotting if requested ---
        if save_plots:
            if duplicates_sum > 0:
                try:
                    duplicates_by_column = self.dataset.df[duplicates].count()
                    feature_names = [f'{i+1}' for i in range(len(duplicates_by_column))]

                    fig, ax = plt.subplots(figsize=(15, 4))
                    sns.barplot(x=feature_names, y=duplicates_by_column.values)
                    plt.title("Number of Duplicates by Column")
                    plt.xlabel("Features")
                    plt.ylabel("Number of Duplicates")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    save_or_store_plot(fig, save_plots, save_path + "/uncomplete_data/duplicates", "duplicates_by_column.png")
                except Exception as e:
                    raise RuntimeError(f"An error occurred while plotting: {e}")
            else:
                print("No duplicates found in the dataset, no need to plot")
        else:
            if duplicates_sum == 0:
                print("No duplicates found in the dataset")

        return f"There are {duplicates_sum} duplicates in the dataset"
      
    def remove_duplicates(self) -> str:
        """
        Removes duplicates from the dataset

        Returns
        -------
        str
            Message indicating the number of duplicates removed
        """

        # --- Check that dataset has a DataFrame ---
        if not hasattr(self.dataset, "df"):
            raise AttributeError("The dataset does not contain an attribute named 'df'.")
        if not hasattr(self.dataset.df, "duplicated"):
            raise TypeError("self.dataset.df must be a pandas DataFrame.")

        try:
            duplicates = self.dataset.df.duplicated()
            duplicates_sum = duplicates.sum()
        except Exception as e:
            raise RuntimeError(f"An error occurred while checking for duplicates: {e}")

        if duplicates_sum > 0:
            try:
                print(f"Dataset duplicates:\n{self.dataset.df[duplicates]}")
                print(f"There are {duplicates_sum} duplicates in the dataset")
                self.dataset.df.drop_duplicates(inplace=True)
            except Exception as e:
                raise RuntimeError(f"An error occurred while removing duplicates: {e}")
            return "Successfully removed duplicates from the dataset"
        else:
            return "No duplicates found in the dataset"
 
    def get_missing_values(self, placeholders: list[str] | None = None, save_plots: bool = False, save_path: str = None):
        """
        Return the subset of rows that contain *any* missing value.

        Parameters
        ----------
        placeholders : list[str] | None
            Additional strings that should be considered *NA* (e.g., "N/A", "-1").
        save_plots : bool, default=False
            When *True*, show a barplot of missing counts per column.
        save_path : str
            The path to save the plot.

        Returns
        -------
        pandas.DataFrame | None
            Rows that include at least one missing value or *None* if the dataset is complete.
        """

        # --- Validation ---
        if not hasattr(self.dataset, "df"):
            raise AttributeError("The dataset does not contain an attribute named 'df'.")
        if not hasattr(self.dataset.df, "isnull"):
            raise TypeError("self.dataset.df must be a pandas DataFrame.")
        if not isinstance(save_plots, bool):
            raise TypeError("Parameter 'plot' must be a boolean.")

        try:
            # Count native NaNs
            missing_values_sum = self.dataset.df.isnull().sum().sum()

            # Include custom placeholders
            if placeholders:
                for placeholder in placeholders:
                    missing_values_sum += (self.dataset.df == placeholder).sum().sum()

            if missing_values_sum > 0:
                print(f"Dataset contains {missing_values_sum} missing values")

                # Identify rows with missing values or placeholders
                condition = self.dataset.df.isnull().any(axis=1)
                if placeholders:
                    for placeholder in placeholders:
                        condition |= self.dataset.df.eq(placeholder).any(axis=1)

                rows_with_missing = self.dataset.df[condition]
                print(f"Rows with missing values:\n{rows_with_missing}")

                if save_plots:
                    try:
                        missing_values_by_column = self.dataset.df.isnull().sum()
                        fig, ax = plt.figure(figsize=(15, 4))
                        sns.barplot(x=self.dataset.df.columns, y=missing_values_by_column.values)
                        plt.title("Missing Values by Column")
                        plt.xlabel("Features")
                        plt.ylabel("Number of Missing Values")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        save_or_store_plot(fig, save_plots, save_path + "/uncomplete_data/missing_values", "missing_values_by_column.png")
                    except Exception as e:
                        raise RuntimeError(f"An error occurred while plotting: {e}")

                return rows_with_missing

            else:
                msg = "No missing values found in the dataset"
                print(msg if not save_plots else msg + ", no need to plot")
                return msg

        except Exception as e:
            raise RuntimeError(f"An error occurred while analyzing missing values: {e}")

      
   