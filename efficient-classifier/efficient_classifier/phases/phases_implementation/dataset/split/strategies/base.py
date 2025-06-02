from abc import ABC, abstractmethod

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from efficient_classifier.utils.miscellaneous.save_or_store_plot import save_or_store_plot


class Split(ABC):
      def __init__(self, dataset) -> None:
            self.dataset = dataset
      @abstractmethod
      def split_data(self,
                     y_column: str,
                     otherColumnsToDrop: list[str] = [],
                     train_size: float = 0.8,
                     validation_size: float = 0.1,
                     test_size: float = 0.1,
                     plot_distribution: bool = True,
                     **kwargs
                     ):
            pass
      

      def __get_X_y__(self, y_column: str, otherColumnsToDrop: list[str] = []) -> tuple[pd.DataFrame, pd.Series]:
            """
            
            Splits the dataframe into features and target variable

            Parameters:
            ----------
            y_column: str
                  The name of the target column
            otherColumnsToDrop: list[str]
                  The names of the other columns to drop
            
            Returns:
            -------
            X: pd.DataFrame
                  The features
            y: pd.Series
                  The target variable
            
            """
            X = self.dataset.df.drop(columns=[y_column] + otherColumnsToDrop)
            y = self.dataset.df[y_column]
            return X, y
      
      def plot_per_set_distribution(self, features: list[str], max_plots: int = 5, save_plots: bool = False, save_path: str = None):
            """
            Plots the distribution of the features for the training, validation and test sets. This is going to be meaningful for checking the similarity in statistical distributions between the sets.
            Note: for high-dimesionality dataset this is going to be computationally expensive.

            Parameters:
            ----------
            features: list[str]
                  The names of the features to plot
            save_plots: bool
                  Whether to save the plots
            save_path: str
                  The path to save the plots
            
            """
            for feature in features[:max_plots] if max_plots > 0 else features:
                  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                  # Training set plot
                  sns.histplot(data=self.dataset.X_train[feature], bins=20, ax=axes[0])
                  axes[0].set_title(f'{feature} - Training Set')
                  
                  # Validation set plot
                  sns.histplot(data=self.dataset.X_val[feature], bins=20, ax=axes[1])
                  axes[1].set_title(f'{feature} - Validation Set')
                  
                  # Test set plot
                  sns.histplot(data=self.dataset.X_test[feature], bins=20, ax=axes[2])
                  axes[2].set_title(f'{feature} - Test Set')
                  
                  plt.tight_layout()

                  save_or_store_plot(fig, save_plots, save_path + "/split/after_split_distribution", f"{feature}_distribution.png")
