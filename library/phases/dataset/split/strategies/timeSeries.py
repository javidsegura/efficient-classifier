
from library.phases.dataset.split.strategies.base import Split

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class TimeSeries(Split):
      def __init__(self, dataset) -> None:
            super().__init__(dataset)

      def split_data(self, 
                              y_column: str, 
                              otherColumnsToDrop: list[str] = [], 
                              train_size: float = 0.8, 
                              validation_size: float = 0.1, 
                              test_size: float = 0.1,
                              plot_distribution: bool = True,
                              **kwargs
                              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
            """
            Splits the dataframe into training, validation and test sets for time series data
            
            Parameters
            ----------
            y_column : str
                  The column name of the target variable
            otherColumnsToDrop : list[str]
                  The columns to drop from the dataframe (e.g: record identifiers)
            train_size : float
                  The proportion of data to use for training
            validation_size : float
                  The proportion of data to use for validation
            test_size : float
                  The proportion of data to use for testing
            orderColumns : list[str]
                  The columns to order the dataframe by (e.g., date, timestamp)
            plot_distribution : bool
                  Whether to plot the distribution of the features
            Returns
            -------
            X_train : pd.DataFrame
                  The training set features
            X_val : pd.DataFrame
                  The validation set features
            X_test : pd.DataFrame
                  The test set features
            y_train : pd.Series
                  The training set target
            y_val : pd.Series
                  The validation set target
            y_test : pd.Series
                  The test set target
            """
            assert train_size + validation_size + test_size == 1, "The sum of the sizes must be 1"
            orderColumns = kwargs.get("orderColumns", [])
            plot_time_splits = kwargs.get("plot_time_splits", True)
            assert len(orderColumns) > 0, "The order columns must be provided"

            # Order the dataframe by the order columns
            self.dataset.df = self.dataset.df.sort_values(by=orderColumns)

            X, y = super().__get_X_y__(y_column, otherColumnsToDrop)

            # Calculate split indices
            n = len(X)
            train_end = int(n * train_size)
            val_end = train_end + int(n * validation_size)
            
            # Split the dataframe into training, validation and test sets
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            
            X_val = X.iloc[train_end:val_end]
            y_val = y.iloc[train_end:val_end]
            
            X_test = X.iloc[val_end:]
            y_test = y.iloc[val_end:]
            
            self.dataset.X_train, self.dataset.X_val, self.dataset.X_test = X_train, X_val, X_test
            self.dataset.y_train, self.dataset.y_val, self.dataset.y_test = y_train, y_val, y_test

            if plot_distribution:
                  super().plot_per_set_distribution(X.columns)
            if plot_time_splits:
                  self.plot_time_splits()

      def plot_time_splits(self):
            """Plots the time splits of the dataframe"""

            plt.figure(figsize=(20, 3))

            plt.plot(self.dataset.X_train['dteday'], [1] * len(self.dataset.X_train), '|', label='Train')
            plt.plot(self.dataset.X_val['dteday'], [1.5] * len(self.dataset.X_val), '|', label='Val')
            plt.plot(self.dataset.X_test['dteday'], [2] * len(self.dataset.X_test), '|', label='Test')

            plt.legend()
            plt.yticks([])

            ax = plt.gca()

            # Set locator for ticks (more dense)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())

            # Date format (year-month)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            plt.xticks(rotation=45, fontsize=8)  # smaller font
            plt.xlabel('Date')
            plt.title('Chronological Order Check of Train/Val/Test Splits')

            plt.tight_layout()
            plt.show()