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

# Global variables
RANDOM_STATE = 99

class Dataset:
    """ Created dataframe, provides info, splits and encodes"""
    def __init__(self, path: str, task: str) -> None:
      """
      Creates a dataframe from a csv file

      Parameters
      ----------
      path : str
          The path to the dataframe
      """ 
      assert task in ["classification", "regression"], "The task must be either classification or regression"
      self.df = pd.read_csv(path)
      self.isXencoded = False
      self.isYencoded = False
      self.task = task

    def get_basic_info(self, print_info: bool = True) -> dict:
      """
      Returns a dictionary with the basic information of the dataframe

      Parameters
      ----------
        print_info : bool
            If True, the information will be printed
      Returns
      -------
        info : dict
            Returns a dictionary with the basic information of the dataframe
      """
      info = {
         "number of rows": self.df.shape[0],
         "number of columns": self.df.shape[1],
         "column names": self.df.columns.tolist(),
         "column data types": self.df.dtypes.to_dict(),
         "number of missing values": self.df.isnull().sum().to_dict()
      }
      if print_info:
         print(info)
      return info
  
    def remove_duplicates(self):
        """
        Removes duplicates from the dataset
        """
        duplicates = self.df.duplicated()
        duplicates_sum = duplicates.sum()
        if duplicates_sum > 0:
            print(f"Dataset duplicates: \n {self.df[duplicates]}")
            print(f"There are {duplicates_sum} duplicates in the dataset")
            self.df.drop_duplicates(inplace=True)
            print(f"Succesfully removed duplicates from the dataset")
        else:
            print("No duplicates found in the dataset")
    
    def __get_X_y__(self, y_column: str, otherColumnsToDrop: list[str] = []) -> tuple[pd.DataFrame, pd.Series]:
      """Splits the dataframe into features and target variable"""
      X = self.df.drop(columns=[y_column] + otherColumnsToDrop)
      y = self.df[y_column]
      return X, y
  
    def asses_split_classifier(self, p: float, step: float, plot: bool = True, upper_bound: float = .50) -> pd.DataFrame:
      """
      Assesses the split of the dataframe

      Parameters
      ----------
        p : float
            The percentage of the dataframe to split
        step : float
            The step size for the split
        upper_bound : float
            The upper bound for the split
        plot : bool
            If True, the split assessment will be plotted
      Returns
      -------
        df_split_assesment : pd.DataFrame
            A dataframe with the split assessment
      """
      computeSE = lambda p, n : np.sqrt((p*(1-p))/n)
      df_split_assesment = pd.DataFrame()
      hold_out_size = step
      priorSE = 0
      while hold_out_size <= upper_bound:
            assert hold_out_size < 1 
            train_size_percentage  = 1 - hold_out_size
            train_size_count = round(self.df.shape[0] * train_size_percentage, 0)

            val_size_percentage = hold_out_size / 2
            val_size_count = round(self.df.shape[0] * (hold_out_size / 2),0)

            test_size_percentage = hold_out_size / 2
            test_size_count = round(self.df.shape[0] * (hold_out_size / 2),0)


            currentSE = computeSE(p, test_size_count)
            differenceToPriorSE = currentSE - priorSE
            differenceToPriorSE_percentage = (currentSE - priorSE) /  priorSE
            priorSE = currentSE

            new_row = pd.DataFrame([{
              "train_size (%)": train_size_percentage, 
              "train_size_count": train_size_count,
              "validation_size (%)": val_size_percentage ,
              "validation_size_count": val_size_count,
              "test_size (%)": test_size_percentage, 
              "test_size_coount": test_size_count,
              "currentSE": currentSE ,
              "differenceToPriorSE": differenceToPriorSE,
              "differenceToPriorSE (%)": differenceToPriorSE_percentage,
            }])

            # Concatenate the new row with your existing DataFrame
            df_split_assesment = pd.concat([df_split_assesment, new_row], ignore_index=True)
            hold_out_size += step
      if plot:
         fig, ax1 = plt.subplots()

         color = 'tab:blue'
         ax1.set_xlabel('Training Set Percentage')
         ax1.set_ylabel('Current SE', color=color)
         ax1.plot(df_split_assesment["train_size (%)"], df_split_assesment["currentSE"], marker='o', color=color)
         ax1.tick_params(axis='y', labelcolor=color)

         ax1.xaxis.set_major_locator(MultipleLocator(0.05))

         ax2 = ax1.twinx()  
         color = 'tab:red'
         ax2.set_ylabel('Difference to Prior SE (%)', color=color)
         ax2.plot(df_split_assesment["train_size (%)"][1:],  df_split_assesment["differenceToPriorSE (%)"][1:], marker='x', linestyle='--', color=color)
         ax2.tick_params(axis='y', labelcolor=color)


         plt.title('Holdout Split Trade-Off: Training Set vs SE')
         plt.show()
      self.df_split_assesment = df_split_assesment
      return df_split_assesment
    
    def plot_time_splits(self):
      """Plots the time splits of the dataframe"""

      plt.figure(figsize=(20, 3))

      plt.plot(self.X_train['dteday'], [1] * len(self.X_train), '|', label='Train')
      plt.plot(self.X_val['dteday'], [1.5] * len(self.X_val), '|', label='Val')
      plt.plot(self.X_test['dteday'], [2] * len(self.X_test), '|', label='Test')

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
    
    def plot_per_set_distribution(self, features: list[str]):
        """Plots the distribution of the features for the training, validation and test sets"""


        for feature in features:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Training set plot
            sns.histplot(data=self.X_train_encoded[feature] if self.isXencoded else self.X_train[feature], bins=20, ax=axes[0])
            axes[0].set_title(f'{feature} - Training Set')
            
            # Validation set plot
            sns.histplot(data=self.X_val_encoded[feature] if self.isXencoded else self.X_val[feature], bins=20, ax=axes[1])
            axes[1].set_title(f'{feature} - Validation Set')
            
            # Test set plot
            sns.histplot(data=self.X_test_encoded[feature] if self.isXencoded else self.X_test[feature], bins=20, ax=axes[2])
            axes[2].set_title(f'{feature} - Test Set')
            
            plt.tight_layout()
    
    def split_data_time_series(self, 
                              y_column: str, 
                              otherColumnsToDrop: list[str] = [], 
                              train_size: float = 0.8, 
                              validation_size: float = 0.1, 
                              test_size: float = 0.1,
                              orderColumns: list[str] = [],
                              plot_distribution: bool = True,
                              plot_time_splits: bool = True
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
      assert len(orderColumns) > 0, "The order columns must be provided"

      # Order the dataframe by the order columns
      self.df = self.df.sort_values(by=orderColumns)

      X, y = self.__get_X_y__(y_column, otherColumnsToDrop)

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
      
      self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
      self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

      if plot_distribution:
        self.plot_per_set_distribution(X.columns)
      if plot_time_splits:
        self.plot_time_splits()
      
    
    def split_data(self, 
                 y_column: str, 
                 otherColumnsToDrop: list[str] = [], 
                 train_size: float = 0.8, 
                 validation_size: float = 0.1, 
                 test_size: float = 0.1,
                 plot_distribution: bool = True
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
      """
      Splits the dataframe into training, validation and test sets

      Parameters
      ----------
        y_column : str
            The column name of the target variable
        otherColumnsToDrop : list[str]
            The columns to drop from the dataframe (e.g: record identifiers)
        train_size : float
            The size of the training set
        validation_size : float
            The size of the validation set
        test_size : float
            The size of the test set
        plot_distribution : bool
            Whether to plot the distribution of the features
      Returns
      -------
        X_train : pd.DataFrame
            The training set
        X_val : pd.DataFrame
            The validation set
        X_test : pd.DataFrame
            The test set
        y_train : pd.Series
            The training set
        y_val : pd.Series
            The validation set
        y_test : pd.Series
            The test set
      """
      X, y = self.__get_X_y__(y_column, otherColumnsToDrop)
      assert train_size + validation_size + test_size == 1, "The sum of the sizes must be 1"
      X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=validation_size + test_size, random_state=RANDOM_STATE) 
      X_val , X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(validation_size + test_size), random_state=RANDOM_STATE) 
      self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = X_train, X_val, X_test, y_train, y_val, y_test
      if plot_distribution:
        self.plot_per_set_distribution(X.columns)

      return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_categorical_features_encoded(self, 
                                          features: list[str],
                                          encode_y: bool = True
                                          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
      """
      Encodes the categorical features for the training, validation and test sets

      Parameters
      ----------
        features : list[str]
            The features to encode
        encode_y : bool
            Whether to encode the target variable
      Returns
      -------
        X_train_encoded : pd.DataFrame
            The training set
        X_val_encoded : pd.DataFrame
            The validation set
        X_test_encoded : pd.DataFrame
            The test set
        y_train_encoded : pd.Series
            The training set
        y_val_encoded : pd.Series
            The validation set
        y_test_encoded : pd.Series
            The test set
        encoding_map : dict
            The encoding map
      """
      encoder = OneHotEncoder(handle_unknown="ignore", 
                        sparse_output=False,
                        dtype=int,
                        drop="first"
                        )
      # Training set
      encoded_array = encoder.fit_transform(self.X_train[features])
      encoded_cols = encoder.get_feature_names_out(features)
      train_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=self.X_train.index)
      X_train_encoded = self.X_train.drop(features, axis=1).join(train_encoded)
      # Validation set
      encoded_array_val = encoder.transform(self.X_val[features])
      val_encoded = pd.DataFrame(encoded_array_val, columns=encoded_cols, index=self.X_val.index)
      X_val_encoded = self.X_val.drop(features, axis=1).join(val_encoded)
      # Test set
      encoded_array_test = encoder.transform(self.X_test[features])
      test_encoded = pd.DataFrame(encoded_array_test, columns=encoded_cols, index=self.X_test.index)
      X_test_encoded = self.X_test.drop(features, axis=1).join(test_encoded)
      self.X_train_encoded, self.X_val_encoded, self.X_test_encoded = X_train_encoded, X_val_encoded, X_test_encoded
      self.isXencoded = True
      del self.X_train, self.X_val, self.X_test

      if encode_y:
        labeller = LabelEncoder()
        labeller.fit(self.y_train)
        y_train_encoded = pd.Series(labeller.transform(self.y_train), index=self.y_train.index)
        y_val_encoded = pd.Series(labeller.transform(self.y_val), index=self.y_val.index)
        y_test_encoded = pd.Series(labeller.transform(self.y_test), index=self.y_test.index)

        encoding_map = dict(zip(labeller.classes_, range(len(labeller.classes_))))
        self.y_train_encoded, self.y_val_encoded, self.y_test_encoded, self.encoding_map = y_train_encoded, y_val_encoded, y_test_encoded, encoding_map
        del self.y_train, self.y_val, self.y_test
        self.isYencoded = True
        return X_train_encoded, X_val_encoded, X_test_encoded, y_train_encoded, y_val_encoded, y_test_encoded, encoding_map
      else:
        return X_train_encoded, X_val_encoded, X_test_encoded
  
    def get_cylical_features_encoded(self, features: list[str]) -> pd.DataFrame:
      """Encodes the cyclical features (done before encoding the categorical features)"""
      # Get features to be numerical
      self.X_train[features] = self.X_train[features].astype("int")
      self.X_val[features] = self.X_val[features].astype("int")
      self.X_test[features] = self.X_test[features].astype("int")
      for feature in features:
          self.X_train[f"{feature}_sin"] = np.sin((2 * np.pi * self.X_train[feature]) / 24)
          self.X_val[f"{feature}_sin"] = np.sin((2 * np.pi * self.X_val[feature]) / 24)
          self.X_test[f"{feature}_sin"] = np.sin((2 * np.pi * self.X_test[feature]) / 24)
          self.X_train[f"{feature}_cos"] = np.cos((2 * np.pi * self.X_train[feature]) / 24)
          self.X_val[f"{feature}_cos"] = np.cos((2 * np.pi * self.X_val[feature]) / 24)
          self.X_test[f"{feature}_cos"] = np.cos((2 * np.pi * self.X_test[feature]) / 24)
          self.X_train.drop(columns=[feature], inplace=True)
          self.X_val.drop(columns=[feature], inplace=True)
          self.X_test.drop(columns=[feature], inplace=True)


    def eliminate_features_from_all_sets(self, featuresToEliminate: list[str]):
      """Eliminates variables from the dataframe"""
      listOfSets = [self.X_train_encoded, self.X_val_encoded, self.X_test_encoded]
      lengthOfSets = [len(set) for set in listOfSets]
      for set in listOfSets:
        set.drop(columns=featuresToEliminate, 
                inplace=True,
                errors='ignore') # ignore errors if the variable is not in the set
      for i in range(len(listOfSets)):
        if len(listOfSets[i]) == lengthOfSets[i]:
          print(f"No modifications were made to the set {i}")
    



