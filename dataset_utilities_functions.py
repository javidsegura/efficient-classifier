import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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
    def __init__(self, path: str) -> None:
      """
      Creates a dataframe from a csv file

      Parameters
      ----------
      path : str
          The path to the dataframe
      """ 
      self.df = pd.read_csv(path)

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
  
    def asses_split(self, p: float, step: float, plot: bool = True, upper_bound: float = .50) -> pd.DataFrame:
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
  
    def __get_X_y__(self, y_column: str, otherColumnsToDrop: list[str] = []) -> tuple[pd.DataFrame, pd.Series]:
      """Splits the dataframe into features and target variable"""
      X = self.df.drop(columns=[y_column] + otherColumnsToDrop)
      y = self.df[y_column]
      return X, y
  
    def split_data(self, 
                 y_column: str, 
                 otherColumnsToDrop: list[str] = [], 
                 train_size: float = 0.8, 
                 validation_size: float = 0.1, 
                 test_size: float = 0.1
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
      """
      Splits the dataframe into training, validation and test sets

      Parameters
      ----------
        y_column : str
            The column name of the target variable
        otherColumnsToDrop : list[str]
            The columns to drop from the dataframe
        train_size : float
            The size of the training set
        validation_size : float
            The size of the validation set
        test_size : float
            The size of the test set
      Returns
      -------
        X_train : pd.DataFrame
            The training set
        X_validation : pd.DataFrame
            The validation set
        X_test : pd.DataFrame
            The test set
        y_train : pd.Series
            The training set
        y_validation : pd.Series
            The validation set
        y_test : pd.Series
            The test set
      """
      X, y = self.__get_X_y__(y_column, otherColumnsToDrop)
      assert train_size + validation_size + test_size == 1, "The sum of the sizes must be 1"
      X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=validation_size + test_size, random_state=RANDOM_STATE) 
      X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(validation_size + test_size), random_state=RANDOM_STATE) 
      self.X_train, self.X_validation, self.X_test, self.y_train, self.y_validation, self.y_test = X_train, X_validation, X_test, y_train, y_validation, y_test

      return X_train, X_validation, X_test, y_train, y_validation, y_test
    
    def get_dataset_encoded(self, 
                            encode_y: bool = True
                            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
      """
      Encodes the categorical features for the training, validation and test sets

      Parameters
      ----------
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
      cat_cols = self.X_train.select_dtypes(include=["object"]).columns
      # Training set
      encoded_array = encoder.fit_transform(self.X_train[cat_cols])
      encoded_cols = encoder.get_feature_names_out(cat_cols)
      train_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=self.X_train.index)
      X_train_encoded = self.X_train.drop(cat_cols, axis=1).join(train_encoded)
      # Validation set
      encoded_array_val = encoder.transform(self.X_validation[cat_cols])
      val_encoded = pd.DataFrame(encoded_array_val, columns=encoded_cols, index=self.X_validation.index)
      X_val_encoded = self.X_validation.drop(cat_cols, axis=1).join(val_encoded)
      # Test set
      encoded_array_test = encoder.transform(self.X_test[cat_cols])
      test_encoded = pd.DataFrame(encoded_array_test, columns=encoded_cols, index=self.X_test.index)
      X_test_encoded = self.X_test.drop(cat_cols, axis=1).join(test_encoded)
      self.X_train_encoded, self.X_val_encoded, self.X_test_encoded = X_train_encoded, X_val_encoded, X_test_encoded
      del self.X_train, self.X_validation, self.X_test

      if encode_y:
        labeller = LabelEncoder()
        labeller.fit(self.y_train)
        y_train_encoded = pd.Series(labeller.transform(self.y_train), index=self.y_train.index)
        y_val_encoded = pd.Series(labeller.transform(self.y_validation), index=self.y_validation.index)
        y_test_encoded = pd.Series(labeller.transform(self.y_test), index=self.y_test.index)

        encoding_map = dict(zip(labeller.classes_, range(len(labeller.classes_))))
        self.y_train_encoded, self.y_val_encoded, self.y_test_encoded, self.encoding_map = y_train_encoded, y_val_encoded, y_test_encoded, encoding_map
        del self.y_train, self.y_validation, self.y_test
        return X_train_encoded, X_val_encoded, X_test_encoded, y_train_encoded, y_val_encoded, y_test_encoded, encoding_map
      else:
        return X_train_encoded, X_val_encoded, X_test_encoded
  
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
    
      
class Plots:
  """ 
  We will be using 'composition' desing pattern to create plots from the dataframe object that is an instance of the Dataset class
  This design pattern allows for two classes to be able to share data (e.g: dataset object)
  """
  def __init__(self, dataset: Dataset) -> None:
    self.dataset = dataset

  def plot_correlation_matrix(self, size: str = "small"):
    """
    Plots the correlation matrix of the dataframe

    Parameters
    ----------
      size : str
        The size of the plot. Taken on ["s", "m", "l", "auto"]
    """
    only_numerical_df = self.dataset.df.select_dtypes(include=["number"])
    corr = only_numerical_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool)) # avoid redundancy
    if size == "s":
      f, ax = plt.subplots(figsize=(5, 3))
    elif size == "m":
      f, ax = plt.subplots(figsize=(10, 6))
    elif size == "l":
      f, ax = plt.subplots(figsize=(20, 15))
    elif size == "auto":
      f, ax = plt.subplots()
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    vmin, vmax = corr.min().min(), corr.max().max()
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
              square=True, linewidths=.5, cbar_kws={"shrink": .8}, vmin=vmin, vmax=vmax)


