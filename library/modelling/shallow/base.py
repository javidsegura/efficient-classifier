import time
import csv
import os
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import numpy as np
import pandas as pd
import concurrent.futures


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from imblearn.over_sampling import SMOTENC
from boruta import BorutaPy
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier

from library.dataset import Dataset

# Global variables
RANDOM_STATE = 99


class ModelAssesment:
  """
  Base class for all shallow models. Provides a general overview of all the models.
  """
  def __init__(self, dataset: Dataset, results_path: str, results_columns: list, columns_to_check_duplicates: list) -> None:
    """
    Initializes the ModelAssesment class

    Parameters
    ----------
      dataset : Dataset
        The dataset to be used for the model assessment
    """
    assert len(results_columns) > 0, "The results columns must be a non-empty list"
    self.models = dict()
    self.dataset = dataset
    self.results_path = results_path
    self.results_columns = sorted(results_columns)
    self.columns_to_check_duplicates = columns_to_check_duplicates
    self.__create_results_file__()

  def __create_results_file__(self):
    if os.path.exists(self.results_path):
      return
    else: 
      os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
      with open(self.results_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(self.results_columns)

  def get_models_names(self) -> list:
    """
    Returns the names of the models

    Returns
    -------
      list
      The names of the models
    """
    return list(self.models.keys())
  
  def get_model_results_saved(self, dataToWrite: dict, featuresUsed: list):
    """
    Returns the model by name
    """
    results = pd.read_csv(self.results_path)
    dataToWrite["features_used"] = featuresUsed
    dataToWrite["id"] = None
    if (sorted(list(dataToWrite.keys())) != self.results_columns):
      raise ValueError(f"The data to write does not match the columns of the results. \n Data to write: {sorted(list(dataToWrite.keys()))} \n Data header: {self.results_columns}")
    
    # Compute hash
    dataForHash = {k: v for k, v in dataToWrite.items() if k in self.columns_to_check_duplicates}
    hash_value = hashlib.sha256(json.dumps(dataForHash).encode()).hexdigest()

    print(f"Hash value: {hash_value}")

    # Debug prints
    isNewModel = hash_value not in results["id"].values
    dataToWrite["id"] = hash_value

    print(f"IS NEW MODEL: {isNewModel}?")

    if isNewModel:
      with open(self.results_path, "a", newline='') as f: 
          writer = csv.writer(f)
          writer.writerow([str(dataToWrite[col]) for col in self.results_columns])
      print(f"!> Model results stored succesfully")
    else:
       print(f"****WARNING****: A model with the same values already exists in the results. Results will not be saved. \n \
             You tried to write {dataToWrite}")
    return isNewModel

  def add_model(self, modelName: str, modelObject: dict):
    """
    Adds a model to the models dictionary

    Parameters
    ----------
      modelName : str
        The name of the model
      modelObject : dict
        The model object
    """
    self.models[modelName] = {
      "model": modelObject,
      "timeToFit": None,
      "timeToPredict": None,
      "val_predictions": None,
      "test_predictions": None,
      "metrics": None
    }
    hyperParameters = []
    attributesToCheck = ["max_iter", "class_weight", "penalty", "C", "solver", "random_state", "max_depth",
                          "n_estimators", "min_samples_split", "min_samples_leaf", "min_weight_fraction_leaf", 
                          "max_features", "max_leaf_nodes", "min_impurity_decrease", "min_impurity_split", "bootstrap", 
                          "oob_score", "n_jobs", "verbose", "warm_start", "class_weight", "ccp_alpha", "max_samples", "positive"]
    for attribute in attributesToCheck:
      if hasattr(modelObject, attribute):
        hyperParameters.append(f"{attribute}: {getattr(modelObject, attribute)}")
    self.models[modelName]["hyperParameters"] = hyperParameters
    print(f"Added {modelName} to the models dictionary successfully")

  def __fit_and_predict__(self, model_item, print_results: bool = True):
    """
    Fits and predicts a model

    Parameters
    ----------
      model_item : tuple
        The model item
      print_results : bool
        Whether to print the results
      y_encoded : bool
        Whether the target variable is encoded
    Returns
    -------
      tuple
      The model item
    """

    classifierName, classifier = model_item
    start_time = time.time()
    if print_results:
      print(f"!> Started fitting {classifierName}")
    if self.dataset.isYencoded:
      classifier["model"].fit(self.dataset.X_train_encoded if self.dataset.isXencoded else self.dataset.X_train, 
                              self.dataset.y_train_encoded)
    else:
      classifier["model"].fit(self.dataset.X_train_encoded if self.dataset.isXencoded else self.dataset.X_train, 
                              self.dataset.y_train)
    end_time = time.time()
    time_to_fit = end_time - start_time
    classifier["timeToFit"] = time_to_fit
    if print_results:
      print(f"\t\t => Fitted {classifierName}. Took {time_to_fit} seconds")
    
    classifierName, classifier = model_item
    if print_results:
      print(f"!> Started predicting for {classifierName}")
    start_time = time.time()
    classifier["val_predictions"] = classifier["model"].predict(self.dataset.X_val_encoded if self.dataset.isXencoded else self.dataset.X_val)
    classifier["test_predictions"] = classifier["model"].predict(self.dataset.X_test_encoded if self.dataset.isXencoded else self.dataset.X_test)
    end_time = time.time()
    classifier["timeToMakePredictions"] = end_time - start_time
    if print_results:
      print(f"\t\t => Predicted {classifierName}. Took {classifier['timeToMakePredictions']} seconds")
    return classifierName, classifier
  
  def fit_models(self, print_results: bool = True, modelsToExclude: list = []) -> dict:
    """Fits and predicts the models in parallel"""
    with concurrent.futures.ProcessPoolExecutor() as executor:
      # Submit all model fitting tasks to the executor
      future_to_model = {executor.submit(self.__fit_and_predict__, item, print_results): item for item in self.models.items() if item[0] not in modelsToExclude}
      
      for future in concurrent.futures.as_completed(future_to_model):
          classifierName, classifier = future.result() 
          self.models[classifierName] = classifier # update results
    if print_results:
      print("All models have been fitted and made predictions in parallel.")
    return self.models
  
  def constrast_sets_predictions(self):
    """
    Constructs a plot to compare the distribution of predicted labels for the test and validation sets for all models

    Returns
    -------
      None
    """
    number_of_models = 0
    for modelName, model in self.models.items():
      if model["test_predictions"] is None or model["val_predictions"] is None:
        continue
      number_of_models += 1
    fig, axes = plt.subplots(number_of_models, 2, figsize=(12, 5*number_of_models))

    if number_of_models == 1:
        axes = [axes]

    for ax, (modelName, model) in zip(axes, self.models.items()):
        ax[0].hist(model["val_predictions"], bins=30, edgecolor='black', alpha=0.5, label='Validation Predictions')
        ax[0].hist(self.dataset.y_val, bins=30, edgecolor='black', alpha=0.5, label='Actual Predictions (Validation Set)')
        ax[1].hist(model["test_predictions"], bins=30, edgecolor='black', alpha=0.5, label='Test Predictions')
        ax[1].hist(self.dataset.y_test, bins=30, edgecolor='black', alpha=0.5, label='Actual Predictions (Test Set)')
        
        ax[0].set_title(f'{modelName} - Distribution of Predicted Values (Validation Set)')
        ax[0].set_xlabel('Predicted Values')
        ax[0].set_ylabel('Frequency')
        ax[0].legend()
        ax[1].set_title(f'{modelName} - Distribution of Predicted Values (Test Set)')
        ax[1].set_xlabel('Predicted Values')
        ax[1].set_ylabel('Frequency')
        ax[1].legend()

    plt.tight_layout()
    plt.show()


  

      
  