import time

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

from dataset_utilities_functions import Dataset

# Global variables
RANDOM_STATE = 99


class ModelAssesmentClassifiers:
  """
  This class is used to asses the performance of a model. Provides a general overview of all the models.
  """
  def __init__(self, dataset: Dataset) -> None:
    """
    Initializes the ModelAssesment class

    Parameters
    ----------
      dataset : Dataset
        The dataset to be used for the model assessment
    """
    self.models = dict()
    self.dataset = dataset
  def get_models_names(self) -> list:
    """
    Returns the names of the models

    Returns
    -------
      list
      The names of the models
    """
    return list(self.models.keys())
  def automatic_feature_selection_l1(self, logistic_model: dict, print_results: bool = True):
    """
    Automatically selects the features that are most predictive of the target variable using the L1 regularization method

    Parameters
    ----------
      logistic_model : dict
        The logistic regression model
      print_results : bool
        Whether to print the results

    Returns
    -------
      tuple
      The predictive power features and the excluded features
    """
    logistic_model.fit(self.dataset.X_train_encoded, self.dataset.y_train_encoded)
    coefficients = logistic_model.coef_
    predictivePowerFeatures = set()
    for i in range(len(coefficients[0])):
      if abs(coefficients[0][i]) > 0:
        predictivePowerFeatures.add(self.dataset.X_train_encoded.columns[i])
    excludedFeatures = set(self.dataset.X_train_encoded.columns) - predictivePowerFeatures
    if print_results:
      print(f"Number of predictive power variables: {len(predictivePowerFeatures)}")
      print(f"Number of excluded variables: {len(excludedFeatures)}")
    return predictivePowerFeatures, excludedFeatures

  def automatic_feature_selection_boruta(self, boruta_model: BorutaPy, print_results: bool = True):
    """
    Automatically selects the features that are most predictive of the target variable using the Boruta method

    Parameters
    ----------
      boruta_model : BorutaPy
        The Boruta model
      print_results : bool
        Whether to print the results

    Returns
    -------
      tuple
      The predictive power features and the excluded features
    """
    boruta_model.fit(self.dataset.X_train_encoded.values, 
                        self.dataset.y_train_encoded.values)
    selected_mask = boruta_model.support_
    selected_features = set(self.dataset.X_train_encoded.columns[selected_mask])
    excludedFeatures = set(self.dataset.X_train_encoded.columns) - selected_features
    if print_results:
      print(f"Number of predictive power variables: {len(selected_features)}")
      print(f"Number of excluded variables: {len(excludedFeatures)}")
    return selected_features, excludedFeatures

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

    Returns
    -------
      tuple
      The model item
    """
    classifierName, classifier = model_item
    start_time = time.time()
    if print_results:
      print(f"!> Started fitting {classifierName}")
    classifier["model"].fit(self.dataset.X_train_encoded, self.dataset.y_train_encoded)
    end_time = time.time()
    time_to_fit = end_time - start_time
    classifier["timeToFit"] = time_to_fit
    if print_results:
      print(f"\t\t => Fitted {classifierName}. Took {time_to_fit} seconds")
    
    classifierName, classifier = model_item
    if print_results:
      print(f"!> Started predicting for {classifierName}")
    start_time = time.time()
    classifier["val_predictions"] = classifier["model"].predict(self.dataset.X_val_encoded)
    classifier["test_predictions"] = classifier["model"].predict(self.dataset.X_test_encoded)
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
    fig, axes = plt.subplots(number_of_models, 1, figsize=(8, 5*number_of_models))

    if number_of_models == 1:
        axes = [axes]

    for ax, (classifierName, classifier) in zip(axes, self.models.items()):
        ax.hist(classifier["test_predictions"], bins=30, edgecolor='black', alpha=0.5, label='Test Predictions')
        ax.hist(classifier["val_predictions"], bins=30, edgecolor='black', alpha=0.5, label='Validation Predictions')
        
        ax.set_title(f'{classifierName} - Distribution of Predicted Labels')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    plt.show()

  def __set_assesment__(self, y_actual: pd.Series, y_pred: pd.Series, plot: bool = True):
    """
    Assesment of the model in a given set 

    Parameters
    ----------
      y_actual : pd.Series
        The actual labels
      y_pred : pd.Series
        The predicted labels
      plot : bool
        Whether to plot the results

    Returns
    -------
      tuple
      The classification report and the confusion matrix
    """
    class_report = classification_report(y_actual, y_pred, output_dict=True) # F1 score, precision, recall for each class
    conf_matrix = confusion_matrix(y_actual, y_pred)
    if plot:
      print(f"Validation Classification Report: \n{class_report}")
      sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
      plt.xlabel("Predicted Label")
      plt.ylabel("True Label")
      plt.title("Confusion Matrix")
      plt.show()
    return class_report, conf_matrix
  
  def evaluate_classifier(self, plot: bool = True, modelsToExclude: list = []):
    for modelName, model in self.models.items():
      if modelName in modelsToExclude or model["test_predictions"] is None or model["val_predictions"] is None:
        continue
      print(f"Evaluating {modelName}")
      print(f"\t => VALIDATION ASSESMENT:")
      y_actual_val = self.dataset.y_val_encoded
      y_pred_val = model["val_predictions"]
      self.__set_assesment__(y_actual_val, y_pred_val, plot)
      print(f"\t => TEST ASSESMENT:")
      y_actual_test = self.dataset.y_test_encoded
      y_pred_test = model["test_predictions"]
      class_report, confusion_matrix = self.__set_assesment__(y_actual_test, y_pred_test, plot)
      self.models[modelName]["metrics"] = {
        "class_report": class_report,
        "confusion_matrix": confusion_matrix
      }

      
  
      
