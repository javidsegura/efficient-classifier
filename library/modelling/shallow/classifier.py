from datetime import datetime

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import concurrent.futures



from library.dataset import Dataset
from library.modelling.shallow.base import ModelAssesment

class ClassifierAssesment(ModelAssesment):
  """
  This class is used to assess the performance of a classifier.
  """
  def __init__(self, dataset: Dataset, results_path: str, metrics_to_evaluate: list) -> None:
    super().__init__(dataset, results_path, metrics_to_evaluate)
  
  def __set_assesment__(self, 
                        y_actual: pd.Series,
                        y_pred: pd.Series):
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
    return class_report, conf_matrix
  
  def _confusion_matrix_per_set(self, confusion_matrix: dict[str, np.ndarray], modelName: str):
    """
    Plot the confusion matrix for each set
    """
    fig, ax = plt.subplots(figsize=(20, 5), nrows=1, ncols=3)
    ax = ax.flatten()
    for i, (set_name, matrix) in enumerate(confusion_matrix.items()):
      ax_count = ax[i]
      sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax_count)
      ax_count.set_xlabel("Predicted Label")
      ax_count.set_ylabel("True Label")
      ax_count.set_title(f"Confusion Matrix for {set_name}")
    performance_metrics = self.models[modelName]["metrics"]
    ax[2].axis('off')
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in performance_metrics.items()])
    ax[2].text(0.5, 0.5, metrics_text, ha='center', va='center', fontfamily='monospace', fontsize=10)
    ax[2].set_title(f"{modelName} - Performance Metrics (weighted avg)")
    plt.tight_layout()
    plt.show() 

  def __model_assesment__(self, 
                          modelName: str, 
                          model: dict, 
                          save_results: bool = True, 
                          phaseProcess: dict = {}):
    """
    Evaluate the classifier and save the results
    """
    featuresUsed = self.dataset.X_train_encoded.columns.tolist() if self.dataset.isXencoded else self.dataset.X_train.columns.tolist()
    y_actual_val = self.dataset.y_val_encoded
    y_pred_val = model["val_predictions"]
    assert y_actual_val is not None and y_pred_val is not None, f"y_actual_val or y_pred_val is None for {modelName}. Model name is {modelName}, model object is {model}"
    class_report_val, confusion_matrix_val = self.__set_assesment__(y_actual_val, y_pred_val)
    y_actual_test = self.dataset.y_test_encoded
    y_pred_test = model["test_predictions"]
    assert y_actual_test is not None and y_pred_test is not None, f"y_actual_test or y_pred_test is None for {modelName}. Model name is {modelName}, model object is {model}"
    class_report_test, confusion_matrix_test = self.__set_assesment__(y_actual_test, y_pred_test)
    model["metrics"] = {}
    model["confusion_matrix"] = {"val": confusion_matrix_val,
                                 "test": confusion_matrix_test}
    for metric in self.metricsToEvaluatePerSet:
      metric_splitted = metric.split("_")
      metric_name = metric_splitted[0]
      metric_type = metric_splitted[1]
      isPercentage = True if len(metric_splitted) > 2 and metric_splitted[2] == "percentage" else False
      if metric_type == "val":
        if metric_name != "accuracy":
          model["metrics"][metric] = class_report_val["weighted avg"][metric_name]
        else:
          model["metrics"][metric] = class_report_val["accuracy"]
      elif metric_type == "test":
        if metric_name != "accuracy":
          model["metrics"][metric] = class_report_test["weighted avg"][metric_name]
        else:
          model["metrics"][metric] = class_report_test["accuracy"]
      elif metric_type == "delta":
        model["metrics"][metric] = model["metrics"][metric_name + "_val"] - model["metrics"][metric_name + "_test"]
      elif isPercentage:
        model["metrics"][metric] = (model["metrics"][metric_name + "_val"] - model["metrics"][metric_name + "_test"]) / model["metrics"][metric_name + "_val"]
    

    self.models[modelName]["featuresUsed"] = featuresUsed
    modelResultsPandasDict = self.get_model_results_saved(phaseProcess=phaseProcess, modelName=modelName, save_results=save_results)
    return modelName, model, modelResultsPandasDict

  def evaluate_classifiers(self, 
                           plot: bool = True, 
                           modelsToExclude: list = [],
                           save_results: bool = True, 
                           phaseProcess: dict = {}):
    """
    Evaluate the classifier and save the results

    Parameters
    ----------
      plot : bool
        Whether to plot the results
      modelsToExclude : list
        The models to exclude from the evaluation
      save_results : bool
        Whether to save the results
      dataToWrite : dict
        The data to write to the results file

    Returns
    -------
      None
    """
    
    if save_results:
      assert phaseProcess is not None, "phaseProcess must be provided if save_results is True"
    for modelName, model in self.models.items():
      if modelName in modelsToExclude or model["test_predictions"] is None or model["val_predictions"] is None:
        modelsToExclude.append(modelName)
    rows = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
      # Submit all model fitting tasks to the executor
        future_to_assesment = {executor.submit(self.__model_assesment__, modelName, model, save_results, phaseProcess): modelName for modelName, model in self.models.items() if modelName not in modelsToExclude}
        
        for future in concurrent.futures.as_completed(future_to_assesment):
            modelName, model, modelResultsPandasDict = future.result() 
            self.models[modelName] = model # update results
            rows.append(modelResultsPandasDict)
        print("All models have been assesed.")
    if plot: # This section need to be done outside the concurrent execution (there are issues with jupyter notebook otherwise)
      for modelName, model in self.models.items():
        if modelName not in modelsToExclude:
          self._confusion_matrix_per_set(model["confusion_matrix"], modelName)
    
    return pd.DataFrame(rows)

     
