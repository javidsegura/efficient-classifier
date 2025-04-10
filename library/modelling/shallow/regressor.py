from datetime import datetime

from library.dataset import Dataset
from library.modelling.shallow.base import ModelAssesment

import concurrent.futures 

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



class RegressorAssesment(ModelAssesment):
  """
  This class is used to assess the performance of a regressor.
  """
  def __init__(self, dataset: Dataset, results_path: str, results_columns: list, columns_to_check_duplicates: list) -> None:
    super().__init__(dataset, results_path, results_columns, columns_to_check_duplicates)

  def __set_assesment__(self, y_actual: pd.Series, y_pred: pd.Series, plot: bool = True):
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    if plot:
      print(f"\t\t => MAE: {mae}")
      print(f"\t\t => MSE: {mse}")
      print(f"\t\t => R2: {r2}")
    return mae, mse, r2
  
  def __model_assesment__(self, 
                          modelName: str, 
                          model: dict, 
                          plot: bool = True, 
                          save_results: bool = True, 
                          dataToWrite: dict = {}):
    """
    Evaluate the regressor and save the results
    """
    print(f"Evaluating {modelName}")
    print(f"\t => VALIDATION ASSESMENT:")
    y_actual_val = self.dataset.y_val_encoded if hasattr(self.dataset, "y_val_encoded") else self.dataset.y_val
    y_pred_val = model["val_predictions"]
    assert y_actual_val is not None and y_pred_val is not None, f"y_actual_val or y_pred_val is None for {modelName}. Model name is {modelName}, model object is {model}"
    mae_val, mse_val, r2_val = self.__set_assesment__(y_actual_val, y_pred_val, plot)
    print(f"\t => TEST ASSESMENT:")
    y_actual_test = self.dataset.y_test_encoded if hasattr(self.dataset, "y_test_encoded") else self.dataset.y_test
    y_pred_test = model["test_predictions"]
    assert y_actual_test is not None and y_pred_test is not None, f"y_actual_test or y_pred_test is None for {modelName}. Model name is {modelName}, model object is {model}"
    mae_test, mse_test, r2_test = self.__set_assesment__(y_actual_test, y_pred_test)
    self.models[modelName]["metrics"] = {
        "mae_val": mae_val,
        "mse_val": mse_val,
        "root_mse_val": np.sqrt(mse_val),
        "r2_val": r2_val,
        "mae_test": mae_test,
        "mse_test": mse_test,
        "root_mse_test": np.sqrt(mse_test),
        "r2_test": r2_test
      }
    if save_results:
        dataToWrite["hyperParameters"] = model["hyperParameters"]
        dataToWrite["modelName"] = modelName
        dataToWrite["mse"] = mse_test
        dataToWrite["mse_root"] = np.sqrt(mse_test)
        dataToWrite["mae"] = mae_test
        dataToWrite["r2"] = r2_test
        dataToWrite["timeStamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        wasSaved = self.get_model_results_saved(dataToWrite=dataToWrite, featuresUsed=self.dataset.X_train_encoded.columns.tolist() if self.dataset.isXencoded else self.dataset.X_train.columns.tolist())
    return modelName, model, wasSaved

  def evaluate_regressors(self, 
                           plot: bool = True, 
                           modelsToExclude: list = [],
                           save_results: bool = True, 
                           dataToWrite: dict = {}):
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
      assert dataToWrite is not None, "dataToWrite must be provided if save_results is True"
    for modelName, model in self.models.items():
      if modelName in modelsToExclude or model["test_predictions"] is None or model["val_predictions"] is None:
        modelsToExclude.append(modelName)
    with concurrent.futures.ProcessPoolExecutor() as executor:
      # Submit all model fitting tasks to the executor
        future_to_assesment = {executor.submit(self.__model_assesment__, modelName, model, plot, save_results, dataToWrite): modelName for modelName, model in self.models.items() if modelName not in modelsToExclude}
        
        for future in concurrent.futures.as_completed(future_to_assesment):
            modelName, model, wasSaved = future.result() 
            self.models[modelName] = model # update results
            self.models[modelName]["wasSaved"] = wasSaved
        print("All models have been assesed and results saved.")
    
    # Prepare a list to hold each model's results.
    rows = []
    for model_name, result in self.models.items():
        metrics = result['metrics']
        # Compute deltas as (test - validation)
        delta_mae = metrics['mae_test'] - metrics['mae_val']
        delta_mae_percentage = delta_mae / metrics['mae_val']
        delta_mse = metrics['mse_test'] - metrics['mse_val']
        delta_mse_percentage = delta_mse / metrics['mse_val']
        delta_root_mse = metrics['root_mse_test'] - metrics['root_mse_val']
        delta_root_mse_percentage = delta_root_mse / metrics['root_mse_val']
        delta_r2 = metrics['r2_test'] - metrics['r2_val']
        delta_r2_percentage = delta_r2 / metrics['r2_val']
        # Combine hyperparameters into a single string if desired (optional)
        hyper_str = "; ".join(result['hyperParameters'])
        
        # Build a row dictionary
        row = {
            "Model": model_name,
            "Model_Definition": result['model'],
            "wasSaved": result['wasSaved'],
            "timeToFit": result['timeToFit'],
            "timeToMakePredictions": result['timeToMakePredictions'],
            "mae_val": metrics['mae_val'],
            "mae_test": metrics['mae_test'],
            "delta_mae": delta_mae,
            "delta_mae_percentage": delta_mae_percentage,
            "mse_val": metrics['mse_val'],
            "mse_test": metrics['mse_test'],
            "delta_mse": delta_mse,
            "delta_mse_percentage": delta_mse_percentage,
            "root_mse_val": metrics['root_mse_val'],
            "root_mse_test": metrics['root_mse_test'],
            "delta_root_mse": delta_root_mse,
            "delta_root_mse_percentage": delta_root_mse_percentage,
            "r2_val": metrics['r2_val'],
            "r2_test": metrics['r2_test'],
            "delta_r2": delta_r2,
            "delta_r2_percentage": delta_r2_percentage,
            "HyperParameters": hyper_str
        }
        rows.append(row)

    # Create a DataFrame from the list of dictionaries.
    model_results = pd.DataFrame(rows)
    return model_results
