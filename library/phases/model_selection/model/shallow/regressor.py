from datetime import datetime

from library.phases.model_selection.model.model import Model
from library.phases.dataset.dataset import Dataset

import concurrent.futures 

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Regressor(Model):
      def __init__(self,  modelName: str, model_sklearn: object, results_header: list[str], dataset: Dataset):
            self.dataset = dataset   
            super().__init__(modelName, model_sklearn, results_header, dataset)
      

      def __set_assesment__(self, 
                        y_actual: pd.Series,
                        y_pred: pd.Series,
                        modelName: str):
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
          mae = mean_absolute_error(y_actual, y_pred)
          mse = mean_squared_error(y_actual, y_pred)
          r2 = r2_score(y_actual, y_pred)

          return mae, mse, r2
      
      def evaluate(self, modelName: str):
            if self.currentPhase == "pre_tuning":
                  y_actual = self.dataset.y_val
                  y_pred = self.preTuningState.assesment["predictions"]
            elif self.currentPhase == "in_tuning":
                  y_actual = self.dataset.y_val
                  y_pred = self.inTuningState.assesment["predictions"]
            elif self.currentPhase == "post_tuning":
                  y_actual = self.dataset.y_test
                  y_pred = self.postTuningState.assesment["predictions"]
            else:
                  raise ValueError("Invalid phase")
            
            mae, mse, r2 = self.__set_assesment__(y_actual, y_pred, modelName)
            print(f"METRIC RESULTS => MAE: {mae}, MSE: {mse}, R2: {r2}")
            metric_results = {"mae": mae, "mse": mse, "r2": r2}
            if self.currentPhase == "pre_tuning":
                  assesment = self.preTuningState.store_assesment(metric_results)
            elif self.currentPhase == "in_tuning":
                  assesment = self.inTuningState.store_assesment(metric_results)
            elif self.currentPhase == "post_tuning":
                  assesment = self.postTuningState.store_assesment(metric_results)
            else:
                  raise ValueError("Invalid phase")
            
            return assesment
      

