from datetime import datetime

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import concurrent.futures

from library.phases.phases_implementation.modelling.shallow.model_definition.model_base import Model
from library.phases.phases_implementation.dataset.dataset import Dataset

class Classifier(Model):
      def __init__(self,  modelName: str, model_sklearn: object, model_type: str, results_header: list[str], dataset: Dataset):
            self.dataset = dataset   
            super().__init__(modelName, model_sklearn, model_type, results_header, dataset)
      

      def __set_assesment(self, 
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
          class_report = classification_report(y_actual, y_pred, output_dict=True) # F1 score, precision, recall for each class

          return class_report
      
      def evaluate(self, modelName: str, current_phase: str):
            print(f"Evaluating {modelName} in {current_phase} phase")
            assert current_phase in ["pre", "in", "post"], "Current phase must be one of the tuning states"
            if current_phase == "pre" or current_phase == "in":
                  y_actual = self.dataset.y_val
                  y_pred = self.tuning_states[current_phase].assesment["predictions_val"]
            elif current_phase == "post":
                  y_actual = self.dataset.y_test
                  y_pred = self.tuning_states[current_phase].assesment["predictions_test"]
            else:
                  raise ValueError("Invalid phase")
            
            assert y_actual is not None, f"y_actual is None for model: {modelName}"
            assert y_pred is not None, f"y_pred is None for model: {modelName}"

            class_report = self.__set_assesment(y_actual, y_pred, modelName)

            accuracy = class_report["accuracy"]
            f1_score = class_report["weighted avg"]["f1-score"]
            precision = class_report["weighted avg"]["precision"]
            recall = class_report["weighted avg"]["recall"]
            results = {
                  "f1-score": f1_score,
                  "precision": precision,
                  "recall": recall,
                  "accuracy": accuracy
            }
            print(f"METRIC RESULTS FOR {modelName} => F1: {f1_score}, Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}")
            # Storing results to assesment attribute
            for metric, value in results.items():
                  self.tuning_states[current_phase].assesment[metric] = value

