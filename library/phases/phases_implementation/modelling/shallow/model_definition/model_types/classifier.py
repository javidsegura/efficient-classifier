from datetime import datetime

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import yaml

from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import concurrent.futures
from sklearn.metrics import accuracy_score 

from library.phases.phases_implementation.modelling.shallow.model_definition.model_base import Model
from library.phases.phases_implementation.dataset.dataset import Dataset

class Classifier(Model):
      def __init__(self, modelName: str, model_sklearn: object, model_type: str, results_header: list[str], dataset: Dataset):
            super().__init__(modelName, model_sklearn, model_type, results_header, dataset)
            self.dataset = dataset
            with open("library/configurations.yaml", "r") as yaml_file:
                  self.variables = yaml.safe_load(yaml_file)

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
          accuracy = class_report["accuracy"]
          f1_score = class_report["weighted avg"]["f1-score"]
          precision = class_report["weighted avg"]["precision"]
          recall = class_report["weighted avg"]["recall"]

          return accuracy, f1_score, precision, recall
      
      def evaluate(self, modelName: str, current_phase: str):
            print(f"Evaluating {modelName} in {current_phase} phase")
            assert current_phase in ["pre", "in", "post"], "Current phase must be one of the tuning states"
            
            if current_phase == "pre" or current_phase == "in":
                  y_actual = self.dataset.y_val
                  y_pred = self.tuning_states[current_phase].assesment["predictions_val"]
                  y_actual_train = self.dataset.y_train
                  y_pred_train = self.tuning_states[current_phase].assesment["predictions_train"]
            elif current_phase == "post":
                  y_actual = self.dataset.y_test
                  y_pred = self.tuning_states[current_phase].assesment["predictions_test"]
                  y_actual_train = np.concatenate([self.dataset.y_train, self.dataset.y_val])
                  y_pred_train = self.tuning_states[current_phase].assesment["predictions_train"]
            else:
                  raise ValueError("Invalid phase")
            
            assert y_actual is not None, f"y_actual is None for model: {modelName}"
            assert y_pred is not None, f"y_pred is None for model: {modelName}"

            # Base metrics
            accuracy, f1_score, precision, recall = self.__set_assesment(y_actual, y_pred, modelName)

            # Additional metrics
            kappa_val = cohen_kappa_score(y_actual, y_pred)
            kappa_train = cohen_kappa_score(y_actual_train, y_pred_train)

            cw = self.variables["modelling_runner"]["class_weights"]

            try:
                  class_weight_dict = {int(k): v for k, v in cw.items()}
            except Exception:
                  class_weight_dict = {}
           
            if not class_weight_dict:
                  unique_classes = np.unique(np.concatenate([y_actual, y_actual_train]))
                  class_weight_dict = {int(cls): 1.0 for cls in unique_classes}

            weightedaccuracy_val, _, _ = self._calculate_weightedaccuracy(y_actual, y_pred, class_weight_dict)
            weightedaccuracy_train, _ , _ = self._calculate_weightedaccuracy(y_actual_train, y_pred_train, class_weight_dict)

            results = {
                  "base_metrics": {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1-score": f1_score,

                  },
                  "additional_metrics": {
                        "not_train": {
                        "kappa": kappa_val,
                        "weightedaccuracy": weightedaccuracy_val,
                        },
                        "train": {
                        "kappa_train": kappa_train,
                        "weightedaccuracy_train": weightedaccuracy_train,
                        }
                  }
            }
            
            print(f"METRIC RESULTS FOR {modelName} => F1: {f1_score:.4f}, Precision: {precision:.4f}, "
                  f"Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, Weighted Accuracy: {weightedaccuracy_val:.4f}, "
                  f"Kappa: {kappa_val:.4f} (val), {kappa_train:.4f} (train)")
            
            # Storing results to assesment attribute
            self.tuning_states[current_phase].assesment["metrics"] = results
            print(f"Metrics for {modelName} in {current_phase} phase: {results}")


      def _calculate_weightedaccuracy(self, y_actual, y_pred, class_weights):
            if not class_weights or not isinstance(class_weights, dict):
                  return 0.0, {}, 0.0 

            y_actual = np.array(y_actual)
            y_pred = np.array(y_pred)

            weights_array = np.array(list(class_weights.values()), dtype=float)
            normalized_weights = weights_array / weights_array.sum()

            # dictionary of normalized weights
            normalized_weights_dict = {cls: normalized_weights[i] for i, cls in enumerate(class_weights.keys())}
            per_class_accuracy = {}
            weightedaccuracy = 0.0
            total_weight = 0.0

            for cls, weight in normalized_weights_dict.items():
                  cls_mask = (y_actual == cls)
                  if np.any(cls_mask):
                        # accuracy for this class
                        cls_correct = (y_pred[cls_mask] == cls).sum()
                        cls_total = cls_mask.sum()
                        cls_accuracy = cls_correct / cls_total if cls_total > 0 else 0

                        per_class_accuracy[cls] = cls_accuracy

                        # weighted contribution
                        weightedaccuracy += cls_accuracy * weight
                        total_weight += weight

            # normalize the weighted accuracy
            if total_weight > 0:
                  weightedaccuracy /= total_weight

            return weightedaccuracy, per_class_accuracy, total_weight


      def score(self, X, y):
            """
            Returns the accuracy score of the model on the given data.

            Parameters
            ----------
            X : array-like
                  Feature matrix.
            y : array-like
                  True labels.

            Returns
            -------
            float
                  Accuracy score.
            """
            y_pred = self.model_sklearn.predict(X)
            return accuracy_score(y, y_pred)

      def predict_default(self, X):
            """
            Sklearn-compatible predict method (for LIME, cross_val_score, etc.).
            Uses the trained model directly.
            """
            return self.model_sklearn.predict(X)
      
      def predict_proba(self, X):
            return self.model_sklearn.predict_proba(X)

