from abc import ABC, abstractmethod

import time
from library.phases.dataset.dataset import Dataset
import matplotlib.pyplot as plt
import pandas as pd

from library.phases.model_selection.model_state import PreTuningState, PostTuningState, InTuningState
from library.phases.model_selection.model_optimization.model_optimization import Optimizer


class Model(ABC):
      def __init__(self, modelName: str, model_sklearn: object, results_header: list[str], dataset: Dataset):
            self.dataset = dataset
            self.modelName = modelName
            self.model_sklearn = model_sklearn
            # Remove from header the duplicate metrics
            cleaned_header = []
            for col in results_header:
                  if col.endswith("_val"):
                       cleaned_header.append(col.split("_")[0])
                  elif col.endswith("_test"):
                       continue
                  else:
                        cleaned_header.append(col)
            self.results_header = cleaned_header + ["predictions", "model_sklearn"]

            self.tuning_states = {
                  "pre": PreTuningState(model_sklearn, modelName, dataset, self.results_header),
                  "in": InTuningState(model_sklearn, modelName, dataset, self.results_header),
                  "post": PostTuningState(model_sklearn, modelName, dataset, self.results_header)
            }
            self.currentPhase = "pre"
            self.optimizer_type = None


      @abstractmethod
      def evaluate(self, modelName: str):
            pass

      @abstractmethod
      def evaluate_training(self, modelName: str):
            pass

      def fit(self, modelName: str):      
            print(f"=> Fitting {modelName} model")
            if self.currentPhase == "pre":
                  self.tuning_states["pre"].fit()
            elif self.currentPhase == "post":
                  self.tuning_states["post"].fit()

      
      def predict(self, modelName: str):
            print(f"=> Predicting {modelName} model")
            if self.currentPhase == "pre":
                  self.tuning_states["pre"].predict()
            elif self.currentPhase == "post":
                  self.tuning_states["post"].predict()
      
      def optimize(self, param_grid: dict, max_iter):
            assert self.optimizer_type, "Optimizer type must be set before optimizing"
            optimizer = Optimizer(self.model_sklearn, self.modelName, self.dataset, self.optimizer_type, param_grid, max_iter)
            cv_results_df, best_model, y_pred = optimizer.start_optimization()
            self.inTuningState.assesment["predictions"] = y_pred
            self.inTuningState.assesment["model_sklearn"] = best_model
            self.inTuningState.assesment["modelName"] = self.modelName
            assesment = self.evaluate(self.modelName)
            return cv_results_df, best_model, assesment, optimizer 

      def _constrast_sets_predictions(self):
            """
            Constructs a plot to compare the distribution of predicted labels for the test and validation sets for all models

            Returns
            -------
                  None
            """

            fig, axes = plt.subplots(2, 1, figsize=(12, 5))

            # Pre-tuning
            axes[0].hist(self.tuning_states["pre"].assesment["predictions"], bins=30, edgecolor='black', alpha=0.5, label='Predictions (Validation Set / Pre-tuning)')
            axes[0].set_title(f'{self.modelName} - Distribution of Predicted Values (Validation Set)')
            axes[0].set_xlabel('Predicted Values')
            axes[0].set_ylabel('Frequency')
            axes[0].legend()
            axes[0].hist(self.dataset.y_val, bins=30, edgecolor='black', alpha=0.5, label='Actual Predictions (Validation Set / Pre-tuning)')
            axes[0].set_title(f'{self.modelName} - Distribution of Predicted Values (Validation Set)')
            axes[0].set_xlabel('Predicted Values')
            axes[0].set_ylabel('Frequency')
            axes[0].legend()
            # Post-tuning
            axes[1].hist(self.tuning_states["post"].assesment["predictions"], bins=30, edgecolor='black', alpha=0.5, label='Predictions (Test Set / Post-tuning)')
            axes[1].set_title(f'{self.modelName} - Distribution of Predicted Values (Test Set)')
            axes[1].set_xlabel('Predicted Values')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
            axes[1].hist(self.dataset.y_test, bins=30, edgecolor='black', alpha=0.5, label='Actual Predictions (Test Set / Post-tuning)')
            axes[1].set_title(f'{self.modelName} - Distribution of Predicted Values (Test Set)')
            axes[1].set_xlabel('Predicted Values')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()

            plt.tight_layout()
            plt.show()

