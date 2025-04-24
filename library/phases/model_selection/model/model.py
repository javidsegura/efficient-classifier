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
            self.results_header = cleaned_header + ["predictions_val", "predictions_train", "predictions_test", "model_sklearn"]

            self.tuning_states = {
                  "pre": PreTuningState(model_sklearn, modelName, dataset, self.results_header),
                  "in": InTuningState(model_sklearn, modelName, dataset, self.results_header),
                  "post": PostTuningState(model_sklearn, modelName, dataset, self.results_header)
            }
            self.optimizer_type = None


      @abstractmethod
      def evaluate(self, modelName: str):
            pass

      def fit(self, modelName: str, current_phase: str, **kwargs):    
            assert current_phase in self.tuning_states.keys(), "Current phase must be one of the tuning states"
            print(f"=> Fitting {modelName} model")
            self.tuning_states[current_phase].fit(**kwargs)

      
      def predict(self, modelName: str, current_phase: str):
            assert current_phase in self.tuning_states.keys(), "Current phase must be one of the tuning states"
            print(f"=> Predicting {modelName} model")
            self.tuning_states[current_phase].predict()

