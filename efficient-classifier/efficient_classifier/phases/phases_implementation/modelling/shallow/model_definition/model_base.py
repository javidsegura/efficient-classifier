from abc import ABC, abstractmethod

import time
from efficient_classifier.phases.phases_implementation.dataset.dataset import Dataset
import matplotlib.pyplot as plt
import pandas as pd

from efficient_classifier.phases.phases_implementation.modelling.shallow.model_definition.model_states.model_state import PreTuningState, PostTuningState, InTuningState

class Model(ABC):
      def __init__(self, modelName: str, model_sklearn: object, model_type: str, results_header: list[str], dataset: Dataset, variables: dict):
            """
            This is the base class for all the model objects. It initializes the differeent tuning states and defines the fitting and predicitng methods for those states
            """
            assert model_type in ["classical", "neural_network", "stacking"], "Model type must be one of the following: classical, neural_network"
            assert model_sklearn is not None, "Model sklearn must be provided"
            self.dataset = dataset
            self.modelName = modelName
            self.model_sklearn = model_sklearn
            self.model_type = model_type
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
                  "pre": PreTuningState(model_sklearn, modelName, model_type, dataset, self.results_header, variables),
                  "in": InTuningState(model_sklearn, modelName, model_type, dataset, self.results_header, variables),
                  "post": PostTuningState(model_sklearn, modelName, model_type, dataset, self.results_header, variables)
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

