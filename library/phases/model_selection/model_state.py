from abc import ABC, abstractmethod
import numpy as np
import time
from library.phases.dataset.dataset import Dataset


class ModelState(ABC):
      def __init__(self, model_sklearn: object, modelName: str, dataset: Dataset, results_header: list[str]):
            self.model_sklearn = model_sklearn
            self.modelName = modelName
            self.dataset = dataset
            self.assesment = {column_name: None for column_name in results_header}
            self.assesment["modelName"] = modelName
      
      @abstractmethod
      def get_fit_data(self):
            pass
      
      @abstractmethod
      def get_predict_data(self):
            pass
      
      def fit(self):
                  print(f"Sklearn model: {self.model_sklearn}")
                  start_time = time.time()
                  print(f"!> Started fitting {self.modelName}")
                  X_data, y_data = self.get_fit_data()
                  print(f"Lenght of X_data: {X_data.shape[0]}")
                  self.assesment["model_sklearn"] = self.model_sklearn.fit(X_data, y_data)
                  end_time = time.time()
                  time_taken = end_time - start_time
                  self.assesment["timeToFit"] = time_taken
                  print(f"\t\t => Fitted {self.modelName}. Took {time_taken} seconds")
      
      def predict(self):
                  start_time = time.time()
                  print(f"!> Started predicting {self.modelName}")
                  X_data = self.get_predict_data()
                  self.assesment["predictions"] = self.model_sklearn.predict(X_data)
                  end_time = time.time()
                  time_taken = end_time - start_time
                  self.assesment["timeToMakePredictions"] = time_taken
                  print(f"\t\t => Predicted {self.modelName}. Took {time_taken} seconds")
      
      def predict_training(self):
            X_data, y_data = self.get_fit_data()
            self.assesment["predictions_train"] = self.model_sklearn.predict(X_data)

      def store_assesment(self, metrics: dict[str, float], **kwargs):
            if self.dataset.modelTask == "classification":
                  conf_matrix = kwargs.get("conf_matrix", None)
                  if conf_matrix is not None:
                        self.assesment["conf_matrix"] = conf_matrix
            for metric, value in metrics.items():
                  self.assesment[metric] = value
            print(f"Metrics stored in assesment")

class PreTuningState(ModelState):
      def __init__(self, model_sklearn: object, modelName: str, dataset: Dataset, results_header: list[str]):
            super().__init__(model_sklearn, modelName, dataset, results_header)
            self.assesment["status"] = "pre_tuning"
      
      def get_fit_data(self):
            return self.dataset.X_train, self.dataset.y_train

      def get_predict_data(self):
            return self.dataset.X_val

class InTuningState(ModelState):
      def __init__(self, model_sklearn: object, modelName: str, dataset: Dataset, results_header: list[str]):
            super().__init__(model_sklearn, modelName, dataset, results_header)
            self.assesment["status"] = "in_tuning"
      
      def get_fit_data(self):
            return self.dataset.X_train, self.dataset.y_train

      def get_predict_data(self):
            return self.dataset.X_val


class PostTuningState(ModelState):
      def __init__(self, model_sklearn: object, modelName: str, dataset: Dataset, results_header: list[str]):
            super().__init__(model_sklearn, modelName, dataset, results_header)
            self.assesment["status"] = "post_tuning"
      
      def get_fit_data(self): # THIS NEEDS TO BE CHANGES (MERGED WITH VAL SET!!!)
            X_train_combined = np.vstack([self.dataset.X_train, self.dataset.X_val])
            y_train_combined = np.concatenate([self.dataset.y_train, self.dataset.y_val])
            return X_train_combined, y_train_combined
      
      def get_predict_data(self):
            return self.dataset.X_test