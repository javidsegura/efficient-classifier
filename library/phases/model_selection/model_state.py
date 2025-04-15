from abc import ABC, abstractmethod
import numpy as np
import time
from library.phases.dataset.dataset import Dataset
from library.phases.model_selection.model_optimization.model_optimization import Optimizer


"""

Assesment currently has the following structure:
- `id`: NoneType
- `timeStamp`: NoneType
- `comments`: NoneType
- `modelName`: str
- `status`: str
- `features_used`: NoneType
- `hyperParameters`: NoneType
- `timeToFit`: float
- `timeToPredict`: float
- `accuracy`: float
- `precision`: float
- `recall`: float
- `f1-score`: float
- `predictions_val`: numpy.ndarray
- `precictions_train`: numpy.ndarray
- `predictions_test`: numpy.ndarray
- `model_sklearn`: sklearn

"""


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
      
      @abstractmethod
      def fit(self):
            pass
      
      @abstractmethod
      def predict(self, is_training: bool = False):
            pass
                
      

class PreTuningState(ModelState):
      def __init__(self, model_sklearn: object, modelName: str, dataset: Dataset, results_header: list[str]):
            super().__init__(model_sklearn, modelName, dataset, results_header)
      
      def get_fit_data(self):
            return self.dataset.X_train, self.dataset.y_train

      def get_predict_data(self):
            return {
                   "training":self.dataset.X_train,
                   "not-training": self.dataset.X_val
                   }
      
      def fit(self, **kwargs):
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
                  data = self.get_predict_data()

                  # Predict training data
                  training_data = data["training"]
                  self.assesment["predictions_train"] = self.model_sklearn.predict(training_data)

                  # Predict not training data
                  not_training_data = data["not-training"]
                  self.assesment["predictions_val"] = self.model_sklearn.predict(not_training_data)

                  end_time = time.time()
                  time_taken = end_time - start_time
                  self.assesment["timeToPredict"] = time_taken
                  print(f"\t\t => Predicted {self.modelName}. Took {time_taken} seconds")
      

class InTuningState(ModelState):
      def __init__(self, model_sklearn: object, modelName: str, dataset: Dataset, results_header: list[str]):
            super().__init__(model_sklearn, modelName, dataset, results_header)
      
      def get_fit_data(self):
            return self.dataset.X_train, self.dataset.y_train

      def get_predict_data(self):
            return {
                   "training":self.dataset.X_train,
                   "not-training": self.dataset.X_val
                   }
      
      def fit(self, **kwargs):
                  param_grid = kwargs.get("param_grid", None)
                  max_iter = kwargs.get("max_iter", None)
                  optimizer_type = kwargs.get("optimizer_type", None)
                  assert optimizer_type in ["grid", "random", "bayes"], "Optimizer type must be one of the following: grid, random, bayes"
                  assert param_grid is not None, "Param grid must be provided"
                  assert max_iter is not None, "Max iter must be provided"

                  optimizer = Optimizer(self.model_sklearn, self.modelName, self.dataset, optimizer_type, param_grid, max_iter)
                  optimizer.fit()
                  self.cv_tuner = optimizer.cv_tuner
                  self.model_sklearn = optimizer.cv_tuner.best_estimator_
                  self.assesment["model_sklearn"] = self.model_sklearn
      
      def predict(self):
                  start_time = time.time()
                  print(f"!> Started predicting {self.modelName}")
                  data = self.get_predict_data()

                  # Predict training data
                  training_data = data["training"]
                  self.assesment["predictions_train"] = self.model_sklearn.predict(training_data)

                  # Predict not training data
                  not_training_data = data["not-training"]
                  self.assesment["predictions_val"] = self.model_sklearn.predict(not_training_data)

                  end_time = time.time()
                  time_taken = end_time - start_time
                  self.assesment["timeToPredict"] = time_taken
                  print(f"\t\t => Predicted {self.modelName}. Took {time_taken} seconds")







class PostTuningState(ModelState):
      def __init__(self, model_sklearn: object, modelName: str, dataset: Dataset, results_header: list[str]):
            super().__init__(model_sklearn, modelName, dataset, results_header)
      
      def get_fit_data(self): 
            X_train_combined = np.vstack([self.dataset.X_train, self.dataset.X_val])
            y_train_combined = np.concatenate([self.dataset.y_train, self.dataset.y_val])
            return X_train_combined, y_train_combined
      
      def get_predict_data(self):
            return self.dataset.X_test
      
      def fit(self, **kwargs):
            pass
      
      def predict(self, **kwargs):
            pass
      
      