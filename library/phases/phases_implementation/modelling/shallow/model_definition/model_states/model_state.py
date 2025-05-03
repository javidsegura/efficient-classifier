from abc import ABC, abstractmethod
import numpy as np
import time
from library.phases.phases_implementation.dataset.dataset import Dataset
from library.phases.phases_implementation.modelling.shallow.model_optimization.model_optimization import Optimizer

from library.utils.ownModels.neuralNets.feedForward import FeedForwardNeuralNetwork

from library.utils.decorators.timer import timer


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
      def __init__(self, model_sklearn: object, modelName: str, model_type: str, dataset: Dataset, results_header: list[str]):
            self.model_sklearn = model_sklearn
            self.modelName = modelName
            self.model_type = model_type
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
      def __init__(self, model_sklearn: object, modelName: str, model_type: str, dataset: Dataset, results_header: list[str]):
            super().__init__(model_sklearn, modelName, model_type, dataset, results_header)
      
      def get_fit_data(self):
            if self.model_type == "neural_network":
                  return self.dataset.X_train, self.dataset.y_train, self.dataset.X_val, self.dataset.y_val
            else:
                  return self.dataset.X_train, self.dataset.y_train

      def get_predict_data(self):
            return {
                   "training":self.dataset.X_train,
                   "not-training": self.dataset.X_val
                   }
      
      def fit(self):
                  print(f"Sklearn model: {self.model_sklearn}")
                  start_time = time.time()
                  print(f"!> Started fitting {self.modelName}")
                  if self.model_type == "neural_network":
                              X_data, y_data, X_val, y_val = self.get_fit_data()
                  else:
                              X_data, y_data = self.get_fit_data()
                  print(f"Lenght of X_data: {X_data.shape[0]}")
                  if self.model_type == "neural_network":
                              self.assesment["model_sklearn"] = self.model_sklearn.fit(X_data, y_data, X_val=X_val, y_val=y_val, isOptimizedVersion=False)
                  else:
                              self.assesment["model_sklearn"] = self.model_sklearn.fit(X_data, y_data)
                  end_time = time.time()
                  time_taken = end_time - start_time
                  self.assesment["timeToFit"] = time_taken
                  print(f"\t\t => Fitted {self.modelName}. Took {time_taken} seconds")
  
      
      def predict(self):
                  data = self.get_predict_data()

                  start_time = time.time()
                  print(f"!> Started predicting {self.modelName}")

                  # Predict training data
                  training_data = data["training"]
                  print(f"Training data: {training_data.shape}")
                  self.assesment["predictions_train"] = self.model_sklearn.predict(training_data)

                  # Predict not training data
                  not_training_data = data["not-training"]
                  print(f"Not training data: {not_training_data.shape}")
                  self.assesment["predictions_val"] = self.model_sklearn.predict(not_training_data)

                  end_time = time.time()
                  time_taken = end_time - start_time
                  self.assesment["timeToPredict"] = time_taken
                  print(f"\t\t => Predicted {self.modelName}. Took {time_taken} seconds")

      

class InTuningState(ModelState):
      def __init__(self, model_sklearn: object, modelName: str, dataset: Dataset, results_header: list[str], model_type: str = "classical"):
            super().__init__(model_sklearn, modelName, dataset, results_header, model_type)
      
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
                  model_object = kwargs.get("model_object", None)
                  if model_object.model_type == "neural_network":
                        epochs = kwargs.get("epochs", None)
                  else:
                        epochs = None
                  assert optimizer_type in ["grid", "random", "bayes", "bayes_nn"], "Optimizer type must be one of the following: grid, random, bayes, bayes_nn"
                  assert max_iter is not None, "Max iter must be provided"
                  assert model_object is not None, "Model object must be provided"
                  print(f"Model object: {model_object}")

                  if model_object.model_type == "neural_network":
                        self.optimizer = Optimizer(
                                                model_sklearn=self.model_sklearn,
                                                modelName=self.modelName, 
                                                model_object=model_object,
                                                dataset=self.dataset,
                                                optimizer_type=optimizer_type, 
                                                param_grid=param_grid,
                                                max_iter=max_iter,
                                                epochs=epochs)
                  else:
                        self.optimizer = Optimizer(
                                                model_sklearn=self.model_sklearn,
                                                modelName=self.modelName, 
                                                model_object=model_object,
                                                dataset=self.dataset,
                                                optimizer_type=optimizer_type, 
                                                param_grid=param_grid,
                                                max_iter=max_iter)
                  time_start = time.time()
                  self.optimizer.fit()
                  time_end = time.time()
                  time_taken = time_end - time_start
                  self.assesment["timeToFit"] = time_taken
                  if optimizer_type != "bayes_nn":
                        self.model_sklearn = self.optimizer.optimizer.best_estimator_
                        self.assesment["model_sklearn"] = self.model_sklearn
                  else:
                        model_keras = self.optimizer.optimizer.get_best_models(num_models=1)[0]
                        self.model_sklearn = FeedForwardNeuralNetwork(num_features=self.dataset.X_train.shape[1], 
                                                               num_classes=self.dataset.y_train.value_counts().shape[0], 
                                                               model_keras=model_keras)
                        self.model_sklearn.is_fitted_ = True
                                                               
                        self.assesment["model_sklearn"] = self.model_sklearn
      
      def predict(self):
                  start_time = time.time()
                  print(f"!> Started predicting {self.modelName}")
                  data = self.get_predict_data()

                  # Predict training data
                  print(f"Predicting training data")
                  print(f"model_sklearn: {self.model_sklearn}")
                  print(f"dir of model_sklearn: {dir(self.model_sklearn)}")
                  training_data = data["training"]
                  self.assesment["predictions_train"] = self.model_sklearn.predict(training_data)

                  # Predict not training data
                  print(f"Predicting not training data")
                  not_training_data = data["not-training"]
                  self.assesment["predictions_val"] = self.model_sklearn.predict(not_training_data)

                  end_time = time.time()
                  time_taken = end_time - start_time
                  self.assesment["timeToPredict"] = time_taken
                  print(f"\t\t => Predicted {self.modelName}. Took {time_taken} seconds")
      
      def plot_convergence(self):
            self.optimizer.plot_convergence()




class PostTuningState(ModelState):
      def __init__(self, model_sklearn: object, modelName: str, dataset: Dataset, results_header: list[str], model_type: str = "classical"):
            super().__init__(model_sklearn, modelName, dataset, results_header, model_type)
            """ model object needs to be overwritten!!!"""
      
      def get_fit_data(self): 
            self.X_train_combined = np.vstack([self.dataset.X_train, self.dataset.X_val])
            self.y_train_combined = np.concatenate([self.dataset.y_train, self.dataset.y_val])
            print(f"X_train_combined: {self.X_train_combined.shape}")
            return self.X_train_combined, self.y_train_combined
      
      def get_predict_data(self):
            return {
                   "training": self.X_train_combined,
                   "not-training": self.dataset.X_test
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
            self.assesment["predictions_test"] = self.model_sklearn.predict(not_training_data)

            end_time = time.time()
            time_taken = end_time - start_time
            self.assesment["timeToPredict"] = time_taken
            print(f"\t\t => Predicted {self.modelName}. Took {time_taken} seconds")
      
      