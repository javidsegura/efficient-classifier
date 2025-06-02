from abc import ABC, abstractmethod
import numpy as np
import time
from efficient_classifier.phases.phases_implementation.dataset.dataset import Dataset
from efficient_classifier.phases.phases_implementation.modelling.shallow.model_optimization.model_optimization import Optimizer

from efficient_classifier.utils.ownModels.neuralNets.feedForward import FeedForwardNeuralNetwork

from efficient_classifier.utils.decorators.timer import timer
from sklearn.calibration import CalibratedClassifierCV

class ModelState(ABC):
      def __init__(self, model_sklearn: object, modelName: str, model_type: str, dataset: Dataset, results_header: list[str], variables: dict):
            """
            This is the base class for all the model **states**.

            Parameters
            ----------
            model_sklearn : object
                  The model to be used
            modelName : str
            
            """
            self.model_sklearn = model_sklearn
            self.modelName = modelName
            self.model_type = model_type
            self.dataset = dataset
            self.assesment = {column_name: None for column_name in results_header}
            self.assesment["modelName"] = modelName
            self.variables = variables

      
      @abstractmethod
      def get_fit_data(self):
            """
            Varies over each state (in post its training + val for instance)
            """
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
      def __init__(self, model_sklearn: object, modelName: str, model_type: str, dataset: Dataset, results_header: list[str], variables: dict):
            super().__init__(model_sklearn, modelName, model_type, dataset, results_header, variables)
      
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
                  training_samples_length = X_data.shape[0]
                  target_samples_length = y_data.shape[0]
                  assert training_samples_length == target_samples_length, f"Training samples: {training_samples_length} and target samples: {target_samples_length}"
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
      def __init__(self, model_sklearn: object, modelName: str, dataset: Dataset, results_header: list[str], model_type: str = "classical", variables: dict = None):
            super().__init__(model_sklearn, modelName, dataset, results_header, model_type, variables)
      
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
                  print(f"1) Model sklearn: {self.model_sklearn} for {self.modelName}")
                  if self.variables["phase_runners"]["modelling_runner"]["calibration"]["calibrate_models"]:
                        print(dir(self.model_sklearn))
                        self.model_sklearn = self.model_sklearn.estimator
                  print(f"2) Model sklearn: {self.model_sklearn} for {self.modelName}")
                  assert self.model_type is not None, f"Model object must have a model_type. {self.modelName}. Model object: {model_object}"

                  if self.model_type == "neural_network":
                              epochs = kwargs.get("epochs", None)
                  else:
                              epochs = None
                  assert optimizer_type in ["grid", "random", "bayes", "bayes_neural_network"], "Optimizer type must be one of the following: grid, random, bayes, bayes_neural_network"
                  assert max_iter is not None, "Max iter must be provided"
                  assert model_object is not None, "Model object must be provided"
                  print(f"Model object: {model_object}")

                  if self.model_type == "neural_network":
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
                  if optimizer_type != "bayes_neural_network":
                              self.model_sklearn = self.optimizer.optimizer.best_estimator_
                              if self.variables["phase_runners"]["modelling_runner"]["calibration"]["calibrate_models"]:
                                    model = CalibratedClassifierCV(self.model_sklearn)
                                    model.fit(self.dataset.X_train, self.dataset.y_train)
                                    self.assesment["model_sklearn"] = model
                              else:
                                    self.assesment["model_sklearn"] = self.model_sklearn
                  else:
                        best_model = self.optimizer.optimizer.get_best_models(num_models=1)[0]
                        best_hps = self.optimizer.optimizer.get_best_hyperparameters(num_trials=1)[0]
                        best_params = best_hps.values
                        n_layers = best_params["n_layers"]
                        learning_rate = best_params["learning_rate"]
                        units_per_layers = []
                        activations = []
                        for i in range(n_layers):
                              units_per_layers.append(best_params[f"units_{i}"])
                              activations.append(best_params[f"act_{i}"])

                        print(f"Best params: {best_params}")
                        self.model_sklearn = FeedForwardNeuralNetwork(
                                                                  num_features=self.dataset.X_train.shape[1], 
                                                                  num_classes=self.dataset.y_train.value_counts().shape[0], 
                                                                  n_layers=n_layers,
                                                                  units_per_layer=units_per_layers,
                                                                  activations=activations,
                                                                  learning_rate=learning_rate)
                        self.model_sklearn.model = best_model
                        if self.variables["phase_runners"]["modelling_runner"]["calibration"]["calibrate_models"]:
                              model = CalibratedClassifierCV(self.model_sklearn)
                              model.fit(self.dataset.X_train, self.dataset.y_train)
                              self.assesment["model_sklearn"] = model
                        else:
                              self.assesment["model_sklearn"] = self.model_sklearn
                        self.model_sklearn.is_fitted_ = True
      
      def predict(self):
                  if self.model_type == "stacking":
                        print(f"ESTIMTORS AT PREDICTION ARE: {self.model_sklearn.estimators}")
                  
                  data = self.get_predict_data()

                  start_time = time.time()
                  print(f"!> Started predicting {self.modelName}")

                  # Predict training data
                  print(f"Predicting training data")
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
      def __init__(self, model_sklearn: object, modelName: str, dataset: Dataset, results_header: list[str], model_type: str, variables: dict = None):
            super().__init__(model_sklearn, modelName, dataset, results_header, model_type, variables)
      
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
            assert training_data.shape[0] == self.X_train_combined.shape[0] == self.y_train_combined.shape[0], f"Training data shape: {training_data.shape} does not match X_train_combined shape: {self.X_train_combined.shape} or y_train_combined shape: {self.y_train_combined.shape}"
            prediction_train = self.model_sklearn.predict(training_data)
            assert len(prediction_train) == self.y_train_combined.shape[0], f"Prediction train shape: {prediction_train.shape} does not match y_train_combined shape: {self.y_train_combined.shape}"
            self.assesment["predictions_train"] = prediction_train

            # Predict not training data
            not_training_data = data["not-training"]
            self.assesment["predictions_test"] = self.model_sklearn.predict(not_training_data)

            end_time = time.time()
            time_taken = end_time - start_time
            self.assesment["timeToPredict"] = time_taken
            print(f"\t\t => Predicted {self.modelName}. Took {time_taken} seconds")
      
      