from abc import ABC, abstractmethod
from library.phases.phases_implementation.dataset.dataset import Dataset

from library.utils.ownModels.neuralNets.utils.earlyStopping import get_early_stopping

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
import kerastuner as kt

from skopt.plots import plot_convergence



class Optimizer():
      def __init__(self, model_sklearn: object, modelName: str, model_object: object, dataset: Dataset, optimizer_type: str, param_grid: dict, max_iter: int = 20, **kwargs):
            """
            Initializes the optimizer class by setting the selected optimizer object.
            """
            assert model_object is not None, "Model object must be provided"
            self.model_sklearn = model_sklearn
            self.modelName = modelName
            self.dataset = dataset
            self.modelObject = model_object
            self.optimizer_type = optimizer_type
            self.optimizer = self._set_up_optimizer(optimizer_type, param_grid, max_iter)
            if model_object.model_type == "neural_network":
                  self.epochs = kwargs.get("epochs", None)

      def _set_up_optimizer(self, type: str, param_grid: dict, max_iter: int = 100):
            if type == "grid":
                  optimizer = GridSearchCV(
                        estimator=self.model_sklearn,
                        param_grid=param_grid,
                        n_iter=max_iter,
                        cv=5,      
                        scoring='r2' if self.dataset.modelTask == "regression" else 'accuracy',
                        verbose=3,
                        random_state=42,
                        n_jobs=1
                  )
            elif type == "random":
                  optimizer = RandomizedSearchCV(
                        estimator=self.model_sklearn,
                        param_distributions=param_grid,
                        n_iter=max_iter,  
                        cv=5,       
                        scoring='r2' if self.dataset.modelTask == "regression" else 'accuracy',
                        verbose=3,
                        random_state=42,
                        n_jobs=1
                  )
            elif type == "bayes": # Usually, you only use this optimizer for neural nets
                  optimizer = BayesSearchCV(
                        estimator=self.model_sklearn,
                        search_spaces=param_grid,
                        n_iter=max_iter,
                        cv=5,
                        scoring='r2' if self.dataset.modelTask == "regression" else 'accuracy',
                        verbose=3,
                        random_state=42,
                        n_jobs=1
                  )
            elif type == "bayes_nn": # Neural nets optimizer needs special treatment
                  assert self.modelObject is not None, "Model object must be provided"
                  print(f"Model object: {self.modelObject.tuning_states['in']}")
                  optimizer = self.modelObject.tuning_states["pre"].model_sklearn.get_tuned_model(max_trials=max_iter,
                                                                                                   executions_per_trial=1,
                                                                                                   directory="results/bayes_opt_results",
                                                                                                   project_name=f"{self.modelName}_bayes_opt")
            else:
                  raise ValueError(f"Invalid optimizer type: {type}")
            return optimizer 
      
      def fit(self):
            """
            Fits the optimize.
            """
            print(f" => STARTING OPTIMIZATION FOR {self.modelName}")
            if self.optimizer_type == "bayes_nn": # Neural nets optimizer needs special treatment
                  self.modelObject.tuning_states["pre"].model_sklearn.tuner_search(self.dataset.X_train, 
                                                                     self.dataset.y_train, 
                                                                     self.dataset.X_val, 
                                                                     self.dataset.y_val)
            else:
                  self.optimizer.fit(self.dataset.X_train, self.dataset.y_train)
            print(f" => FINISHED OPTIMIZATION FOR {self.modelName}")
      
      def plot_convergence(self):
            if self.optimizer_type == "bayes":
                  plot_convergence(self.optimizer.cv_tuner.optimizer_results_)
                  plt.title(f"Convergence Plot for {self.modelName}")
                  plt.gca().get_lines()[0].set_label(self.modelName)
                  plt.legend()

                  
