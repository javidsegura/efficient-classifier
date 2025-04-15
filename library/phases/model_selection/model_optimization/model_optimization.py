
from abc import ABC, abstractmethod
from library.phases.dataset.dataset import Dataset

import pandas as pd

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV


class Optimizer():
      def __init__(self, model_sklearn: object, modelName: str, dataset: Dataset, optimizer_type: str, param_grid: dict, max_iter: int = 20):
            self.model_sklearn = model_sklearn
            self.modelName = modelName
            self.dataset = dataset
            self.cv_tuner = self._set_up_optimizer(optimizer_type, param_grid, max_iter)
           
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
            elif type == "bayes":
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
            return optimizer 
      
      def fit(self):
            print(f" => STARTING OPTIMIZATION FOR {self.modelName}")
            self.cv_tuner.fit(self.dataset.X_train, self.dataset.y_train)
            print(f" => FINISHED OPTIMIZATION FOR {self.modelName}")
