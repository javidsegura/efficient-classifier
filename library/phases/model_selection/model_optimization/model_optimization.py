
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
            self.optimizer_method = OptimizationMethod(model_sklearn, modelName, dataset)
            self.optimizer_method._set_up_optimizer(optimizer_type, param_grid, max_iter)
      
      def start_optimization(self):
            return self.optimizer_method.fit()

class OptimizationMethod():
      def __init__(self, model_sklearn: object, modelName: str, dataset: Dataset):
            self.model_sklearn = model_sklearn
            self.modelName = modelName
            self.dataset = dataset

      def _set_up_optimizer(self, type: str, param_grid: dict, max_iter: int = 100):
            if type == "grid":
                 self.opt = GridSearchCV(
                        estimator=self.model_sklearn,
                        param_grid=param_grid,
                        n_iter=max_iter,
                        cv=5,       # 5-fold cross-validation
                        scoring='r2' if self.dataset.modelTask == "regression" else 'accuracy',
                        verbose=3,
                        random_state=42,
                        n_jobs=-1
                  )
            elif type == "random":
                  self.opt = RandomizedSearchCV(
                        estimator=self.model_sklearn,
                        param_distributions=param_grid,
                        n_iter=max_iter,  # number of random combinations to try
                        cv=5,       # 5-fold cross-validation
                        scoring='r2' if self.dataset.modelTask == "regression" else 'accuracy',
                        verbose=3,
                        random_state=42,
                        n_jobs=-1
                  )
            elif type == "bayes":
                  self.opt = BayesSearchCV(
                        estimator=self.model_sklearn,
                        search_spaces=param_grid,
                        n_iter=max_iter,
                        cv=5,
                        scoring='r2' if self.dataset.modelTask == "regression" else 'accuracy',
                        verbose=3,
                        random_state=42,
                        n_jobs=-1
                  )
      def fit(self):
            print(f" => STARTING OPTIMIZATION FOR {self.modelName}")
            self.opt.fit(self.dataset.X_train, self.dataset.y_train)
            print(f" => FINISHED OPTIMIZATION FOR {self.modelName}")
            best_params = self.opt.best_params_
            best_model = self.opt.best_estimator_
            print(f"\tBest parameters: {best_params}")
            print(f"\tBest model: {best_model}")
            y_pred = best_model.predict(self.dataset.X_val)
            print(f"\t\t Going to calculate feature importance...")
      
            return pd.DataFrame(self.opt.cv_results_), best_model, y_pred
