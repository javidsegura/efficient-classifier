import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class MajorityClassClassifier(BaseEstimator, ClassifierMixin):
      """
      Simulates a sklearn model object (with the corresponding methods) that computes always the most common class as prediction
      """
      def __init__(self):
            self.most_common_class = None

      def fit(self, X_data, y_data):
           if not isinstance(y_data, pd.Series):
                y_data = pd.Series(y_data)
           most_common_class = y_data.mode()[0]
           self.most_common_class = most_common_class
           self.is_fitted_ = True # Needed for sklearn compatibility
           return self

      def predict(self, X_data):
           return [self.most_common_class] * len(X_data)
      
      def get_params(self, deep=True):
            return {} # No hyperparameters to tune
      
      def predict_proba(self, X_data):
            return np.array([[1 if y == self.most_common_class else 0 for y in y_data] for y_data in X_data])
      
      
      