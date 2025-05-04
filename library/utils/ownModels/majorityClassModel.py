import pandas as pd


class MajorityClassClassifier:
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
      
      def get_params(self):
            return {} # No hyperparameters to tune
      
      
      