
from library.phases.dataset.dataset import Dataset
from sklearn.preprocessing import PolynomialFeatures

from library.phases.data_preprocessing.data_preprocessing import DataPreprocessing

import pandas as pd

class FeatureEngineering:
      """
      """
      def __init__(self, dataset: Dataset) -> None:
            self.dataset = dataset

      def polynomial_interaction_effects(self, 
                                         degree: int = 2, 
                                         interaction_only: bool = False,
                                         standarize: bool = False, 
                                         scaler: str = "standard"):
        """
        Computes the polynomial interaction effects of the features.
        """
        original_number_of_features = self.dataset.X_train.shape[1]
        pol_obj = PolynomialFeatures(degree=degree, interaction_only=interaction_only)
        
        x_arr = pol_obj.fit_transform(self.dataset.X_train)
        new_features_name = pol_obj.get_feature_names_out()
        self.dataset.X_train = pd.DataFrame(x_arr, columns=new_features_name)
        x_arr = pol_obj.fit_transform(self.dataset.X_val)
        self.dataset.X_val = pd.DataFrame(x_arr, columns=new_features_name)
        x_arr = pol_obj.fit_transform(self.dataset.X_test)
        self.dataset.X_test = pd.DataFrame(x_arr, columns=new_features_name)

        if standarize:
            preprocessing = Preprocessing(self.dataset)
            preprocessing.scale_features(scaler=scaler, columnsToScale=new_features_name)
        print(f"Added {self.dataset.X_train.shape[1] - original_number_of_features} features")
        
        