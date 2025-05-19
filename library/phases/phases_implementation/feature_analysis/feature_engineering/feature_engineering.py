
from library.phases.phases_implementation.dataset.dataset import Dataset
from sklearn.preprocessing import PolynomialFeatures

from library.phases.phases_implementation.data_preprocessing.data_preprocessing import Preprocessing

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
        Generates polynomial and interaction features for the dataset.
        
        Parameters
        ----------
        degree : int, optional (default=2)
            The degree of the polynomial features.
        interaction_only : bool, optional (default=False)
            If True, only interaction features are produced: features that are products of 
            at most `degree` distinct input features, without powers of single features.
        standarize : bool, optional (default=False)
            Whether to standardize the resulting features after transformation.
        scaler : str, optional (default="standard")
            Type of scaler to use if `standarize` is True (e.g., "standard" for StandardScaler).
        
        Returns
        -------
        None.  
        Transforms the training, validation, and test feature sets by adding polynomial and interaction terms,
        and optionally scales these new features. Updates the dataset's feature DataFrames in place.
        
        Prints the number of new features added.
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
        
        