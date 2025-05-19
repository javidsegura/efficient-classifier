from efficient_classifier.phases.phases_implementation.dataset.dataset import Dataset
from efficient_classifier.phases.phases_implementation.feature_analysis.feature_transformation.strategies.main import LogStrategy
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import pandas as pd
import numpy as np

class FeatureTransformation:
    """
    """
    _transform_target_strategies = {
        "log": LogStrategy
    }
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def transform_target(self, strategy: str, plot: bool = False):
        transformation_class = self._transform_target_strategies[strategy]
        if transformation_class is None:
            raise ValueError(f"Strategy {strategy} not found")
        transformation_class(self.dataset).transform_target(plot)
    
    def get_cylical_features_encoded(self, features: list[str]) -> pd.DataFrame:
            """Encodes the cyclical features (done before encoding the categorical features)"""
            # Get features to be numerical
            self.dataset.X_train[features] = self.dataset.X_train[features].astype("int")
            self.dataset.X_val[features] = self.dataset.X_val[features].astype("int")
            self.dataset.X_test[features] = self.dataset.X_test[features].astype("int")
            for feature in features:
                  self.dataset.X_train[f"{feature}_sin"] = np.sin((2 * np.pi * self.dataset.X_train[feature]) / 24)
                  self.dataset.X_val[f"{feature}_sin"] = np.sin((2 * np.pi * self.dataset.X_val[feature]) / 24)
                  self.dataset.X_test[f"{feature}_sin"] = np.sin((2 * np.pi * self.dataset.X_test[feature]) / 24)
                  self.dataset.X_train[f"{feature}_cos"] = np.cos((2 * np.pi * self.dataset.X_train[feature]) / 24)
                  self.dataset.X_val[f"{feature}_cos"] = np.cos((2 * np.pi * self.dataset.X_val[feature]) / 24)
                  self.dataset.X_test[f"{feature}_cos"] = np.cos((2 * np.pi * self.dataset.X_test[feature]) / 24)
                  self.dataset.X_train.drop(columns=[feature], inplace=True)
                  self.dataset.X_val.drop(columns=[feature], inplace=True)
                  self.dataset.X_test.drop(columns=[feature], inplace=True)
    
    def get_categorical_features_encoded(self, 
                                          features: list[str],
                                          encode_y: bool = True
                                          ) -> None | dict:
      """
      Encodes the categorical features for the training, validation and test sets

      Parameters
      ----------
        features : list[str]
            The features to encode
        encode_y : bool
            Whether to encode the target variable
      Returns
      -------
        X_train_encoded : pd.DataFrame
            The training set
        X_val_encoded : pd.DataFrame
            The validation set
        X_test_encoded : pd.DataFrame
            The test set
        y_train_encoded : pd.Series
            The training set
        y_val_encoded : pd.Series
            The validation set
        y_test_encoded : pd.Series
            The test set
        encoding_map : dict
            The encoding map
      """
      encoder = OneHotEncoder(handle_unknown="ignore", 
                        sparse_output=False,
                        dtype=int,
                        drop="first"
                        )
      # Training set
      encoded_array = encoder.fit_transform(self.dataset.X_train[features])
      encoded_cols = encoder.get_feature_names_out(features)
      train_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=self.dataset.X_train.index)
      X_train_encoded = self.dataset.X_train.drop(features, axis=1).join(train_encoded)
      # Validation set
      encoded_array_val = encoder.transform(self.dataset.X_val[features])
      val_encoded = pd.DataFrame(encoded_array_val, columns=encoded_cols, index=self.dataset.X_val.index)
      X_val_encoded = self.dataset.X_val.drop(features, axis=1).join(val_encoded)
      # Test set
      encoded_array_test = encoder.transform(self.dataset.X_test[features])
      test_encoded = pd.DataFrame(encoded_array_test, columns=encoded_cols, index=self.dataset.X_test.index)
      X_test_encoded = self.dataset.X_test.drop(features, axis=1).join(test_encoded)
      self.dataset.X_train, self.dataset.X_val, self.dataset.X_test = X_train_encoded, X_val_encoded, X_test_encoded
      self.isXencoded = True

      if encode_y:
        labeller = LabelEncoder()
        labeller.fit(self.dataset.y_train)
        y_train_encoded = pd.Series(labeller.transform(self.dataset.y_train), index=self.dataset.y_train.index)
        y_val_encoded = pd.Series(labeller.transform(self.dataset.y_val), index=self.dataset.y_val.index)
        y_test_encoded = pd.Series(labeller.transform(self.dataset.y_test), index=self.dataset.y_test.index)

        encoding_map = dict(zip(labeller.classes_, range(len(labeller.classes_))))
        self.dataset.y_train, self.dataset.y_val, self.dataset.y_test, self.encoding_map = y_train_encoded, y_val_encoded, y_test_encoded, encoding_map
        self.isYencoded = True
        return encoding_map
