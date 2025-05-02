from library.phases.phases_implementation.dataset.dataset import Dataset

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from boruta import BorutaPy

from abc import ABC, abstractmethod

class AutomaticFeatureSelection():
      """
      """
      def __init__(self, dataset: Dataset) -> None:
            self.dataset = dataset
            self.options = {
                  "L1": L1AutomaticFeatureSelection,
                  "Boruta": BorutaAutomaticFeatureSelection
            }
      
      def fit(self, type: str, max_iter: int, delete_features: bool, save_plots: bool = False, save_path: str = ""):
            print(f"Running {type} feature selection")
            return self.options[type](self.dataset).fit(max_iter, delete_features, save_plots, save_path)
      
      def speak(self, message: str):
            print(f"{message} from {id(self)}. You are at automatic feature selection!")


class AutomaticFeatureSelectionFactory(ABC):
      def __init__(self, dataset: Dataset):
            self.dataset = dataset

      @abstractmethod
      def fit(self, max_iter: int, delete_features: bool, save_plots: bool = False, save_path: str = ""):
            pass


class L1AutomaticFeatureSelection(AutomaticFeatureSelectionFactory):
      def __init__(self, dataset: Dataset):
            super().__init__(dataset)

      def fit(self, max_iter: int = 1000, delete_features: bool = True, save_plots: bool = False, save_path: str = ""):
            """
            Automatically selects the features that are most predictive of the target variable using the L1 regularization method

            Parameters
            ----------
            isRegression : bool
                  Whether the model is a regression model
            print_results : bool
                  Whether to print the results

            Returns
            -------
            tuple
            The predictive power features and the excluded features
            """
            if self.dataset.modelTask == "regression":
                  model = Lasso(max_iter=max_iter)
            else:
                  model = LogisticRegression(n_jobs=-1, max_iter=max_iter)

            model.fit(self.dataset.X_train, self.dataset.y_train)
            coefficients = model.coef_ # FOR MULTICLASS RETURNS A LIST OF COEFFICEINTS. CURRENTLY NOT SUPPORTED.

            predictivePowerFeatures = set()
            for i in range(len(coefficients)):
                  if abs(coefficients[i]) > 0:
                        predictivePowerFeatures.add(self.dataset.X_train.columns[i])
                        excludedFeatures = set(self.dataset.X_train.columns) - predictivePowerFeatures
            print(f"Number of predictive power variables: {len(predictivePowerFeatures)}")
            print(f"Number of excluded variables: {len(excludedFeatures)}")
            if delete_features:
                        self.dataset.X_train.drop(columns=excludedFeatures, inplace=True)
                        self.dataset.X_val.drop(columns=excludedFeatures, inplace=True)
                        self.dataset.X_test.drop(columns=excludedFeatures, inplace=True)
            return predictivePowerFeatures, excludedFeatures, coefficients

class BorutaAutomaticFeatureSelection(AutomaticFeatureSelectionFactory):
      def __init__(self, dataset: Dataset):
            super().__init__(dataset)     

      def fit(self, max_iter: int = 100, delete_features: bool = True, save_plots: bool = False, save_path: str = ""):
            """
            Automatically selects the features that are most predictive of the target variable using the Boruta method

            Parameters
            ----------
            boruta_model : BorutaPy
                  The Boruta model

            Returns
            -------
            tuple
            The predictive power features and the excluded features
            """
            RANDOM_STATE = 99
            if self.dataset.modelTask == "regression":
                  rf = RandomForestRegressor(
                  n_estimators=100,    
                  n_jobs=-1, 
                  random_state=RANDOM_STATE
                  )
            else:
                  rf = RandomForestClassifier(
                  n_estimators=100,    
                  n_jobs=-1, 
                  class_weight='balanced',
                  random_state=RANDOM_STATE
                  )
            boruta_model = BorutaPy(rf, 
                                    n_estimators='auto',
                                    verbose=3, 
                                    random_state=RANDOM_STATE, 
                                    max_iter=max_iter,
                                    
                                    )
            boruta_model.fit(self.dataset.X_train.values, 
                              self.dataset.y_train.values)
            selected_mask = boruta_model.support_
            selected_features = set(self.dataset.X_train.columns[selected_mask])
            excludedFeatures = set(self.dataset.X_train.columns) - selected_features
            print(f"Number of predictive power variables: {len(selected_features)}")
            print(f"Number of excluded variables: {len(excludedFeatures)}") 
            if delete_features:     
                  self.dataset.X_train.drop(columns=excludedFeatures, inplace=True)
                  self.dataset.X_val.drop(columns=excludedFeatures, inplace=True)
                  self.dataset.X_test.drop(columns=excludedFeatures, inplace=True)
            return selected_features, excludedFeatures