

from library.dataset import Dataset

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures

from imblearn.over_sampling import SMOTENC
from boruta import BorutaPy
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import  LogisticRegression, Lasso

import matplotlib.pyplot as plt


class FeatureEngineering:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __calculate_vif(self):
        """
        Calculates the VIF of the features.

        Returns
        -------
        pd.DataFrame
            A dataframe with the features and their VIF.
        """
        vif_data = pd.DataFrame()
        only_numerical_features = self.dataset.X_train_encoded.select_dtypes(include=["number"]) if self.dataset.isXencoded else self.dataset.X_train.select_dtypes(include=["number"])
        vif_data["Feature"] = only_numerical_features.columns
        vif_data["VIF"] = [variance_inflation_factor(only_numerical_features.values, i) for i in range(len(only_numerical_features.columns))]
        return vif_data
    
    def start_vif_elimination(self, threshold=10, delete=True):
        """
        Starts the VIF elimination process. Eliminates in all sets.
        Note: this is computationally expensive for high-feature datasets.

        Parameters
        ----------
        threshold : float
            The threshold for the VIF.

        Returns
        -------
        None
        """
        number_of_iterations = 0
        while True:
            number_of_iterations += 1
            vif_data = self.__calculate_vif()
            print(f"VIF computed for iteration {number_of_iterations}:")
            max_vif = vif_data["VIF"].max()
            if max_vif < threshold:
                  break
            feature_to_drop = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
            if delete: 
                if self.dataset.isXencoded:
                    self.dataset.X_train_encoded.drop(columns=[feature_to_drop], inplace=True)
                    self.dataset.X_val_encoded.drop(columns=[feature_to_drop], inplace=True)
                    self.dataset.X_test_encoded.drop(columns=[feature_to_drop], inplace=True)
                else:
                    self.dataset.X_train.drop(columns=[feature_to_drop], inplace=True)
                    self.dataset.X_val.drop(columns=[feature_to_drop], inplace=True)
                    self.dataset.X_test.drop(columns=[feature_to_drop], inplace=True)
                print(f"\tDropped: '{feature_to_drop}' with a VIF of {max_vif}")
            else:
                print(f"Feature with highest VIF: '{feature_to_drop}' with VIF: {max_vif}")
                break

    def remove_constant_features(self):
        """
        Removes constant features from the dataset.
        """
        original_number_of_features = self.dataset.df.shape[1]
        zero_variance_features = self.dataset.df.select_dtypes(include='number').std() == 0
        if zero_variance_features.any():
            print("Zero-variance features found:")
            print(zero_variance_features[zero_variance_features].index)
            self.dataset.df.drop(columns=zero_variance_features[zero_variance_features].index, inplace=True)
            print(f"Removed {original_number_of_features - self.dataset.df.shape[1]} features with zero variance")
        else:
            print("No zero-variance features found.")
      
    def compute_feature_relevance(self, feature: str):
        """
        Computes the relevance of the features.
        """
        mutual_info_train = mutual_info_regression(self.dataset.X_train_encoded[[feature]] if self.dataset.isXencoded else self.dataset.X_train[[feature]], self.dataset.y_train)
        return mutual_info_train[0]
    
    def polynomial_interaction_effects(self, degree: int = 2, interaction_only: bool = False):
        """
        Computes the polynomial interaction effects of the features.
        """
        original_number_of_features = self.dataset.X_train.shape[1]
        pol_obj = PolynomialFeatures(degree=degree, interaction_only=interaction_only)
        x_arr = pol_obj.fit_transform(self.dataset.X_train)
        self.dataset.X_train = pd.DataFrame(x_arr, columns=pol_obj.get_feature_names_out())
        x_arr = pol_obj.fit_transform(self.dataset.X_val)
        self.dataset.X_val = pd.DataFrame(x_arr, columns=pol_obj.get_feature_names_out())
        x_arr = pol_obj.fit_transform(self.dataset.X_test)
        self.dataset.X_test = pd.DataFrame(x_arr, columns=pol_obj.get_feature_names_out())
        print(f"Added {self.dataset.X_train.shape[1] - original_number_of_features} features")
    
    def l1_automatic_feature_selection(self, max_iter: int = 1000, print_results: bool = True, **kwargs):
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
        if self.dataset.task == "regression":
            model = Lasso(max_iter=max_iter, **kwargs)
        else:
            model = LogisticRegression(n_jobs=-1, max_iter=max_iter, **kwargs)

        model.fit(self.dataset.X_train_encoded if self.dataset.isXencoded else self.dataset.X_train, self.dataset.y_train_encoded if self.dataset.isYencoded else self.dataset.y_train)
        coefficients = model.coef_

        predictivePowerFeatures = set()
        for i in range(len(coefficients)):
            if abs(coefficients[i]) > 0:
                predictivePowerFeatures.add(self.dataset.X_train_encoded.columns[i] if self.dataset.isXencoded else self.dataset.X_train.columns[i])
            excludedFeatures = set(self.dataset.X_train_encoded.columns if self.dataset.isXencoded else self.dataset.X_train.columns) - predictivePowerFeatures
        if print_results:
            print(f"Number of predictive power variables: {len(predictivePowerFeatures)}")
            print(f"Number of excluded variables: {len(excludedFeatures)}")
        return predictivePowerFeatures, excludedFeatures, coefficients

    def boruta_automatic_feature_selection(self, max_iter: int = 100, print_results: bool = True, delete_features: bool = True):
        """
        Automatically selects the features that are most predictive of the target variable using the Boruta method

        Parameters
        ----------
        boruta_model : BorutaPy
            The Boruta model
        print_results : bool
            Whether to print the results

        Returns
        -------
        tuple
        The predictive power features and the excluded features
        """
        RANDOM_STATE = 99
        if self.dataset.task == "regression":
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
        boruta_model.fit(self.dataset.X_train_encoded.values if self.dataset.isXencoded else self.dataset.X_train.values, 
                            self.dataset.y_train_encoded.values if self.dataset.isYencoded else self.dataset.y_train.values)
        selected_mask = boruta_model.support_
        selected_features = set(self.dataset.X_train_encoded.columns if self.dataset.isXencoded else self.dataset.X_train.columns[selected_mask])
        excludedFeatures = set(self.dataset.X_train_encoded.columns if self.dataset.isXencoded else self.dataset.X_train.columns) - selected_features
        if print_results:
            print(f"Number of predictive power variables: {len(selected_features)}")
            print(f"Number of excluded variables: {len(excludedFeatures)}") 
        if delete_features:
            self.dataset.X_train_encoded.drop(columns=excludedFeatures, inplace=True)
            self.dataset.X_val_encoded.drop(columns=excludedFeatures, inplace=True)
            self.dataset.X_test_encoded.drop(columns=excludedFeatures, inplace=True)
        return selected_features, excludedFeatures
    
    def remove_low_variance_features(self, threshold: float = 0.01, plot: bool = True):
        """
        Removes the features with low variance.
        """
        
        # Create dataframe with feature and standard deviation (spreadness)
        spreadness_df = pd.DataFrame({
            "feature": self.dataset.X_train.columns if not self.dataset.isXencoded else self.dataset.X_train_encoded.columns,
            "spreadness": self.dataset.X_train.std() if not self.dataset.isXencoded else self.dataset.X_train_encoded.std()
        }).reset_index(drop=True)
        if plot:
            plt.figure(figsize=(12, 8))
            plt.hist(spreadness_df["spreadness"], bins=30, edgecolor='black')
            plt.title('Distribution of Standard Deviations for Numeric Features')
            plt.xlabel('Standard Deviation')
            plt.ylabel('Frequency')
            plt.show()
        columns_to_drop = spreadness_df[spreadness_df["spreadness"] < threshold]["feature"].tolist()
        if not self.dataset.isXencoded:
            self.dataset.X_train.drop(columns=columns_to_drop, inplace=True)
            self.dataset.X_val.drop(columns=columns_to_drop, inplace=True)
            self.dataset.X_test.drop(columns=columns_to_drop, inplace=True)
        else:
            self.dataset.X_train_encoded.drop(columns=columns_to_drop, inplace=True)
            self.dataset.X_val_encoded.drop(columns=columns_to_drop, inplace=True)
            self.dataset.X_test_encoded.drop(columns=columns_to_drop, inplace=True)


