import pandas as pd
import os
from abc import ABC, abstractmethod

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from efficient_classifier.phases.phases_implementation.dataset.dataset import Dataset
from efficient_classifier.phases.phases_implementation.EDA.EDA import EDA
from efficient_classifier.utils.miscellaneous.save_or_store_plot import save_or_store_plot

class ManualFeatureSelection():
      """
      """
      def __init__(self, dataset: Dataset):
            self.dataset = dataset
            self.options = {
                  "VIF": VIFElimination,
                  "LowVariances": LowVariancesFeatureReduction,
                  "MutualInformation": MutualInformationFeatureReduction,
                  "PCA": PCAFeatureReduction
            }

      def fit(self, type: str, threshold: float, delete_features: bool, save_plots: bool, save_path: str):
            print(f"Running {type} feature selection")
            return self.options[type](self.dataset).fit(threshold, delete_features, save_plots, save_path)
  

class ManualFeatureSelectionFactory(ABC):
      def __init__(self, dataset: Dataset):
            self.dataset = dataset
      @abstractmethod
      def fit(self, threshold: float, delete_features: bool, save_plots: bool, save_path: str):
            pass

class VIFElimination(ManualFeatureSelectionFactory):
        def __init__(self, dataset: Dataset):
            super().__init__(dataset)

        def __calculate_vif(self):
            """
            Calculates the VIF of the features.

            Returns
            -------
            pd.DataFrame
                A dataframe with the features and their VIF.
            """
            vif_data = pd.DataFrame()
            only_numerical_features = self.dataset.X_train.select_dtypes(include=["number"])
            vif_data["Feature"] = only_numerical_features.columns
            vif_data["VIF"] = [variance_inflation_factor(only_numerical_features.values, i) for i in range(len(only_numerical_features.columns))]
            return vif_data
    
        def fit(self, threshold=10, delete_features: bool = True, save_plots: bool = False, save_path: str = ""):
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
            if save_plots:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                eda = EDA(self.dataset)
                eda.plot_correlation_matrix(size="l", numerical_df=self.dataset.X_train.select_dtypes(include=["number"]), title="Prior-Elimination", save_plots=save_plots, save_path=save_path)
            while True:
                number_of_iterations += 1
                vif_data = self.__calculate_vif()
                print(f"VIF computed for iteration {number_of_iterations}:")
                max_vif = vif_data["VIF"].max()
                if max_vif < threshold:
                    break
                feature_to_drop = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
                if delete_features: 
                    self.dataset.X_train.drop(columns=[feature_to_drop], inplace=True)
                    self.dataset.X_val.drop(columns=[feature_to_drop], inplace=True)
                    self.dataset.X_test.drop(columns=[feature_to_drop], inplace=True)
                    print(f"\tDropped: '{feature_to_drop}' with a VIF of {max_vif}")
                else:
                    print(f"Feature with highest VIF: '{feature_to_drop}' with VIF: {max_vif}")
                    break
            
            if save_plots:
                eda.plot_correlation_matrix(size="l", numerical_df=self.dataset.X_train.select_dtypes(include=["number"]), title="Post-Elimination", save_plots=save_plots, save_path=save_path)



class LowVariancesFeatureReduction(ManualFeatureSelectionFactory):
        def __init__(self, dataset: Dataset):
            super().__init__(dataset)

        def constant_features_reduction(self):
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

        def fit(self, threshold: float = 0.01, delete_features: bool = True, save_plots: bool = False, save_path: str = ""):
            """
            Removes the features with low variance.
            """

            self.constant_features_reduction()
            
            # Create dataframe with feature and standard deviation (spreadness)
            spreadness_df = pd.DataFrame({
                "feature": self.dataset.X_train.columns,
                "spreadness": self.dataset.X_train.std()
            }).reset_index(drop=True)
            if save_plots:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.hist(spreadness_df["spreadness"], bins=30, edgecolor='black')
                ax.set_title('Distribution of Standard Deviations for Numeric Features')
                ax.set_xlabel('Standard Deviation')
                ax.set_ylabel('Frequency')
                save_or_store_plot(fig, save_plots, save_path + "/feature_selection/manual/low_variances", "low_variances.png")
            columns_to_drop = spreadness_df[spreadness_df["spreadness"] < threshold]["feature"].tolist()
            if delete_features:
                self.dataset.X_train.drop(columns=columns_to_drop, inplace=True)
                self.dataset.X_val.drop(columns=columns_to_drop, inplace=True)
                self.dataset.X_test.drop(columns=columns_to_drop, inplace=True)

class MutualInformationFeatureReduction(ManualFeatureSelectionFactory):
        def __init__(self, dataset: Dataset):
            super().__init__(dataset)

        def _compute_feature_relevance(self, feature: str):
            """
            Computes the relevance of the features.
            """
            mutual_info_train = mutual_info_regression(self.dataset.X_train[[feature]], self.dataset.y_train)
            return mutual_info_train[0]
      
        def fit(self, threshold: float, delete_features: bool, save_plots: bool = False, save_path: str = ""):
            relevance_scores = {
                col: self._compute_feature_relevance(col)
                for col in self.dataset.X_train.columns
            }

            relevance_df = pd.DataFrame(list(relevance_scores.items()), columns=['Feature', 'Relevance'])
            relevance_df = relevance_df.sort_values(by='Relevance', ascending=False)

            irrelevant_features = relevance_df[relevance_df["Relevance"] < threshold]["Feature"].tolist()
            print(f"Number of irrelevant features: {len(irrelevant_features)}. They are: {irrelevant_features}")

            if save_plots:
                fig, ax = plt.subplots(figsize=(10, min(0.3 * len(relevance_df), 20)))
                ax.barh(relevance_df['Feature'], relevance_df['Relevance'], color='skyblue')
                ax.set_xlabel('Feature Relevance')
                ax.set_title('Feature Relevance Scores')
                ax.invert_yaxis()  # Highest relevance on top
                save_or_store_plot(fig, save_plots, save_path + "/feature_selection/manual/mutual_information", "mutual_information.png")

            if delete_features:
                self.dataset.X_train.drop(columns=irrelevant_features, inplace=True)
                self.dataset.X_val.drop(columns=irrelevant_features, inplace=True)
                self.dataset.X_test.drop(columns=irrelevant_features, inplace=True)

            return irrelevant_features

class PCAFeatureReduction(ManualFeatureSelectionFactory):
        def __init__(self, dataset: Dataset):
            super().__init__(dataset)

        def fit(self, threshold: float = 0.95, delete_features: bool = True, save_plots: bool = False, save_path: str = ""):
            """
            Reduces the number of features using PCA.
            """
            # After fitting PCA
            pca = PCA(n_components=threshold)
            pca.fit(self.dataset.X_train)

            # Determine how many components were kept
            num_components = pca.n_components_

            print(f'PCA kept {num_components} components')

            columns = [f'PC{i+1}' for i in range(num_components)]

            if delete_features:
                # Transform and convert back to DataFrame
                self.dataset.X_train = pd.DataFrame(
                    pca.transform(self.dataset.X_train),
                    columns=columns,
                    index=self.dataset.X_train.index
                )

                self.dataset.X_val = pd.DataFrame(
                    pca.transform(self.dataset.X_val),
                    columns=columns,
                    index=self.dataset.X_val.index
                )

                self.dataset.X_test = pd.DataFrame(
                    pca.transform(self.dataset.X_test),
                    columns=columns,
                    index=self.dataset.X_test.index
                )


    
      
