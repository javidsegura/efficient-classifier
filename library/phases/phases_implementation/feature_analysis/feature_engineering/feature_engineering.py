
from library.phases.phases_implementation.dataset.dataset import Dataset
from sklearn.preprocessing import PolynomialFeatures

from library.phases.phases_implementation.data_preprocessing.data_preprocessing import Preprocessing
from library.phases.phases_implementation.feature_analysis.feature_engineering.feature_clustering import FeatureClustering
import numpy as np
import pandas as pd

class FeatureEngineering:
      """
      """
      def __init__(self, dataset: Dataset) -> None:
            self.dataset = dataset
            self.feature_clustering = FeatureClustering(self.dataset)

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
        
      def apply_feature_clustering(self, method="kmeans", n_clusters=None, 
                              correlation_threshold=0.7, use_representatives=True,
                              save_plots=False, save_path=""):
            """
            Apply feature clustering to reduce dimensionality.
            
            Parameters
            ----------
            method : str, default="kmeans"
                  Clustering method: "kmeans", "hierarchical", or "correlation"
            n_clusters : int, default=None
                  Number of clusters to form. If None, it will be estimated.
            correlation_threshold : float, default=0.7
                  Correlation threshold for correlation-based clustering
            use_representatives : bool, default=True
                  If True, replace clusters with representative features
            save_plots : bool, default=False
                  Whether to save clustering visualization plots
            save_path : str, default=""
                  Path to save plots
                  
            Returns
            -------
            dict
                  Dictionary containing clustering information
            """
            return self.feature_clustering.cluster_features(
                  method=method,
                  n_clusters=n_clusters,
                  correlation_threshold=correlation_threshold,
                  use_representatives=use_representatives,
                  save_plots=save_plots,
                  save_path=save_path
            )
            
      def apply_log_transformation(self, pipeline_name: str, exclude_features: list = None) -> dict:
            """
            Apply log transformation to appropriate numeric features.
            
            Parameters
            ----------
            pipeline_name : str
                  Name of the pipeline for logging purposes
            exclude_features : list, optional
                  List of features to exclude from log transformation
            
            Returns
            -------
            dict
                  Information about the applied transformations
            """
            if exclude_features is None:
                  exclude_features = []
            
            results = {
                  "transformed_features": [],
                  "skipped_features": [],
                  "errors": []
            }
            
            # Get numeric features
            numeric_features = self.dataset.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Filter out excluded features
            features_to_transform = [f for f in numeric_features if f not in exclude_features]
            
            for feature in features_to_transform:
                  try:
                        # Check if feature has positive values (necessary for log transformation)
                        if (self.dataset.X_train[feature] <= 0).any():
                              results["skipped_features"].append({
                                    "feature": feature,
                                    "reason": "contains zero or negative values"
                              })
                              continue
                        
                        # Apply log transformation
                        self.dataset.X_train[feature] = np.log1p(self.dataset.X_train[feature])
                        
                        if self.dataset.X_val is not None:
                              self.dataset.X_val[feature] = np.log1p(self.dataset.X_val[feature])
                        
                        if self.dataset.X_test is not None:
                              self.dataset.X_test[feature] = np.log1p(self.dataset.X_test[feature])
                        
                        results["transformed_features"].append(feature)
                        
                  except Exception as e:
                        results["errors"].append({
                        "feature": feature,
                        "error": str(e)
                        })
            
            print(f"[{pipeline_name}] Log-transformed {len(results['transformed_features'])} features, "
                  f"skipped {len(results['skipped_features'])}")
            
            return results

      def create_specific_interaction_features(self, interaction_pairs: list, pipeline_name: str = "") -> dict:
            """
            Create interaction features between specific pairs of features.
            
            Parameters
            ----------
            interaction_pairs : list
                  List of tuples containing feature pairs to create interactions for
                  Example: [("feature1", "feature2"), ("feature3", "feature4")]
            pipeline_name : str, optional
                  Name of the pipeline for logging purposes
                  
            Returns
            -------
            dict
                  Information about the created interactions
            """
            results = {"created_interactions": []}
            
            try:
                  for pair in interaction_pairs:
                        feature1, feature2 = pair
                        interaction_name = f"{feature1}_x_{feature2}"
                        
                        # Check if features exist in the dataset
                        if feature1 not in self.dataset.X_train.columns or feature2 not in self.dataset.X_train.columns:
                              print(f"Warning: Can't create interaction {interaction_name} - Missing features")
                              continue
                        
                        # Create interaction features (multiplication)
                        self.dataset.X_train[interaction_name] = self.dataset.X_train[feature1] * self.dataset.X_train[feature2]
                        
                        if self.dataset.X_val is not None:
                              self.dataset.X_val[interaction_name] = self.dataset.X_val[feature1] * self.dataset.X_val[feature2]
                        
                        if self.dataset.X_test is not None:
                              self.dataset.X_test[interaction_name] = self.dataset.X_test[feature1] * self.dataset.X_test[feature2]
                        
                        results["created_interactions"].append(interaction_name)
                  
                  if pipeline_name and results["created_interactions"]:
                        print(f"[{pipeline_name}] Created {len(results['created_interactions'])} interaction features")
                  
                  return results
                  
            except Exception as e:
                  print(f"Error creating interaction features: {str(e)}")
                  return {"error": str(e), "created_interactions": []}
            
            


   