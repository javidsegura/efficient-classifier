from library.utils.phase_runner_definition.phase_runner import PhaseRunner
from library.pipeline.pipeline_manager import PipelineManager
from library.pipeline.pipeline import Pipeline

class FeatureAnalysisRunner(PhaseRunner):
      def __init__(self, pipeline_manager: PipelineManager, include_plots: bool = False, save_path: str = "") -> None:
            super().__init__(pipeline_manager, include_plots, save_path)

      def _run_feature_transformation(self) -> None:
            # Do not add encoding, this is added in _feature_encoding_helper() in dataPreprocessing_runner.py
            pass

      def _update_pipelines_datasets(self, reference_pipeline: Pipeline) -> None:
            """
            Updates the dataset with the new features.
            """
            
            # Not baseline
            for pipeline in self.pipeline_manager.pipelines["not_baseline"]:
                  pipeline.dataset.X_train = reference_pipeline.dataset.X_train
                  pipeline.dataset.X_val = reference_pipeline.dataset.X_val
                  pipeline.dataset.X_test = reference_pipeline.dataset.X_test
      
            # Baseline
            for pipeline in self.pipeline_manager.pipelines["baseline"]:
                  pipeline.dataset.df = reference_pipeline.dataset.df
      
      def _run_manual_feature_selection(self) -> None:
            """
            A pipeline-specific procedure is then applied to all other pipelines. Think of a brief convergence after divergence of the pipelines occured.
            """
            from library.utils.miscellaneous.feature_variance_analysis import analyze_scaled_feature_variances
            
            # First run feature variance analysis to help determine thresholds
            print("Analyzing feature variances to identify optimal threshold...")
            feature_variances = analyze_scaled_feature_variances(
                self.pipeline_manager,
                save_path=self.save_path
            )
            

            reference_category = "not_baseline"
            reference_pipeline = "tree_based"


            # 1) Mutual Information
            self.pipeline_manager.pipelines[reference_category][reference_pipeline].feature_analysis.feature_selection.manual_feature_selection.fit(
                  type="MutualInformation",
                  threshold=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["mutual_information"]["threshold"],
                  delete_features=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["mutual_information"]["delete_features"],
                  save_plots=self.include_plots,
                  save_path=self.save_path
            )
            # 2) Low Variances
            self.pipeline_manager.pipelines[reference_category][reference_pipeline].feature_analysis.feature_selection.manual_feature_selection.fit(
                  type="LowVariances",
                  threshold=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["low_variances"]["threshold"],
                  delete_features=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["low_variances"]["delete_features"],
                  save_plots=self.include_plots,
                  save_path=self.save_path
            )
            # 3) VIF
            self.pipeline_manager.pipelines[reference_category][reference_pipeline].feature_analysis.feature_selection.manual_feature_selection.fit(
                  type="VIF",
                  threshold=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["vif"]["threshold"],
                  delete_features=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["vif"]["delete_features"],
                  save_plots=self.include_plots,
                  save_path=self.save_path
            )
            # # 4) PCA
            #self.pipeline_manager.pipelines["not_baseline"]["tree_based"].feature_analysis.feature.selection.manual_feature_selection.fit(
            #      type="PCA",
            #      threshold=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["pca"]["threshold"],
            #      delete_features=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["pca"]["delete_features"],
            #      save_plots=self.include_plots,
            #      save_path=self.save_path
            #)
            




            # self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature.selection.manual_feature_selection.fit",
            #                                             verbose=True,
            #                                             type="MutualInformation",
            #                                             threshold=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["mutual_information"]["threshold"],
            #                                             delete_features=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["mutual_information"]["delete_features"],
            #                                             save_plots=self.include_plots,
            #                                             save_path=self.save_path,
            #                                             exclude_pipeline_names=["support_vector_machine"]
            #                                             )
            # # 2) Low Variances
            # self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature.selection.manual_feature_selection.fit",
            #                                             verbose=True,
            #                                             type="LowVariances",
            #                                             threshold=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["low_variances"]["threshold"],
            #                                             delete_features=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["low_variances"]["delete_features"],
            #                                             save_plots=self.include_plots,
            #                                             save_path=self.save_path,
            #                                             exclude_pipeline_names=["support_vector_machine"]
            #                                             )
            # # 3) VIF
            # self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature.selection.manual_feature_selection.fit",
            #                                             verbose=True,
            #                                             type="VIF",
            #                                             threshold=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["vif"]["threshold"],
            #                                             delete_features=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["vif"]["delete_features"],
            #                                             save_plots=self.include_plots,
            #                                             save_path=self.save_path,
            #                                             exclude_pipeline_names=["support_vector_machine"]
            #                                             )
            # 4) PCA
            # self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature.selection.manual_feature_selection.fit",
            #                                             verbose=True,
            #                                             type="PCA",
            #                                             threshold=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["pca"]["threshold"],
            #                                             delete_features=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["pca"]["delete_features"],
            #                                             save_plots=self.include_plots,
            #                                             save_path=self.save_path
            #                                             exclude_pipeline_names=["support_vector_machine"]
            #                                             )
            return None
      
      def _run_automatic_feature_selection(self) -> None:
            # # 1) L1
            # predictivePowerFeatures, excludedFeatures, coefficients = self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature_selection.automatic_feature_selection.fit",
            #                                             verbose=False,
            #                                             type="L1",
            #                                             max_iter=self.pipeline_manager.variables["feature_analysis_runner"]["automatic_feature_selection"]["l1"]["max_iter"],
            #                                             delete_features=self.pipeline_manager.variables["feature_analysis_runner"]["automatic_feature_selection"]["l1"]["delete_features"],
            #                                             )
            # 2) Boruta
            selected_features, excludedFeatures = self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature_selection.automatic_feature_selection.fit",
                                                        verbose=True,
                                                        type="Boruta",
                                                        max_iter=self.pipeline_manager.variables["feature_analysis_runner"]["automatic_feature_selection"]["boruta"]["max_iter"],
                                                        delete_features=self.pipeline_manager.variables["feature_analysis_runner"]["automatic_feature_selection"]["boruta"]["delete_features"],
                                                        exclude_pipeline_names=["support_vector_machine"]
                                                        )
            
            return {"L1": {"predictivePowerFeatures": None, "excludedFeatures": None, "coefficients": None},
                    "Boruta": {"selected_features": selected_features, "excludedFeatures": excludedFeatures}}
      

            

      
      def _run_feature_engineering_after_split(self) -> None:
            """Apply feature engineering techniques after dataset split"""
            
            # Keep track of which features were added to which pipelines
            added_features_by_pipeline = {pipeline_name: [] for pipeline_name in self.pipeline_manager.pipelines["not_baseline"]}
    
            # Define which pipelines should get log transformation
            log_transformation_configs = {
                  "tree_based": False,  # Tree-based models don't need log transformation
                  "support_vector_machine": False,
                  "naive_bayes": False,  # Naive Bayes is sensitive to data distribution changes
                  "feed_forward_neural_network": False,  
                  "stacking": True,
                  "ensembled": False
            }
            
            # Apply log transformation to selected pipelines
            log_transform_results = {}
            
            for pipeline_name, apply_log_transform in log_transformation_configs.items():
                  if pipeline_name in self.pipeline_manager.pipelines["not_baseline"] and apply_log_transform:
                        pipeline = self.pipeline_manager.pipelines["not_baseline"][pipeline_name]
                        
                        # Apply log transformation with Memory_PssClean excluded
                        result = pipeline.feature_analysis.feature_engineering.apply_log_transformation(
                        pipeline_name=pipeline_name,
                        exclude_features=["Memory_PssClean"]
                        )
                        log_transform_results[pipeline_name] = result
                        print(f"Applied log transformation to {pipeline_name} pipeline")
      
            # Define pipeline-specific feature interactions
            pipeline_interactions = {
                  "tree_based": [
                        ("Network_TotalTransmittedBytes", "Network_TotalReceivedBytes"),
                        ("CPU_Utilization", "Memory_PssTotal")
                  ],
                  "support_vector_machine": [],
                  "naive_bayes": [],  # No interactions for Naive Bayes
                  "feed_forward_neural_network": [],  # Empty to avoid dimension mismatch
                  "stacking": [ ('API_Network_java.net.URL_openConnection', 'API_Binder_android.app.Activity_startActivity'),
                               ('Memory_PssTotal','API_Crypto-Hash_java.security.MessageDigest_digest')],
                  "ensembled": [
                        ("Network_TotalTransmittedBytes", "API_Network_java.net.URL_openConnection"),
                        ("API_DeviceInfo_android.net.wifi.WifiInfo_getMa", "API_Network_java.net.URL_openConnection"),
                        ('Memory_PssTotal','API_Crypto-Hash_java.security.MessageDigest_digest')
                  ]
            }
            
            # Apply specific interactions to each pipeline
            interaction_results = {}
            
            for pipeline_name, interactions in pipeline_interactions.items():
                  if pipeline_name in self.pipeline_manager.pipelines["not_baseline"]:
                        pipeline = self.pipeline_manager.pipelines["not_baseline"][pipeline_name]
                        
                        if interactions:  # Only apply if interactions are specified
                              result = pipeline.feature_analysis.feature_engineering.create_specific_interaction_features(
                                    interaction_pairs=interactions,
                                    pipeline_name=pipeline_name
                              )
                              interaction_results[pipeline_name] = result
                              
                              # Track the added feature names for each pipeline
                              for pair in interactions:
                                  added_features_by_pipeline[pipeline_name].append(f"{pair[0]}_{pair[1]}_interaction")
                                  
            # Only add Riskware_Adware_Ratio to specific pipelines that can handle additional features
            for pipeline_name in ["stacking", "tree_based", "ensembled"]:
                  # Skip feed_forward_neural_network as it's sensitive to feature count
                  if pipeline_name in self.pipeline_manager.pipelines["not_baseline"] and pipeline_name != "feed_forward_neural_network":
                        pipeline = self.pipeline_manager.pipelines["not_baseline"][pipeline_name]
                        
                        # Check if the required features exist
                        required_features = ['API_IPC_android.content.ContextWrapper_startService', 'API_Network_java.net.URL_openConnection']
                        if all(feature in pipeline.dataset.X_train.columns for feature in required_features):
                            # Add the Riskware_Adware_Ratio feature
                            # Help distinguish Riskware from Adware
                            pipeline.dataset.X_train['Riskware_Adware_Ratio'] = pipeline.dataset.X_train['API_IPC_android.content.ContextWrapper_startService'] / \
                                                (pipeline.dataset.X_train['API_Network_java.net.URL_openConnection'] + 1)
                    
                            if pipeline.dataset.X_val is not None:
                                  pipeline.dataset.X_val['Riskware_Adware_Ratio'] = pipeline.dataset.X_val['API_IPC_android.content.ContextWrapper_startService'] / \
                                                (pipeline.dataset.X_val['API_Network_java.net.URL_openConnection'] + 1)
                    
                            if pipeline.dataset.X_test is not None:
                                  pipeline.dataset.X_test['Riskware_Adware_Ratio'] = pipeline.dataset.X_test['API_IPC_android.content.ContextWrapper_startService'] / \
                                                (pipeline.dataset.X_test['API_Network_java.net.URL_openConnection'] + 1)
                    
                            print(f"Added Riskware_Adware_Ratio feature to {pipeline_name} pipeline")
                            added_features_by_pipeline[pipeline_name].append("Riskware_Adware_Ratio")
                        else:
                            print(f"Cannot add Riskware_Adware_Ratio to {pipeline_name} - required features missing")
    
            # Add memory aggregation features for ensembled trees
            memory_feature_results = {}
            if "ensembled" in self.pipeline_manager.pipelines["not_baseline"]:
                  pipeline = self.pipeline_manager.pipelines["not_baseline"]["ensembled"]
                  
                  # Create new features all at once instead of individually
                  train_new_features = {}
                  val_new_features = {}
                  test_new_features = {}
                  
                  # Define the memory aggregations to create
                  memory_aggregations = [
                        {"name": "Memory_PssPrivate", "components": ["Memory_PrivateClean", "Memory_PrivateDirty"]},
                        {"name": "Memory_PssShared", "components": ["Memory_SharedClean", "Memory_SharedDirty"]},
                        {"name": "Memory_PssDirty", "components": ["Memory_PrivateDirty", "Memory_SharedDirty"]}
                  ]
                  
                  created_features = []
                  
                  for agg in memory_aggregations:
                        name = agg["name"]
                        components = agg["components"]
                        
                        # Check if all component features exist
                        missing_components = [comp for comp in components if comp not in pipeline.dataset.X_train.columns]
                        if missing_components:
                              print(f"Warning: Can't create {name} - Missing features: {missing_components}")
                              continue
                        
                        # Calculate aggregations but don't add to dataframe yet
                        train_new_features[name] = pipeline.dataset.X_train[components].sum(axis=1)
                        
                        if pipeline.dataset.X_val is not None:
                              val_new_features[name] = pipeline.dataset.X_val[components].sum(axis=1)
                        
                        if pipeline.dataset.X_test is not None:
                              test_new_features[name] = pipeline.dataset.X_test[components].sum(axis=1)
                        
                        created_features.append(name)
                        print(f"Created memory aggregation feature: {name}")
                  
                  # Now add all features at once using pd.concat for better performance
                  if train_new_features:
                        # Create dataframes with new features
                        import pandas as pd
                        train_df = pd.DataFrame(train_new_features)
                        val_df = pd.DataFrame(val_new_features)
                        test_df = pd.DataFrame(test_new_features)
                        
                        # Add all new columns at once by concatenation
                        pipeline.dataset.X_train = pd.concat([pipeline.dataset.X_train, train_df], axis=1)
                        
                        if pipeline.dataset.X_val is not None:
                              pipeline.dataset.X_val = pd.concat([pipeline.dataset.X_val, val_df], axis=1)
                        
                        if pipeline.dataset.X_test is not None:
                              pipeline.dataset.X_test = pd.concat([pipeline.dataset.X_test, test_df], axis=1)
                  
                  memory_feature_results["ensembled"] = {"created_features": created_features}
                  added_features_by_pipeline["ensembled"].extend(created_features)
                  print(f"Added {len(created_features)} memory aggregation features to ensembled pipeline")
    
            # IMPORTANT: Since feed_forward_neural_network is used in stacking model,
            # make sure the data dimensions are consistent
            if "feed_forward_neural_network" in self.pipeline_manager.pipelines["not_baseline"]:
                # Make a note of feature dimension
                nn_pipeline = self.pipeline_manager.pipelines["not_baseline"]["feed_forward_neural_network"]
                print(f"Neural network feature dimension: {nn_pipeline.dataset.X_train.shape[1]}")
                
                # If stacking exists, ensure the stacking dataset has the same features as feed_forward_neural_network
                if "stacking" in self.pipeline_manager.pipelines["not_baseline"]:
                    stacking_pipeline = self.pipeline_manager.pipelines["not_baseline"]["stacking"]
                    
                    # Get the columns from both datasets
                    nn_columns = set(nn_pipeline.dataset.X_train.columns)
                    stacking_columns = set(stacking_pipeline.dataset.X_train.columns)
                    
                    # Check if there are extra features in stacking that aren't in nn
                    extra_features = stacking_columns - nn_columns
                    if extra_features:
                        print(f"Warning: Stacking has {len(extra_features)} features not in neural network: {extra_features}")
                        print(f"This will cause dimension mismatch in stacking estimator.")
                        print(f"Synchronizing feature sets between feed_forward_neural_network and stacking...")
                        
                        # Option 1: Add missing columns to neural network (as zeros)
                        import pandas as pd
                        for feature in extra_features:
                            nn_pipeline.dataset.X_train[feature] = 0
                            if nn_pipeline.dataset.X_val is not None:
                                nn_pipeline.dataset.X_val[feature] = 0
                            if nn_pipeline.dataset.X_test is not None:
                                nn_pipeline.dataset.X_test[feature] = 0
                        
                        print(f"Added missing features to neural network. New dimension: {nn_pipeline.dataset.X_train.shape[1]}")
    
            combined_results = {
                  "log_transformation": log_transform_results,
                  "interactions": interaction_results,
                  "memory_aggregations": memory_feature_results,
                  "added_features_by_pipeline": added_features_by_pipeline
            }
    
            return combined_results



      def run(self) -> None:
            feature_transformation_results = self._run_feature_transformation() 
            manual_feature_selection_results = self._run_manual_feature_selection() # Comment out cause it goes too slow
            #automatic_feature_selection_results = self._run_automatic_feature_selection() # Comment out cause it goes too slow
            feature_engineering_results = self._run_feature_engineering_after_split()
            #self._run_feature_engineering_after_split()
            return {
                  "feature_transformation_results": feature_transformation_results,
                  "manual_feature_selection_results": None,
                  "automatic_feature_selection_results": None,
                  "feature_engineering_results": feature_engineering_results
                  }


