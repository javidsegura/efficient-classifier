from efficient_classifier.utils.phase_runner_definition.phase_runner import PhaseRunner
from efficient_classifier.pipeline.pipeline_manager import PipelineManager
from efficient_classifier.phases.phases_implementation.data_preprocessing.data_preprocessing import Preprocessing
import yaml
from pathlib import Path

class DataPreprocessingRunner(PhaseRunner):
      def __init__(self, pipeline_manager: PipelineManager, include_plots: bool = False, save_path: str = "") -> None:
            super().__init__(pipeline_manager, include_plots, save_path)
            self.variables = pipeline_manager.variables

      def _feature_encoding_helper(self) -> dict:
            encoded_maps_per_pipeline = self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature_transformation.get_categorical_features_encoded", 
                                                                                    verbose=True, 
                                                                                    features=self.pipeline_manager.variables["phase_runners"]["data_preprocessing_runner"]["features_to_encode"],
                                                                                    encode_y=True)
            print(f"ENCODED MAP PIPELINS IS: {encoded_maps_per_pipeline}")
            # Get the first key for the not basleine category
            encoded_map_reference = list(encoded_maps_per_pipeline["not_baseline"].keys())[0]
            self.pipeline_manager.pipelines_analysis.encoded_map = encoded_maps_per_pipeline["not_baseline"][encoded_map_reference]

      
      def _create_pipelines_divergences(self):
            for pipelineName in self.pipeline_manager.variables["general"]["pipelines_names"]["not_baseline"]:
                  self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName=pipelineName)
            self.pipeline_manager.create_pipeline_divergence(category="baseline", pipelineName="baselines")
            print(f"Pipelines AFTER divergences: {self.pipeline_manager.pipelines}")
      
            return None
        
      def _execute_preprocessing(self, preprocessing: Preprocessing, pipeline_name: str) -> str:
            """
            Apply missing-value handling, duplicate analysis, outlier bounding,
            feature scaling, and class-imbalance correction based on config,
            returning a composed summary of operations/results.

            Parameters
            ----------
            preprocessing: Preprocessing
                  The preprocessing object contains all the methods 
            pipeline_name: str
                  The name of the pipeline to use.

            Returns
            -------
            str
                  A composed summary of operations/results.

            """
            messages = []
            save_path = self.save_path + f"/{pipeline_name}"

            # 1) Missing values & Duplicate analysis
            print(f"\nPreprocessing --- Missing Values & Duplicates - {pipeline_name}\n")
            missing_res = preprocessing.uncomplete_data_obj.get_missing_values(
                  placeholders=self.variables["phase_runners"]["data_preprocessing_runner"]["placeholders"],
                  save_plots=self.include_plots,
                  save_path=save_path
            )
            messages.append(f"Handled missing values : {missing_res}")
            self.pipeline_manager.dag.add_procedure(pipeline_name, "data_preprocessing", "missing_values_and_duplicates", missing_res)

            dup_res = preprocessing.uncomplete_data_obj.analyze_duplicates(
                  save_plots=self.include_plots,
                  save_path=save_path
                  )
            messages.append(f"Duplicates analyzed : {dup_res}")
            self.pipeline_manager.dag.add_procedure(pipeline_name, "data_preprocessing", "analyze_duplicates", dup_res)

            # 2) Outlier detection & bounding
            print(f"\nPreprocessing --- Bounds & Outliers - {pipeline_name}\n")
            out_res = preprocessing.outliers_bounds_obj.get_outliers(
                  detection_type=self.variables["phase_runners"]["data_preprocessing_runner"]["outliers"]["detection_type"],
                  save_plots=False, 
                  save_path=save_path
            )
            preprocessing.outliers_bounds_obj.bound_checking()
            messages.append(f"Outliers detected by {self.variables['phase_runners']['data_preprocessing_runner']['outliers']['detection_type']} : {None}")
            self.pipeline_manager.dag.add_procedure(pipeline_name, "data_preprocessing", "outliers_detection", out_res)

            # 3) Feature scaling
            print(f"\nPreprocessing --- Feature Scaling - {pipeline_name}\n")
            scaler = self.variables["phase_runners"]["data_preprocessing_runner"]["pipeline_specific_configurations"]["scaler"][pipeline_name]
            if scaler == "no_scaler":
                  scale_res = "No scaling performed"
            else:
                  scale_res = preprocessing.feature_scaling_obj.scale_features(
                        scaler=scaler,
                        columnsToScale=preprocessing.dataset.X_train.select_dtypes(include=["number"]).columns,
                        save_plots=self.include_plots,
                        save_path=save_path
                  )
            messages.append(f"Features scaled with {scaler} : {scale_res}")
            self.pipeline_manager.dag.add_procedure(pipeline_name, "data_preprocessing", "feature_scaling", scale_res)

            # 4) Class imbalance correction
            print(f"\nPreprocessing --- Class Imbalance - {pipeline_name}\n")
            imbalancer = self.variables["phase_runners"]["data_preprocessing_runner"]["pipeline_specific_configurations"]["imbalancer"][pipeline_name]
            if imbalancer == "no_imbalancer":
                  imb_res = "No imbalancer performed"
            else:
                  imb_res = preprocessing.class_imbalance_obj.class_imbalance(
                        method=imbalancer,
                        save_plots=self.include_plots,
                        save_path=save_path
                  )
            messages.append(f"Class imbalance: {imb_res}")
            self.pipeline_manager.dag.add_procedure(pipeline_name, "data_preprocessing", "class_imbalance", imb_res)

            print(f"Messages: {messages}")
            return "; ".join(messages)
    
      
      
      def run(self) -> None:
            self._feature_encoding_helper()
            self._create_pipelines_divergences()
            
            print("-"*30)
            print("STARTING PREPROCESSING")
            print("-"*30)
            
            results = {}

            # for category_name, pipelines in self.pipeline_manager.pipelines.items():
            #       results[category_name] = {}
            #       for pipeline_name, pipeline in pipelines.items():
            #             if pipeline_name == "stacking":
            #                   continue
            #             print(f"--> Running preprocessing on pipeline: {category_name} / {pipeline_name}")
            #             print("-"*30)
            #             summary = self._execute_preprocessing(preprocessing=pipeline.preprocessing, pipeline_name=pipeline_name)
            #             print(summary)
            #             results[category_name][pipeline_name] = summary
            return results