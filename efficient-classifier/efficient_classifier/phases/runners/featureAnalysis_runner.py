

from efficient_classifier.utils.phase_runner_definition.phase_runner import PhaseRunner
from efficient_classifier.pipeline.pipeline_manager import PipelineManager
from efficient_classifier.pipeline.pipeline import Pipeline

class FeatureAnalysisRunner(PhaseRunner):
      def __init__(self, pipeline_manager: PipelineManager, include_plots: bool = False, save_path: str = "") -> None:
            super().__init__(pipeline_manager, include_plots, save_path)
            self.reference_category = "not_baseline"
            self.reference_pipeline = self.pipeline_manager.variables["general"]["pipelines_names"]["not_baseline"][0]


      def _run_feature_transformation(self) -> None:
            pass

      def _update_pipelines_datasets(self, reference_pipeline: Pipeline) -> None:
            """
            Updates the dataset with the new features.
            """
            
            # Not baseline
            for pipeline, pipeline_obj in self.pipeline_manager.pipelines["not_baseline"].items():
                  pipeline_obj.dataset.X_train = reference_pipeline.dataset.X_train
                  pipeline_obj.dataset.X_val = reference_pipeline.dataset.X_val
                  pipeline_obj.dataset.X_test = reference_pipeline.dataset.X_test
      
            # Baseline
            for pipeline, pipeline_obj in self.pipeline_manager.pipelines["baseline"].items():
                  pipeline_obj.dataset.X_train = reference_pipeline.dataset.X_train
                  pipeline_obj.dataset.X_val = reference_pipeline.dataset.X_val
                  pipeline_obj.dataset.X_test = reference_pipeline.dataset.X_test
      
      def _run_manual_feature_selection(self) -> None:
            """
            A pipeline-specific procedure is then applied to all other pipelines. Think of a brief convergence after divergence of the pipelines occured.
            """

            # 1) Mutual Information
            self.pipeline_manager.pipelines[self.reference_category][self.reference_pipeline].feature_analysis.feature_selection.manual_feature_selection.fit(
                  type="MutualInformation",
                  threshold=self.pipeline_manager.variables["phase_runners"]["feature_analysis_runner"]["manual_feature_selection"]["mutual_information"]["threshold"],
                  delete_features=self.pipeline_manager.variables["phase_runners"]["feature_analysis_runner"]["manual_feature_selection"]["mutual_information"]["delete_features"],
                  save_plots=self.include_plots,
                  save_path=self.save_path
            )
            #2) Low Variances
            self.pipeline_manager.pipelines[self.reference_category][self.reference_pipeline].feature_analysis.feature_selection.manual_feature_selection.fit(
                  type="LowVariances",
                  threshold=self.pipeline_manager.variables["phase_runners"]["feature_analysis_runner"]["manual_feature_selection"]["low_variances"]["threshold"],
                  delete_features=self.pipeline_manager.variables["phase_runners"]["feature_analysis_runner"]["manual_feature_selection"]["low_variances"]["delete_features"],
                  save_plots=self.include_plots,
                  save_path=self.save_path
            )
            # 3) VIF
            self.pipeline_manager.pipelines[self.reference_category][self.reference_pipeline].feature_analysis.feature_selection.manual_feature_selection.fit(
                  type="VIF",
                  threshold=self.pipeline_manager.variables["phase_runners"]["feature_analysis_runner"]["manual_feature_selection"]["vif"]["threshold"],
                  delete_features=self.pipeline_manager.variables["phase_runners"]["feature_analysis_runner"]["manual_feature_selection"]["vif"]["delete_features"],
                  save_plots=self.include_plots,
                  save_path=self.save_path
            )
            # # 4) PCA
            # self.pipeline_manager.pipelines[self.reference_category][self.reference_pipeline].feature_analysis.feature_selection.manual_feature_selection.fit(
            #       type="PCA",
            #       threshold=self.pipeline_manager.variables["phase_runners"]["feature_analysis_runner"]["manual_feature_selection"]["pca"]["threshold"],
            #       delete_features=self.pipeline_manager.variables["phase_runners"]["feature_analysis_runner"]["manual_feature_selection"]["pca"]["delete_features"],
            #       save_plots=self.include_plots,
            #       save_path=self.save_path
            # )

            # Update all other pipelines
            self._update_pipelines_datasets(self.pipeline_manager.pipelines[self.reference_category][self.reference_pipeline])

            return {
                  "MutualInformation": f"Threshold: {self.pipeline_manager.variables['phase_runners']['feature_analysis_runner']['manual_feature_selection']['mutual_information']['threshold']}, Delete features: {self.pipeline_manager.variables['phase_runners']['feature_analysis_runner']['manual_feature_selection']['mutual_information']['delete_features']}",
                  "LowVariances": f"Threshold: {self.pipeline_manager.variables['phase_runners']['feature_analysis_runner']['manual_feature_selection']['low_variances']['threshold']}, Delete features: {self.pipeline_manager.variables['phase_runners']['feature_analysis_runner']['manual_feature_selection']['low_variances']['delete_features']}",
                  "VIF": f"Threshold: {self.pipeline_manager.variables['phase_runners']['feature_analysis_runner']['manual_feature_selection']['vif']['threshold']}, Delete features: {self.pipeline_manager.variables['phase_runners']['feature_analysis_runner']['manual_feature_selection']['vif']['delete_features']}",
                  "PCA": f"Threshold: {self.pipeline_manager.variables['phase_runners']['feature_analysis_runner']['manual_feature_selection']['pca']['threshold']}, Delete features: {self.pipeline_manager.variables['phase_runners']['feature_analysis_runner']['manual_feature_selection']['pca']['delete_features']}"
            }
      
      def _run_automatic_feature_selection(self) -> None:
            # 1) L1
            # predictivePowerFeatures, excludedFeatures, coefficients = self.pipeline_manager.pipelines[self.reference_category][self.reference_pipeline].feature_analysis.feature_selection.automatic_feature_selection.fit(
            #       type="L1",
            #       max_iter=self.pipeline_manager.variables["phase_runners"]["feature_analysis_runner"]["automatic_feature_selection"]["l1"]["max_iter"],
            #       delete_features=self.pipeline_manager.variables["phase_runners"]["feature_analysis_runner"]["automatic_feature_selection"]["l1"]["delete_features"],
            # )

            # 2) Boruta
            selected_features, excludededFeatures = self.pipeline_manager.pipelines[self.reference_category][self.reference_pipeline].feature_analysis.feature_selection.automatic_feature_selection.fit(
                  type="Boruta",
                  max_iter=self.pipeline_manager.variables["phase_runners"]["feature_analysis_runner"]["automatic_feature_selection"]["boruta"]["max_iter"],
                  delete_features=self.pipeline_manager.variables["phase_runners"]["feature_analysis_runner"]["automatic_feature_selection"]["boruta"]["delete_features"],
            )

            # Update all other pipelines
            self._update_pipelines_datasets(self.pipeline_manager.pipelines[self.reference_category][self.reference_pipeline])

            
            return {
                  "L1": {
                        "predictivePowerFeatures": None, 
                        "excludedFeatures": None, 
                        "deletesFeatures": self.pipeline_manager.variables["phase_runners"]["feature_analysis_runner"]["automatic_feature_selection"]["l1"]["delete_features"]
                        },
                  "Boruta": {
                        "selected_features": None,
                        "excludedFeatures": None,
                        "deletesFeatures": self.pipeline_manager.variables["phase_runners"]["feature_analysis_runner"]["automatic_feature_selection"]["boruta"]["delete_features"]
                        }
                  }
      
      def _update_dag_scheme(self, 
                             feature_transformation_results,
                             manual_feature_selection_results,
                             automatic_feature_selection_results
                             ):
            for category in self.pipeline_manager.pipelines:
                  for pipeline in self.pipeline_manager.pipelines[category]:
                        if pipeline == "stacking":
                              continue
                        # 1) Feature transformation
                        self.pipeline_manager.dag.add_procedure(pipeline, "feature_analysis", "feature_transformation", None)
                        # 2) Feature engineering
                        self.pipeline_manager.dag.add_procedure(pipeline, "feature_analysis", "feature_engineering", None)
                        # 3) Feature selection
                        self.pipeline_manager.dag.add_procedure(pipeline, "feature_analysis", "feature_selection", None)

                        # 3.1) Manual feature selection
                        self.pipeline_manager.dag.add_subprocedure(pipeline, "feature_analysis", "feature_selection", "manual", None)
                        self.pipeline_manager.dag.add_method(pipeline, "feature_analysis", "feature_selection", "manual", "MutualInformation", manual_feature_selection_results["MutualInformation"])
                        self.pipeline_manager.dag.add_method(pipeline, "feature_analysis", "feature_selection", "manual", "LowVariances", manual_feature_selection_results["LowVariances"])
                        self.pipeline_manager.dag.add_method(pipeline, "feature_analysis", "feature_selection", "manual", "VIF", manual_feature_selection_results["VIF"])
                        self.pipeline_manager.dag.add_method(pipeline, "feature_analysis", "feature_selection", "manual", "PCA", manual_feature_selection_results["PCA"])
                        # 3.2) Automatic feature selection
                        self.pipeline_manager.dag.add_subprocedure(pipeline, "feature_analysis", "feature_selection", "automatic", None)
                        self.pipeline_manager.dag.add_method(pipeline, "feature_analysis", "feature_selection", "automatic", "L1", automatic_feature_selection_results["L1"])
                        self.pipeline_manager.dag.add_method(pipeline, "feature_analysis", "feature_selection", "automatic", "Boruta", automatic_feature_selection_results["Boruta"])

      def run(self) -> None:
            feature_transformation_results = self._run_feature_transformation() 
            manual_feature_selection_results = self._run_manual_feature_selection() # Comment out cause it goes too slow
            automatic_feature_selection_results = self._run_automatic_feature_selection() # Comment out cause it goes too slow

            # Print X_train shape for all pipelines
            for category in self.pipeline_manager.pipelines:
                  for pipeline in self.pipeline_manager.pipelines[category]:
                        print(f"Pipeline: {pipeline}, X_train shape: {self.pipeline_manager.pipelines[category][pipeline].dataset.X_train.shape}")
            

            self._update_dag_scheme(feature_transformation_results, manual_feature_selection_results, automatic_feature_selection_results)

            return {
                  "feature_transformation_results": feature_transformation_results,
                  "manual_feature_selection_results": manual_feature_selection_results,
                  "automatic_feature_selection_results": automatic_feature_selection_results
                  }
      


