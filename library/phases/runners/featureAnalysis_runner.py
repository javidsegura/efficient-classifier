

from library.utils.phase_runner_definition.phase_runner import PhaseRunner
from library.pipeline.pipeline_manager import PipelineManager

class FeatureAnalysisRunner(PhaseRunner):
      def __init__(self, pipeline_manager: PipelineManager, include_plots: bool = False, save_path: str = "") -> None:
            super().__init__(pipeline_manager, include_plots, save_path)

      def _run_feature_transformation(self) -> None:
            # Do not add encoding, this is added in _feature_encoding_helper() in dataPreprocessing_runner.py
            pass
      
      def _run_manual_feature_selection(self) -> None:
            """
            Note for @fede:
                  - Currently the same procedure is applied to all pipelines. Better results can be achieved if we apply different procedures to different pipelines.
                  - These runs suppor slow, but they are not the best.
            """


            # 1) Mutual Information
            self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature_selection.manual_feature_selection.fit",
                                                        verbose=True,
                                                        type="MutualInformation",
                                                        threshold=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["mutual_information"]["threshold"],
                                                        delete_features=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["mutual_information"]["delete_features"],
                                                        save_plots=self.include_plots,
                                                        save_path=self.save_path
                                                        )
            # 2) Low Variances
            self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature_selection.manual_feature_selection.fit",
                                                        verbose=True,
                                                        type="LowVariances",
                                                        threshold=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["low_variances"]["threshold"],
                                                        delete_features=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["low_variances"]["delete_features"],
                                                        save_plots=self.include_plots,
                                                        save_path=self.save_path
                                                        )
            # 3) VIF
            self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature_selection.manual_feature_selection.fit",
                                                        verbose=True,
                                                        type="VIF",
                                                        threshold=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["vif"]["threshold"],
                                                        delete_features=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["vif"]["delete_features"],
                                                        save_plots=self.include_plots,
                                                        save_path=self.save_path
                                                        )
            # 4) PCA
            # self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature_selection.manual_feature_selection.fit",
            #                                             verbose=True,
            #                                             type="PCA",
            #                                             threshold=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["pca"]["threshold"],
            #                                             delete_features=self.pipeline_manager.variables["feature_analysis_runner"]["manual_feature_selection"]["pca"]["delete_features"],
            #                                             save_plots=self.include_plots,
            #                                             save_path=self.save_path
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
                                                        )
            
            return {"L1": {"predictivePowerFeatures": None, "excludedFeatures": None, "coefficients": None},
                    "Boruta": {"selected_features": selected_features, "excludedFeatures": excludedFeatures}}
      

            

      
      def _run_feature_engineering_after_split(self) -> None:
            # FEDE (expand here)
            # after split
            ...


      def run(self) -> None:
            feature_transformation_results = self._run_feature_transformation() 
            manual_feature_selection_results = self._run_manual_feature_selection() # Comment out cause it goes too slow
            #automatic_feature_selection_results = self._run_automatic_feature_selection() # Comment out cause it goes too slow
            self._run_feature_engineering_after_split()
            return {
                  "feature_transformation_results": feature_transformation_results,
                  "manual_feature_selection_results": None,
                  "automatic_feature_selection_results": None
                  }
      


