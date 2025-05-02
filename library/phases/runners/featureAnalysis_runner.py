

from library.utils.phase_runner.phase_runner import PhaseRunner
from library.pipeline.pipeline_manager import PipelineManager

class FeatureAnalysisRunner(PhaseRunner):
      def __init__(self, pipeline_manager: PipelineManager, include_plots: bool = False, save_path: str = "") -> None:
            super().__init__(pipeline_manager, include_plots, save_path)

      def _run_feature_transformation(self) -> None:
            features_to_encode = ["Reboot"] #harcoded
            encoded_maps_per_pipeline = self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature_transformation.get_categorical_features_encoded", 
                                                                                    verbose=True, 
                                                                                    features=features_to_encode,
                                                                                    encode_y=True)
            return encoded_maps_per_pipeline
      
      def _run_manual_feature_selection(self) -> None:
            # 1) Mutual Information
            self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature_selection.manual_feature_selection.fit",
                                                        verbose=True,
                                                        type="MutualInformation",
                                                        threshold=.2,
                                                        delete_features=True,
                                                        save_plots=self.include_plots,
                                                        save_path=self.save_path
                                                        )
            # 2) Low Variances
            self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature_selection.manual_feature_selection.fit",
                                                        verbose=True,
                                                        type="LowVariances",
                                                        threshold=.01,
                                                        delete_features=True,
                                                        save_plots=self.include_plots,
                                                        save_path=self.save_path
                                                        )
            # 3) VIF
            self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature_selection.manual_feature_selection.fit",
                                                        verbose=True,
                                                        type="VIF",
                                                        threshold=10,
                                                        delete_features=True,
                                                        save_plots=self.include_plots,
                                                        save_path=self.save_path
                                                        )
            # 4) PCA
            # self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature_selection.manual_feature_selection.fit",
            #                                             verbose=True,
            #                                             type="PCA",
            #                                             threshold=.95,
            #                                             delete_features=True,
            #                                             save_plots=self.include_plots,
            #                                             save_path=self.save_path
            #                                             )
            return None
      
      def _run_automatic_feature_selection(self) -> None:
            # 1) L1
            # predictivePowerFeatures, excludedFeatures, coefficients = self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature_selection.automatic_feature_selection.fit",
            #                                             verbose=False,
            #                                             type="L1",
            #                                             max_iter=1000,
            #                                             delete_features=True,
            #                                             )
            # 2) Boruta
            selected_features, excludedFeatures = self.pipeline_manager.all_pipelines_execute(methodName="feature_analysis.feature_selection.automatic_feature_selection.fit",
                                                        verbose=True,
                                                        type="Boruta",
                                                        max_iter=10,
                                                        delete_features=True,
                                          
                                                        )
            
            return {"L1": {"predictivePowerFeatures": None, "excludedFeatures": None, "coefficients": None},
                    "Boruta": {"selected_features": selected_features, "excludedFeatures": excludedFeatures}}
      
      def _run_feature_engineering(self) -> None:
            # FEDE (expand here) => ADD FEDE AN EXAMPLE 
            # after split
            ...


      def run(self) -> None:
            feature_transformation_results = self._run_feature_transformation()
            self._run_manual_feature_selection()
            automatic_feature_selection_results = self._run_automatic_feature_selection()
            self._run_feature_engineering()
            return {
                  "feature_transformation_results": feature_transformation_results,
                  "automatic_feature_selection_results": automatic_feature_selection_results
                  }

