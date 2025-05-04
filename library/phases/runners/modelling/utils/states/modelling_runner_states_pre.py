
from library.phases.runners.modelling.utils.states.modelling_runner_states_base import ModellingRunnerStates
from library.pipeline.pipeline_manager import PipelineManager

from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier

class PreTuningRunner(ModellingRunnerStates):
      def __init__(self, pipeline_manager: PipelineManager, save_plots: bool = False, save_path: str = None):
            super().__init__(pipeline_manager, save_plots, save_path)

      def _general_analysis(self):
            # Evaluating and storing models
            comments = self.pipeline_manager.variables["modelling_runner"]["model_assesment"]["comments"]
            self.pipeline_manager.all_pipelines_execute(methodName="modelling.evaluate_and_store_models", 
                                                       verbose=False,
                                                       comments=comments, 
                                                       current_phase="pre")
            
            
            # Cross model comparison
            self.pipeline_manager.pipelines_analysis.plot_cross_model_comparison(
                  metric=self.pipeline_manager.variables["modelling_runner"]["model_assesment"]["cross_model_metrics"],
                  save_plots=self.save_plots,
                  save_path=self.save_path)
            
            # Time based model performance
            metrics_df = self.pipeline_manager.pipelines_analysis.plot_results_df(metrics=self.pipeline_manager.variables["modelling_runner"]["model_assesment"]["results_df_metrics"],
                                                                                 save_plots=self.save_plots,
                                                                                 save_path=self.save_path)

            # Results summary
            self.pipeline_manager.pipelines_analysis.plot_results_summary(training_metric=self.pipeline_manager.variables["modelling_runner"]["model_assesment"]["results_summary"]["training_metric"],
                                                                         performance_metric=self.pipeline_manager.variables["modelling_runner"]["model_assesment"]["results_summary"]["performance_metric"],
                                                                         save_plots=self.save_plots,
                                                                         save_path=self.save_path)
            # Intra model comparison
            self.pipeline_manager.pipelines_analysis.plot_intra_model_comparison(metrics=self.pipeline_manager.variables["modelling_runner"]["model_assesment"]["intra_model_metrics"],
                                                                                 save_plots=self.save_plots,
                                                                                 save_path=self.save_path)
            # Per-epoch progress
            self.pipeline_manager.pipelines_analysis.plot_per_epoch_progress(metrics=self.pipeline_manager.variables["modelling_runner"]["model_assesment"]["per_epoch_metrics"],
                                                                                 save_plots=self.save_plots,
                                                                                 save_path=self.save_path)
            
            # Residual analyisis 
            residuals, confusion_matrices = self.pipeline_manager.pipelines_analysis.plot_confusion_matrix(save_plots=self.save_plots,
                                                                                                          save_path=self.save_path)
            

            # Feature importance
            importances_dfs = self.pipeline_manager.pipelines_analysis.plot_feature_importance(save_plots=self.save_plots,
                                                                                                save_path=self.save_path)

            return metrics_df.to_dict(), residuals, confusion_matrices, importances_dfs

      def _set_up_stacking_model(self):
            estimators = []
            for pipelineName, pipelineObject in self.pipeline_manager.pipelines["not_baseline"].items():
                  for modelName, modelObject in pipelineObject.modelling.list_of_models.items():
                        if modelName in pipelineObject.modelling.models_to_exclude:
                              continue
                        modelSklearn = modelObject.tuning_states["pre"].assesment["model_sklearn"]
                        estimators.append((modelName, modelSklearn))
            
            #Stacking model
            stackingModel = StackingClassifier(
                  estimators=estimators,
                  final_estimator=DecisionTreeClassifier(), # to be manually tuned
                  cv="prefit",
                  verbose=3
            )

            self.pipeline_manager.pipelines["not_baseline"]["stacking"].modelling.add_model("Stacking", stackingModel, model_type="stacking")

            all_pipelines_to_exclude = []

            for pipelineName, pipelineObject in self.pipeline_manager.pipelines["not_baseline"].items():
                  if pipelineName == "stacking":
                        continue
                  all_pipelines_to_exclude.append(pipelineName)

            self.pipeline_manager.all_pipelines_execute(methodName="modelling.fit_models", 
                                       current_phase="pre",
                                       exclude_category="baseline",
                                       exclude_pipeline_names=all_pipelines_to_exclude
                                       )
      def run(self):
            self.pipeline_manager.pipeline_state = "pre"
            print("Pre tuning runner about to start")
            # Fitting models
            self.pipeline_manager.all_pipelines_execute(
                  methodName="modelling.fit_models",
                                       exclude_pipeline_names=["stacking"], # debugging
                                       current_phase="pre")
            self._set_up_stacking_model()
            general_analysis_results = self._general_analysis()

            return general_analysis_results

            
