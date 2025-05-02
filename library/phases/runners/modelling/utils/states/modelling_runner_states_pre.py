
from library.phases.runners.modelling.utils.states.modelling_runner_states_base import ModellingRunnerStates
from library.pipeline.pipeline_manager import PipelineManager


class PreTuningRunner(ModellingRunnerStates):
      def __init__(self, pipeline_manager: PipelineManager, save_plots: bool = False, save_path: str = None):
            super().__init__(pipeline_manager, save_plots, save_path)

      def run(self):
            self.pipeline_manager.pipeline_state = "pre"
            print("Pre tuning runner about to start")
            # Fitting models
            self.pipeline_manager.all_pipelines_execute(methodName="modelling.fit_models",
                                       verbose=False, 
                                       exclude_pipeline_names=["stacking"], # debugging
                                       current_phase="pre")
            # Evaluating and storing models
            comments = "Cate will definetely not like this?"
            self.pipeline_manager.all_pipelines_execute(methodName="modelling.evaluate_and_store_models", 
                                                       verbose=False,
                                                       exclude_pipeline_names=["stacking"],
                                                       comments=comments, 
                                                       current_phase="pre")
            
            # Cross model comparison
            self.pipeline_manager.pipelines_analysis.plot_cross_model_comparison(
                  metric=["f1-score", "recall", "precision", "accuracy"],
                  save_plots=self.save_plots,
                  save_path=self.save_path)
            
            # Time based model performance
            metrics_df = self.pipeline_manager.pipelines_analysis.plot_results_df(metrics=["timeToFit", "timeToPredict"],
                                                                                 save_plots=self.save_plots,
                                                                                 save_path=self.save_path)
            
            # Results summary
            self.pipeline_manager.pipelines_analysis.plot_results_summary(training_metric="timeToFit",
                                                                         performance_metric="accuracy",
                                                                         save_plots=self.save_plots,
                                                                         save_path=self.save_path)
            # Intra model comparison
            self.pipeline_manager.pipelines_analysis.plot_intra_model_comparison(metrics=["f1-score", "recall", "precision", "accuracy"],
                                                                                 save_plots=self.save_plots,
                                                                                 save_path=self.save_path)
            
            # Per-epoch progress
            self.pipeline_manager.pipelines_analysis.plot_per_epoch_progress(metrics=["accuracy", "loss"],
                                                                                 save_plots=self.save_plots,
                                                                                 save_path=self.save_path)
            
            # Residual analyisis 
            residuals, confusion_matrices = self.pipeline_manager.pipelines_analysis.plot_confusion_matrix(save_plots=self.save_plots,
                                                                                                          save_path=self.save_path)
            
            # Feature importance
            importances_dfs = self.pipeline_manager.pipelines_analysis.plot_feature_importance(save_plots=self.save_plots,
                                                                                                save_path=self.save_path)
            

            return metrics_df.to_dict(), residuals, confusion_matrices, importances_dfs


            
