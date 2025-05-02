
from library.phases.runners.modelling.utils.states.modelling_runner_states_base import ModellingRunnerStates
from library.pipeline.pipeline_manager import PipelineManager

           
class PostTuningRunner(ModellingRunnerStates):
      def __init__(self, pipeline_manager: PipelineManager, save_plots: bool = False, save_path: str = None):
            super().__init__(pipeline_manager, save_plots, save_path)
      
      def _general_analysis(self):
            # Evaluating and storing models
            comments = "Cate will definetely not like this?"

            # TODO: Add this back in
            # self.pipeline_manager.all_pipelines_execute(methodName="modelling.evaluate_and_store_models", 
            #                                            exclude_category="baseline",
            #                                            comments=comments, 
            #                                            current_phase=self.pipeline_manager.pipeline_state)
            
            
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
            
            # Residual analyisis 
            residuals, confusion_matrices = self.pipeline_manager.pipelines_analysis.plot_confusion_matrix(save_plots=self.save_plots,
                                                                                                          save_path=self.save_path)

            return metrics_df.to_dict(), residuals, confusion_matrices
      

      def run(self):
           print("Post tuning runner")
           best_model, best_score = self.pipeline_manager.select_best_performing_model(metric="f1-score")
           self.pipeline_manager.fit_final_models()
           self.pipeline_manager.evaluate_store_final_models()
           self.pipeline_manager.pipeline_state = "post"

           general_analysis_results = self._general_analysis()


           return best_model, best_score, general_analysis_results
      