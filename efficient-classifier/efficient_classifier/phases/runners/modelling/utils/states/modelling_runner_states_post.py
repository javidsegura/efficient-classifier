
from efficient_classifier.phases.runners.modelling.utils.states.modelling_runner_states_base import ModellingRunnerStates
from efficient_classifier.pipeline.pipeline_manager import PipelineManager

           
class PostTuningRunner(ModellingRunnerStates):
      def __init__(self, pipeline_manager: PipelineManager, save_plots: bool = False, save_path: str = None):
            super().__init__(pipeline_manager, save_plots, save_path)
      
      def _general_analysis(self):
            # Cross model comparison
            self.pipeline_manager.pipelines_analysis.plot_cross_model_comparison(
                  save_plots=self.save_plots,
                  save_path=self.save_path)
            
            # Time based model performance
            metrics_df = self.pipeline_manager.pipelines_analysis.plot_results_df(metrics=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["model_assesment"]["results_df_metrics"],
                                                                                 save_plots=self.save_plots,
                                                                                 save_path=self.save_path)

            # Results summary
            self.pipeline_manager.pipelines_analysis.plot_results_summary(training_metric=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["model_assesment"]["results_summary"]["training_metric"],
                                                                         performance_metric=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["model_assesment"]["results_summary"]["performance_metric"],
                                                                         save_plots=self.save_plots,
                                                                         save_path=self.save_path)
            # Intra model comparison
            self.pipeline_manager.pipelines_analysis.plot_intra_model_comparison(
                                                                                 save_plots=self.save_plots,
                                                                                 save_path=self.save_path)
            
            # Residual analyisis 
            residuals, confusion_matrices = self.pipeline_manager.pipelines_analysis.plot_confusion_matrix(save_plots=self.save_plots,
                                                                                                          save_path=self.save_path)

            return {
                  "metrics_df": metrics_df.to_dict(), 
                  "residuals": residuals, 
                  "confusion_matrices": confusion_matrices
                  }
      
      def _update_dag_scheme(self, best_model, best_score):
            flag = True 
            for pipeline in self.pipeline_manager.variables["general"]["pipelines_names"]["not_baseline"]:
                 if not flag:
                        break
                 for model in self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["models_to_include"]["not_baseline"][pipeline]:
                        if model == best_model:
                              self.pipeline_manager.dag.add_procedure(pipeline, "modelling", f"post-tuning ({self.pipeline_manager.variables['phase_runners']['dataset_runner']['metrics_to_evaluate']['preferred_metric']})", best_score)
                              flag = False 
                              break

      def run(self):
           print("Post tuning runner")
           best_model, best_score = self.pipeline_manager.select_best_performing_model()
           self.pipeline_manager.fit_final_models()
           self.pipeline_manager.evaluate_store_final_models()
           self.pipeline_manager.pipeline_state = "post"

           general_analysis_results = self._general_analysis()
           self._update_dag_scheme(best_model, best_score)

           return best_model, best_score, general_analysis_results
      