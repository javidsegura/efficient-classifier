
from library.phases.runners.modelling.utils.states.modelling_runner_states_base import ModellingRunnerStates
from library.pipeline.pipeline_manager import PipelineManager
from skopt.space import Real


class InTuningRunner(ModellingRunnerStates):
      def __init__(self, pipeline_manager: PipelineManager, save_plots: bool = False, save_path: str = None):
            super().__init__(pipeline_manager, save_plots, save_path)

      def _general_analysis(self):
            # Evaluating and storing models
            comments = "Cate will definetely not like this?"
            self.pipeline_manager.all_pipelines_execute(methodName="modelling.evaluate_and_store_models", 
                                                       exclude_category="baseline",
                                                       comments=comments, 
                                                       current_phase=self.pipeline_manager.pipeline_state)
            
            
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
   
            importances_dfs = self.pipeline_manager.pipelines_analysis.plot_feature_importance(save_plots=self.save_plots,
                                                                                                save_path=self.save_path)

            return metrics_df.to_dict(), residuals, confusion_matrices, importances_dfs

      def _get_grid_space(self):
            rf_grid = {
                  'n_estimators': [50, 100, 150, 200], 
                  'max_depth': [None, 10, 20, 30], 
                  'min_samples_split': [2, 5, 10], 
                  'min_samples_leaf': [1, 2, 4,]
            }

            dt_grid = {
                  'criterion': ['gini', 'entropy'],
                  'max_depth': [None, 10, 20, 30],
                  'min_samples_split': [2, 5, 10],
                  'min_samples_leaf': [1, 2, 5],
                  'max_features': [None, 'sqrt', 'log2'],
                  'ccp_alpha': [0.0, 0.01, 0.1]

            } 

            gnb_grid = {
                  'var_smoothing': Real(1e-12, 1e-6, prior='log-uniform')
            }

            return rf_grid, dt_grid, gnb_grid
      
      def _get_grid_search_params(self):
            rf_grid, dt_grid, gnb_grid = self._get_grid_space()
            modelNameToOptimizer = {
                  "Random Forest": {
                        "optimizer_type": "bayes",
                        "param_grid": rf_grid,
                        "max_iter": 1
                  },
                  "Decision Tree": {
                        "optimizer_type": "bayes",
                        "param_grid": dt_grid,
                        "max_iter": 1
                  },
                  "Naive Bayes": {
                        "optimizer_type": "bayes",
                        "param_grid": gnb_grid,
                        "max_iter": 1
                  },
                  "Feed Forward Neural Network": {
                        "optimizer_type": "bayes_nn",
                        "param_grid": None, # its hardcoded
                        "max_iter": 2,
                        "epochs": 2
                  }
            }
            return modelNameToOptimizer
      def run(self):
            self.pipeline_manager.pipeline_state = "in"
            print("In tuning runner")
            # Fitting models
            modelNameToOptimizer = self._get_grid_search_params()
            optimized_models = self.pipeline_manager.all_pipelines_execute(methodName="modelling.fit_models", 
                                                                           current_phase=self.pipeline_manager.pipeline_state,
                                                                           modelNameToOptimizer=modelNameToOptimizer)
            general_analysis_results = self._general_analysis()
            return general_analysis_results
