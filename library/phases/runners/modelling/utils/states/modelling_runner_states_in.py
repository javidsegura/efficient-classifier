
from library.phases.runners.modelling.utils.states.modelling_runner_states_base import ModellingRunnerStates
from library.pipeline.pipeline_manager import PipelineManager
from skopt.space import Real

from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier


class InTuningRunner(ModellingRunnerStates):
      def __init__(self, pipeline_manager: PipelineManager, save_plots: bool = False, save_path: str = None):
            super().__init__(pipeline_manager, save_plots, save_path)

      def _general_analysis(self):
            # Evaluating and storing models
            comments = self.pipeline_manager.variables["modelling_runner"]["model_assesment"]["comments"]
            self.pipeline_manager.all_pipelines_execute(methodName="modelling.evaluate_and_store_models", 
                                                       exclude_category="baseline",
                                                       comments=comments, 
                                                       current_phase=self.pipeline_manager.pipeline_state)
            
            
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
            
            # Residual analyisis 
            residuals, confusion_matrices = self.pipeline_manager.pipelines_analysis.plot_confusion_matrix(save_plots=self.save_plots,
                                                                                                          save_path=self.save_path)
   
            importances_dfs = self.pipeline_manager.pipelines_analysis.plot_feature_importance(save_plots=self.save_plots,
                                                                                                save_path=self.save_path)

            return metrics_df.to_dict(), residuals, confusion_matrices, importances_dfs

      def _get_grid_space(self):
            rf_grid = {
                  'n_estimators': self.pipeline_manager.variables["modelling_runner"]["hyperparameters"]["grid_space"]["random_forest"]["n_estimators"], 
                  'max_depth': self.pipeline_manager.variables["modelling_runner"]["hyperparameters"]["grid_space"]["random_forest"]["max_depth"], 
                  'min_samples_split': self.pipeline_manager.variables["modelling_runner"]["hyperparameters"]["grid_space"]["random_forest"]["min_samples_split"], 
                  'min_samples_leaf': self.pipeline_manager.variables["modelling_runner"]["hyperparameters"]["grid_space"]["random_forest"]["min_samples_leaf"]
            }

            dt_grid = {
                  'criterion': self.pipeline_manager.variables["modelling_runner"]["hyperparameters"]["grid_space"]["decision_tree"]["criterion"],
                  'max_depth': self.pipeline_manager.variables["modelling_runner"]["hyperparameters"]["grid_space"]["decision_tree"]["max_depth"],
                  'min_samples_split': self.pipeline_manager.variables["modelling_runner"]["hyperparameters"]["grid_space"]["decision_tree"]["min_samples_split"],
                  'min_samples_leaf': self.pipeline_manager.variables["modelling_runner"]["hyperparameters"]["grid_space"]["decision_tree"]["min_samples_leaf"],
                  'max_features': self.pipeline_manager.variables["modelling_runner"]["hyperparameters"]["grid_space"]["decision_tree"]["max_features"],
                  'ccp_alpha': self.pipeline_manager.variables["modelling_runner"]["hyperparameters"]["grid_space"]["decision_tree"]["ccp_alpha"]

            } 

            gnb_grid = { # has to be hard-coded => Real datatype is not supported
                  'var_smoothing': Real(1e-12, 1e-6, prior='log-uniform')
            }

            return rf_grid, dt_grid, gnb_grid
      
      def _get_grid_search_params(self):
            rf_grid, dt_grid, gnb_grid = self._get_grid_space()
            modelNameToOptimizer = {
                  "Random Forest": {
                        "optimizer_type": "bayes",
                        "param_grid": rf_grid,
                        "max_iter": self.pipeline_manager.variables["modelling_runner"]["hyperparameters"]["tuner_params"]["max_iter"]
                  },
                  "Decision Tree": {
                        "optimizer_type": "bayes",
                        "param_grid": dt_grid,
                        "max_iter": self.pipeline_manager.variables["modelling_runner"]["hyperparameters"]["tuner_params"]["max_iter"]
                  },
                  "Naive Bayes": {
                        "optimizer_type": "bayes",
                        "param_grid": gnb_grid,
                        "max_iter": self.pipeline_manager.variables["modelling_runner"]["hyperparameters"]["tuner_params"]["max_iter"]
                  },
                  "Feed Forward Neural Network": {
                        "optimizer_type": "bayes_nn",
                        "param_grid": None, # its hardcoded
                        "max_iter": self.pipeline_manager.variables["modelling_runner"]["hyperparameters"]["tuner_params"]["max_iter"],
                        "epochs": self.pipeline_manager.variables["modelling_runner"]["hyperparameters"]["tuner_params"]["epochs"]
                  }
            }
            return modelNameToOptimizer
      
      def _set_up_stacking_model(self, optimized_models):
            estimators = []

            for pipelineName, results in optimized_models["not_baseline"].items():
                  if isinstance(results, dict):
                        for modelName, modelObject in results.items():
                              estimators.append((modelName, modelObject))
            
            #Stacking model
            stackingModel = StackingClassifier(
                  estimators=estimators,
                  final_estimator=DecisionTreeClassifier(),
                  cv="prefit",
                  verbose=3
            )

            self.pipeline_manager.pipelines["not_baseline"]["stacking"].modelling.list_of_models["Stacking"].tuning_states["in"].assesment["model_sklearn"] = stackingModel
            self.pipeline_manager.pipelines["not_baseline"]["stacking"].modelling.list_of_models["Stacking"].tuning_states["post"].assesment["model_sklearn"] = stackingModel

            self.pipeline_manager.pipelines["not_baseline"]["stacking"].modelling.list_of_models["Stacking"].tuning_states["in"].model_sklearn = stackingModel
            self.pipeline_manager.pipelines["not_baseline"]["stacking"].modelling.list_of_models["Stacking"].tuning_states["post"].model_sklearn = stackingModel

            all_pipelines_to_exclude = []

            for pipelineName, pipelineObject in self.pipeline_manager.pipelines["not_baseline"].items():
                  if pipelineName == "stacking":
                        continue
                  all_pipelines_to_exclude.append(pipelineName)
            
            self.pipeline_manager.all_pipelines_execute(methodName="modelling.fit_models", 
                                       current_phase=self.pipeline_manager.pipeline_state,
                                       exclude_category="baseline",
                                       exclude_pipeline_names=all_pipelines_to_exclude
                                       )

            

      def run(self):
            self.pipeline_manager.pipeline_state = "in"
            print("In tuning runner")
            # Fitting models
            modelNameToOptimizer = self._get_grid_search_params()
            optimized_models = self.pipeline_manager.all_pipelines_execute(methodName="modelling.fit_models", 
                                                                           exclude_pipeline_names=["stacking"],
                                                                           current_phase=self.pipeline_manager.pipeline_state,
                                                                           modelNameToOptimizer=modelNameToOptimizer)
            self._set_up_stacking_model(optimized_models)
            general_analysis_results = self._general_analysis()
            return general_analysis_results
