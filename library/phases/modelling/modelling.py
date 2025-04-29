from library.phases.modelling.results_analysis.results_df import ResultsDF
from library.phases.modelling.shallow.model_definition.model_types.classifier import Classifier
from library.phases.modelling.shallow.model_definition.model_types.regressor import Regressor
from library.phases.modelling.shallow.model_definition.model_base import Model
from library.phases.dataset.dataset import Dataset

from library.phases.modelling.results_analysis.result_analysis import PreTuningResultAnalysis, InTuningResultAnalysis, PostTuningResultAnalysis

import concurrent.futures
import pandas as pd
import time

class Modelling:
      def __init__(self, dataset: Dataset, results_path: str):
            self.results_df = ResultsDF(results_path, dataset)
            self.list_of_models = {}
            self.dataset = dataset
            self._models_to_exclude = []
            self.comments = ""
            self.results_analysis = {
                  "pre": PreTuningResultAnalysis(phase_results_df= pd.DataFrame()),
                  "in": InTuningResultAnalysis(phase_results_df= pd.DataFrame()),
                  "post": PostTuningResultAnalysis(phase_results_df= pd.DataFrame()),
            }

      @property
      def models_to_exclude(self):
            return self._models_to_exclude
      
      @models_to_exclude.setter
      def models_to_exclude(self, value: list[str]):
            for modelName in value:
                  assert modelName in self.list_of_models, f"Model {modelName} not found in list of models"
            self._models_to_exclude = value

      def add_model(self, model_name: str, model_sklearn: object, model_type: str = "classical"):
            new_model = None
            if self.dataset.modelTask == "classification":
                  new_model = Classifier(model_name, model_sklearn, model_type=model_type, results_header=self.results_df.header, dataset=self.dataset)
            elif self.dataset.modelTask == "regression":
                  new_model = Regressor(model_name, model_sklearn, model_type=model_type, results_header=self.results_df.header, dataset=self.dataset)

            self.list_of_models[model_name] = new_model
      
      def _fit_and_predict(self, modelName, modelObject: Model, current_phase: str):
            modelObject.fit(modelName=modelName, current_phase=current_phase)
            modelObject.predict(modelName=modelName, current_phase=current_phase)
            print(f"Fitted and predicted model {modelName}")
            return modelName, modelObject

      def _optimize_model(self, 
                          modelName: str, 
                          modelObject: Model, 
                          current_phase: str,
                          optimization_params: dict):
            assert current_phase == "in", "Optimize model can only be used in the 'in' phase"
            modelObject.optimizer_type = optimization_params["optimizer_type"]
      
            modelObject.fit(modelName=modelName, current_phase=current_phase,
                            param_grid=optimization_params["param_grid"],
                            max_iter=optimization_params["max_iter"],
                            optimizer_type=optimization_params["optimizer_type"])
            modelObject.predict(modelName=modelName, current_phase=current_phase)
            print(f"Optimized model {modelName}")
            return modelName, modelObject

      def fit_models(self, current_phase: str, **kwargs):
            assert current_phase in ["pre", "in", "post"], "Current phase must be one of the tuning states"
            with concurrent.futures.ProcessPoolExecutor() as executor:
                  # Submit all model fitting tasks to the executor
                  if current_phase == "pre":
                        future_to_model = [executor.submit(self._fit_and_predict, modelName, modelObject, current_phase) for modelName, modelObject in self.list_of_models.items() if modelName not in self.models_to_exclude]
                  
                        for future in concurrent.futures.as_completed(future_to_model):
                              modelName, model = future.result() 
                              self.list_of_models[modelName] = model # update results
                  elif current_phase == "in":
                        modelNameToOptimizer = kwargs.get("modelNameToOptimizer", None)
                        assert modelNameToOptimizer is not None, "modelNameToOptimizer must be provided"
                        future_to_model = []
                        optimized_models = {} # Stores 'modelName: modelSklearn'
                         
                        for modelName, optimization_params in modelNameToOptimizer.items():
                              if modelName not in list(self.list_of_models.keys()):
                                    continue
                              if modelName in self.models_to_exclude:
                                    continue
                              print(f"Optimizing model {modelName}")
                              modelObject = self.list_of_models[modelName]
                              future = executor.submit(self._optimize_model, modelName, modelObject, current_phase, optimization_params)
                              future_to_model.append(future)
                        for future in concurrent.futures.as_completed(future_to_model):
                              modelName, modelObject = future.result()
                              self.list_of_models[modelName] = modelObject
                              optimized_models[modelName] = modelObject.tuning_states["in"].assesment["model_sklearn"]
                        return optimized_models
                  elif current_phase == "post":
                        best_model_name, baseline_model_name = kwargs.get("best_model_name", None), kwargs.get("baseline_model_name", None)
                        assert (best_model_name is not None) or (baseline_model_name is not None), "You must provide at least one of the best or baseline model"
                        future_to_model = []

                        if best_model_name:
                              future = executor.submit(self._fit_and_predict, best_model_name, self.list_of_models[best_model_name], current_phase)
                              future_to_model.append(future)
                        if baseline_model_name:
                              future = executor.submit(self._fit_and_predict, baseline_model_name, self.list_of_models[baseline_model_name], current_phase)
                              future_to_model.append(future)

                        for future in concurrent.futures.as_completed(future_to_model):
                              modelName, modelObject = future.result()
                              self.list_of_models[modelName] = modelObject

            print("All models have been fitted and made predictions in parallel.")

      def _evaluate_model(self, modelName, modelObject, current_phase: str):
            print(f"Evaluating model {modelName}")
            modelObject.evaluate(modelName=modelName, current_phase=current_phase)
            return modelName, modelObject

      def evaluate_and_store_models(self, comments: str, current_phase: str, **kwargs):
            if comments:
                  self.comments = comments
            assert self.comments, "comments must be provided"
            with concurrent.futures.ProcessPoolExecutor() as executor:
                  # Submit all model fitting tasks to the executor
                  if current_phase != "post":
                        future_to_model = [executor.submit(self._evaluate_model, modelName, modelObject, current_phase) for modelName, modelObject in self.list_of_models.items() if modelName not in self.models_to_exclude]
                        
                        for future in concurrent.futures.as_completed(future_to_model):
                              modelName, modelObject = future.result() 
                              self.list_of_models[modelName] = modelObject # update results
                  else:
                        best_model_name, baseline_model_name = kwargs.get("best_model_name", None), kwargs.get("baseline_model_name", None)
                        assert (best_model_name is not None) or (baseline_model_name is not None), "You must provide at least one of the best or baseline model"
                        future_to_model = []

                        if best_model_name:
                              future = executor.submit(self._evaluate_model, best_model_name, self.list_of_models[best_model_name], current_phase)
                              future_to_model.append(future)
                        if baseline_model_name:
                              future = executor.submit(self._evaluate_model, baseline_model_name, self.list_of_models[baseline_model_name], current_phase)
                              future_to_model.append(future)

                        for future in concurrent.futures.as_completed(future_to_model):
                              modelName, modelObject = future.result()
                              self.list_of_models[modelName] = modelObject
         
                        
            print("All models have been evaluated.")
            model_logs = self.results_df.store_results(list_of_models=self.list_of_models, 
                                                       current_phase=current_phase,
                                                      comments=self.comments,
                                                      models_to_exclude=self.models_to_exclude)
            self.results_analysis[current_phase].phase_results_df = pd.DataFrame(model_logs)

            return self.results_analysis[current_phase].phase_results_df
      

      def plot_convergence(self):
            for modelName, modelObject in self.list_of_models.items():
                  if hasattr(modelObject.tuning_states["in"], "optimizer"):
                        modelObject.tuning_states["in"].optimizer.plot_convergence()
