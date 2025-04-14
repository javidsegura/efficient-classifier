from library.phases.model_selection.results_analysis.results_df import ResultsDF
from library.phases.model_selection.model.shallow.classifier import Classifier
from library.phases.model_selection.model.shallow.regressor import Regressor
from library.phases.model_selection.model.model import Model
from library.phases.dataset.dataset import Dataset

from library.phases.model_selection.results_analysis.result_analysis import PreTuningResultAnalysis, InTuningResultAnalysis, PostTuningResultAnalysis

import concurrent.futures
import pandas as pd
import time

class ModelSelection:
      def __init__(self, dataset: Dataset, results_path: str, metrics_to_evaluate: list[str]):
            self.results_df = ResultsDF(results_path, metrics_to_evaluate, dataset)
            self.list_of_models = {}
            self.dataset = dataset
            self.models_to_exclude = []
            self.phaseProcess = {}
            self.comments = ""
            self.results_analysis = {
                  "pre_tuning": PreTuningResultAnalysis(self.results_df, self.dataset),
                  "in_tuning": InTuningResultAnalysis(self.results_df, self.dataset),
                  "post_tuning": PostTuningResultAnalysis(self.results_df, self.dataset),
            }

      def add_model(self, model_name: str, model_sklearn: object):
            new_model = None
            if self.dataset.modelTask == "classification":
                  new_model = Classifier(model_name, model_sklearn, results_header=self.results_df.header, dataset=self.dataset)
            elif self.dataset.modelTask == "regression":
                  new_model = Regressor(model_name, model_sklearn, results_header=self.results_df.header, dataset=self.dataset)

            self.list_of_models[model_name] = new_model
      
      def _fit_and_predict(self, modelName, modelObject):
            modelObject.fit(modelName=modelName)
            modelObject.predict(modelName=modelName)
            return modelName, modelObject

      def fit_models(self):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                  # Submit all model fitting tasks to the executor
                  future_to_model = [executor.submit(self._fit_and_predict, modelName, modelObject) for modelName, modelObject in self.list_of_models.items() if modelName not in self.models_to_exclude]
                  
                  for future in concurrent.futures.as_completed(future_to_model):
                        modelName, model = future.result() 
                        self.list_of_models[modelName] = model # update results
            print("All models have been fitted and made predictions in parallel.")

      def _evaluate_model(self, modelName, modelObject):
            assesment = modelObject.evaluate(modelName=modelName)
            return modelName, modelObject, assesment

      def evaluate_models(self, phaseProcess: dict[str, bool], comments: str):
            self.phaseProcess = phaseProcess 
            self.comments = comments
            assert self.phaseProcess and self.comments, "Either phaseProcess and comments must be provided"

            for modelName, modelObject in self.list_of_models.items():
                  if self.phaseProcess["is_HyperParameterOptimization_done"]:
                        modelObject.currentPhase = "in"
                  else:
                        modelObject.currentPhase = "pre"

            with concurrent.futures.ProcessPoolExecutor() as executor:
                  # Submit all model fitting tasks to the executor
                  future_to_model = [executor.submit(self._evaluate_model, modelName, modelObject) for modelName, modelObject in self.list_of_models.items() if modelName not in self.models_to_exclude]
                  
                  for future in concurrent.futures.as_completed(future_to_model):
                        modelName, modelObject, assesment = future.result() 
                        self.list_of_models[modelName] = modelObject # update results
                        self.list_of_models[modelName].tuning_states[modelObject.currentPhase].assesment = assesment
         
                        
            print("All models have been evaluated.")
            model_logs = self.results_df.store_results(list_of_models=self.list_of_models, 
                                                       phaseProcess=self.phaseProcess,
                                                      comments=self.comments,
                                                      models_to_exclude=self.models_to_exclude)
            return pd.DataFrame(model_logs)
      
      def constrast_results(self):
            for modelName, modelObject in self.list_of_models.items():
                  modelObject._constrast_sets_predictions()
      
      def _optimize_model(self, modelName: str, optimization_params: dict):
            
            self.list_of_models[modelName].optimizer_type = optimization_params["optimizer_type"]
            cv_results_df, best_model, assesment, optimizer = self.list_of_models[modelName].optimize(param_grid=optimization_params["param_grid"], max_iter=optimization_params["max_iter"])
            print("I am done")
            return cv_results_df, best_model, assesment, optimizer

      def models_optimization(self, modelNameToOptimizer: dict[str, dict[str, dict]]):
            cv_results_dataframes = {}
            best_models = {}

            print(f"Models to optimize: {list(modelNameToOptimizer.keys())}")

            for modelName, modelObject in self.list_of_models.items():
                  if modelName in modelNameToOptimizer:
                        modelObject.currentPhase = "in_tuning"
                        print(f"In-tuning model: {modelName} \n {modelObject.inTuningState.assesment}")
                        assert modelObject.inTuningState.assesment["status"] == "in_tuning", f"Model {modelName} is not in in-tuning phase"
            
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                  future_to_model_name = {}
                  for modelName, optimization_params in modelNameToOptimizer.items():
                        future = executor.submit(self._optimize_model, modelName, optimization_params)
                        future_to_model_name[future] = modelName
                  
                  # Process results as they complete
                  for future in concurrent.futures.as_completed(future_to_model_name.keys()):
                        modelName = future_to_model_name[future]  # Get the model name for this future
                        print(f"Getting results for {modelName}")
                        cv_results_df, best_model, assesment, optimizer = future.result()
                        print(f"Results for {modelName} obtained")
                        cv_results_dataframes[modelName] = cv_results_df
                        best_models[modelName] = best_model
                        self.list_of_models[modelName].inTuningState.assesment = assesment
                        self.list_of_models[modelName].optimizer = optimizer
                        print(f"Results for {modelName} stored")
                  

            print("All models have been optimized in parallel. Gonna store results.")
            start_time = time.time()
            
            print(f"[DEBUG {time.time() - start_time:.2f}s] Starting to print model assessments")
            for modelName, modelObject in self.list_of_models.items():
                  # List all the assesment 
                  if modelObject.currentPhase == "in_tuning":
                        print(f"Model {modelName} in-tuning assesment: {modelObject.inTuningState.assesment}")
            
            print(f"[DEBUG {time.time() - start_time:.2f}s] About to call store_results")
            model_logs = self.results_df.store_results(list_of_models=self.list_of_models, phaseProcess=self.phaseProcess, comments=self.comments, models_to_include=list(modelNameToOptimizer.keys()))
            print(f"[DEBUG {time.time() - start_time:.2f}s] store_results completed")

            return cv_results_dataframes, best_models, pd.DataFrame(model_logs)

      def final_model(self, best_model: object, finalModelName: str, baseModelName: str, plot: bool = True):
            """
            TO BE DONE: 
            - Add parallelism to fit, predict, evalute 
            """
            finalModelObj = self.list_of_models[finalModelName]
            finalModelObj.currentPhase = "post_tuning"
            finalModelObj.postTuningState.model_sklearn = finalModelObj.model_sklearn = best_model
            finalModelObj.fit(finalModelName)
            finalModelObj.predict(finalModelName)
            finalModelObj.evaluate(finalModelName)

            baselineModelObj = self.list_of_models[baseModelName]
            baselineModelObj.currentPhase = "post_tuning"
            baselineModelObj.fit(baseModelName)
            baselineModelObj.predict(baseModelName)
            baselineModelObj.evaluate(baseModelName)
            
            if plot:
                  ...

            return finalModelObj, baselineModelObj
      
      def plot_bias_variance(self, finalModelName: str):
            ...
      
      def plot_test_metrics(self, finalModelName: str, baseModelName: str):
            ...

