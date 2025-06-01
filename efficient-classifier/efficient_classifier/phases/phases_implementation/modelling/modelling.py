from efficient_classifier.phases.phases_implementation.modelling.results_analysis.results_df import ResultsDF
from efficient_classifier.phases.phases_implementation.modelling.shallow.model_definition.model_types.classifier import Classifier
from efficient_classifier.phases.phases_implementation.modelling.shallow.model_definition.model_types.regressor import Regressor
from efficient_classifier.phases.phases_implementation.modelling.shallow.model_definition.model_base import Model
from efficient_classifier.phases.phases_implementation.dataset.dataset import Dataset

from efficient_classifier.phases.phases_implementation.modelling.results_analysis.result_analysis import PreTuningResultAnalysis, InTuningResultAnalysis, PostTuningResultAnalysis

import concurrent.futures
import pandas as pd



class Modelling:
      def __init__(self, dataset: Dataset, model_results_path: str):
            self.results_df = ResultsDF(model_results_path, dataset)
            self.list_of_models = {}
            self.dataset = dataset
            self._models_to_exclude = []
            self.results_analysis = {
                  "pre": PreTuningResultAnalysis(phase_results_df= pd.DataFrame()),
                  "in": InTuningResultAnalysis(phase_results_df= pd.DataFrame()),
                  "post": PostTuningResultAnalysis(phase_results_df= pd.DataFrame()),
            }

      # 0) Attibutes logic
      @property
      def models_to_exclude(self):
            return self._models_to_exclude
      
      @models_to_exclude.setter
      def models_to_exclude(self, value: list[str]):
            for modelName in value:
                  assert modelName in self.list_of_models, f"Model {modelName} not found in list of models"
            self._models_to_exclude = value

      # 1) Adding models
      def add_model(self, model_name: str, model_sklearn: object, model_type: str = "classical"): 
            """
            Warning: as soon you add a model you cant modify the dataset 
            
            """
            assert model_type in ["classical", "neural_network", "stacking"]
            new_model = None
            if self.dataset.modelTask == "classification":
                  new_model = Classifier(model_name, model_sklearn, model_type=model_type, results_header=self.results_df.header, dataset=self.dataset)
            elif self.dataset.modelTask == "regression":
                  new_model = Regressor(model_name, model_sklearn, model_type=model_type, results_header=self.results_df.header, dataset=self.dataset)

            self.list_of_models[model_name] = new_model
      
      # 2) Fitting, predicting and optimizing models
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

            if modelObject.model_type == "neural_network":
                  epochs = optimization_params.get("epochs", None)
                  modelObject.fit(modelName=modelName, current_phase=current_phase,
                              param_grid=optimization_params["param_grid"],
                              max_iter=optimization_params["max_iter"],
                              optimizer_type=optimization_params["optimizer_type"],
                              model_object=modelObject,
                              epochs=epochs
                              )
            else:
                  modelObject.fit(modelName=modelName, current_phase=current_phase,
                              param_grid=optimization_params["param_grid"],
                              max_iter=optimization_params["max_iter"],
                              optimizer_type=optimization_params["optimizer_type"],
                              model_object=modelObject
                              )
            modelObject.predict(modelName=modelName, current_phase=current_phase)
            print(f"Optimized model {modelName}")
            # Setting the final model to be the tuned one
            modelObject.tuning_states["post"].assesment["model_sklearn"] = modelObject.tuning_states["in"].assesment["model_sklearn"]
            return modelName, modelObject

      def fit_models(self, current_phase: str, **kwargs):
            """
            Note: for the in phase, we need to optimize the models in parallel except for the bayes_nn models, which we need to optimize sequentially (keras-specific reasons).
            
            """
            assert current_phase in ["pre", "in", "post"], "Current phase must be one of the tuning states"
            print(f"Gonna start fitting models in {current_phase} phase")
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
                        optimized_models = {}

                        # Separate models
                        bayes_nn_models = []
                        other_models = []

                        for modelName, optimization_params in modelNameToOptimizer.items():
                                    if modelName not in list(self.list_of_models.keys()):
                                          continue
                                    if modelName in self.models_to_exclude:
                                          continue
                                    if optimization_params.get("optimizer_type") == "bayes_nn":
                                          bayes_nn_models.append((modelName, optimization_params))
                                    else:
                                          other_models.append((modelName, optimization_params))

                        # Run non-bayes_nn models in process pool
                        for modelName, optimization_params in other_models:
                                    print(f"Optimizing model {modelName}")
                                    modelObject = self.list_of_models[modelName]
                                    future = executor.submit(self._optimize_model, modelName, modelObject, current_phase, optimization_params)
                                    future_to_model.append(future)

                        for future in concurrent.futures.as_completed(future_to_model):
                                    modelName, modelObject = future.result()
                                    self.list_of_models[modelName] = modelObject
                                    optimized_models[modelName] = modelObject.tuning_states["in"].assesment["model_sklearn"]

                        # Run bayes_nn models sequentially (outside process pool)
                        for modelName, optimization_params in bayes_nn_models:
                                    print(f"Optimizing bayes_nn model {modelName}")
                                    modelObject = self.list_of_models[modelName]
                                    # Direct call, not via executor
                                    modelName, modelObject = self._optimize_model(modelName, modelObject, current_phase, optimization_params)
                                    self.list_of_models[modelName] = modelObject
                                    optimized_models[modelName] = modelObject.tuning_states["in"].assesment["model_sklearn"]

                        return optimized_models
                  elif current_phase == "post":
                              # Exclude neural-nets fro conccurent
                              best_model_name, baseline_model_name = kwargs.get("best_model_name", None), kwargs.get("baseline_model_name", None)
                              assert (best_model_name is not None) or (baseline_model_name is not None), "You must provide at least one of the best or baseline model"
                              future_to_model = []

                              if best_model_name:
                                    if self.list_of_models[best_model_name].optimizer_type == "bayes_nn":
                                          modelName, modelObject = self._fit_and_predict(best_model_name, self.list_of_models[best_model_name], current_phase)
                                          self.list_of_models[best_model_name] = modelObject
                                    else:
                                          future = executor.submit(self._fit_and_predict, best_model_name, self.list_of_models[best_model_name], current_phase)
                                          future_to_model.append(future)
                              if baseline_model_name:
                                    future = executor.submit(self._fit_and_predict, baseline_model_name, self.list_of_models[baseline_model_name], current_phase)
                                    future_to_model.append(future)

                              for future in concurrent.futures.as_completed(future_to_model):
                                    modelName, modelObject = future.result()
                                    self.list_of_models[modelName] = modelObject

      # 3) Evaluating and store model results 
      def _evaluate_model(self, modelName, modelObject, current_phase: str):
            print(f"Evaluating model {modelName}")
            modelObject.evaluate(modelName=modelName, current_phase=current_phase)
            return modelName, modelObject

      def evaluate_and_store_models(self, current_phase: str, **kwargs) -> pd.DataFrame | None:
            """
            It asses each model and stores the results in the results_df.

            Parameters
            ----------
            comments: str
                  The comments to store in the results_df (and in disk)
            current_phase: str
                  The current phase of the modelling
            kwargs: dict
                  Additional keyword arguments defining phase-specific parameters

            Returns
            -------
            pd.DataFrame or None
                  The results of the evaluation
            """

            # Separate "bayes_nn" models from others. This is because bayes_nn cant use parallel processing (for some keras-specific reasons)
            bayes_nn_models = []
            other_models = []

            if current_phase != "post":
                  # Split models based on optimizer type
                  for modelName, modelObject in self.list_of_models.items():
                        if modelName in self.models_to_exclude:
                              continue
                        if hasattr(modelObject, 'optimizer_type') and modelObject.optimizer_type == "bayes_nn":
                              bayes_nn_models.append((modelName, modelObject))
                        else:
                              other_models.append((modelName, modelObject))
            else:
                  # Handle post-tuning phase
                  best_model_name = kwargs.get("best_model_name")
                  baseline_model_name = kwargs.get("baseline_model_name")
                  assert (best_model_name is not None) or (baseline_model_name is not None), \
                        "You must provide at least one of the best or baseline model"
                  
                  # Check if best/baseline models are bayes_nn
                  if best_model_name:
                        model = self.list_of_models[best_model_name]
                        if hasattr(model, 'optimizer_type') and model.optimizer_type == "bayes_nn":
                              bayes_nn_models.append((best_model_name, model))
                        else:
                              other_models.append((best_model_name, model))
                  
                  if baseline_model_name:
                        model = self.list_of_models[baseline_model_name]
                        if hasattr(model, 'optimizer_type') and model.optimizer_type == "bayes_nn":
                              bayes_nn_models.append((baseline_model_name, model))
                        else:
                              other_models.append((baseline_model_name, model))

            # Process non-bayes_nn models in parallel
            with concurrent.futures.ProcessPoolExecutor() as executor:
                  future_to_model = [
                        executor.submit(self._evaluate_model, modelName, modelObject, current_phase)
                        for modelName, modelObject in other_models
                  ]
                  
                  for future in concurrent.futures.as_completed(future_to_model):
                        modelName, modelObject = future.result()
                        self.list_of_models[modelName] = modelObject

            # Process bayes_nn models sequentially
            for modelName, modelObject in bayes_nn_models:
                  modelName, modelObject = self._evaluate_model(modelName, modelObject, current_phase)
                  self.list_of_models[modelName] = modelObject
            
            # Store results and update analysis
            model_logs = self.results_df.store_results(
                  list_of_models=self.list_of_models,
                  current_phase=current_phase,
                  models_to_exclude=self.models_to_exclude
            )
            if model_logs is not None:
                  model_logs = pd.DataFrame(model_logs)
                  self.results_analysis[current_phase].phase_results_df = model_logs
                  return model_logs
            else:
                  print(f"NO MODEL LOGS TO STORE FOR {current_phase} PHASE")
                  return None
      


