from library.pipeline.pipeline import Pipeline

from copy import deepcopy
import concurrent.futures
import threading
from typing import Any
import os

from library.pipeline.analysis.pipelines_analysis import PipelinesAnalysis
import joblib

class PipelineManager:
      """
      Trains all pipelines. 
      Evaluates all pipelines
      
      """
      def __init__(self, pipelines: dict[str, dict[str, Pipeline]]):
            self.pipelines = pipelines
            self.pipelines_analysis = PipelinesAnalysis(pipelines)
            self._pipeline_state = None # Can only take upon "pre", "in", "post"
            self.best_performing_model = None

      @property
      def pipeline_state(self):
            return self._pipeline_state

      @pipeline_state.setter
      def pipeline_state(self, pipeline_state: str):
            assert pipeline_state in ["pre", "in", "post"], "Pipeline state must be one of the following: pre, in, post"
            self._pipeline_state = pipeline_state
            self.pipelines_analysis.phase = pipeline_state
      
      def create_pipeline_divergence(self, category: str, pipelineName: str, print_results: bool = False):
            """
            Compares two pipelines and returns the difference in their metrics.
            """
            assert category in self.pipelines, "Category not found"
            assert pipelineName in self.pipelines[category], "Pipeline name not found"

            priorPipeline = self.pipelines[category][pipelineName]
            newPipeline = deepcopy(priorPipeline)
            self.pipelines[category][pipelineName] = newPipeline
            if print_results:
                  print(f"Pipeline {pipelineName} in category {category} has diverged\n Pipeline schema is now: {self.pipelines}")
            return newPipeline
      
      def all_pipelines_execute(self, methodName: str, verbose: bool = True, exclude_baseline: bool = False, **kwargs):
            """
            Executes a method for all pipelines using threading for parallelization.
            Method name can include dot notation for nested attributes (e.g. "model.fit")

            Note for verbose:
            - If u dont see a given pipeline in the results, it is because it has already been processed (its a copy of another pipeline)
            """
            results = {}
            processed_pipelines = set()
            results_lock = threading.Lock()  # Thread-safe lock for updating shared results

            def execute_pipeline_method(category: str, pipelineName: str, pipeline: Any) -> None:
                  try:
                        # Handle nested attribute access
                        obj = pipeline
                        for attr in methodName.split('.')[:-1]:
                              obj = getattr(obj, attr)
                        method = getattr(obj, methodName.split('.')[-1])
                        result = method(**kwargs)

                        # Thread-safe update of results
                        with results_lock:
                              if category not in results:
                                    results[category] = {}
                              results[category][pipelineName] = result
                              if verbose:
                                    print(f"Pipeline {pipelineName} in category {category} has executed {methodName}. Result is: {result}")
                  except Exception as e:
                        print(f"Error executing {methodName} on pipeline {pipelineName} in {category}: {str(e)}")
                        raise

            # Create thread pool
            with concurrent.futures.ThreadPoolExecutor() as executor:
                  futures = []
                  if exclude_baseline:
                        categories = ["not-baseline"]
                  else:
                        categories = self.pipelines.keys()
                  # Submit tasks for each unique pipeline
                  for category in categories:
                        if category not in results:
                              results[category] = {}
                      
                        for pipelineName, pipeline in self.pipelines[category].items():
                              if id(pipeline) not in processed_pipelines:
                                    processed_pipelines.add(id(pipeline))
                                    futures.append(
                                          executor.submit(
                                                execute_pipeline_method,
                                                category,
                                                pipelineName,
                                                pipeline
                                          )
                                    )

                  # Wait for all tasks to complete and handle any exceptions
                  concurrent.futures.wait(futures)
                  for future in futures:
                        if future.exception():
                              raise future.exception()

            return results
      
      def select_best_performing_model(self, metric: str):
            """
            Selects the best performing model based on the classification report
            """
            assert metric in self.pipelines_analysis.merged_report.columns, f"Metric not found. Columns are: {self.pipelines_analysis.merged_report.columns}"
            metric_df = self.pipelines_analysis.merged_report[metric]
            model_names = metric_df.loc["modelName"].tolist()  # Last row: model names
            metric_df = metric_df.drop(index='modelName')     # Drop last row
            metric_df.columns = model_names            # Rename columns to model names

            weighted_avg = metric_df.loc['weighted avg']
            filtered = weighted_avg[~weighted_avg.index.str.endswith('_train')] # Remove training models
            best_model = filtered.idxmax()
            best_score = filtered.max()
            self.best_performing_model = {
                  "pipelineName": None,
                  "modelName": best_model,
            }
            self.pipelines_analysis.best_performing_model = self.best_performing_model
            print(f"Best performing model: {best_model} with {metric} {best_score:.4f}")

            # Overwrite the sklearn_model for the post state 
            for pipeline in self.pipelines["not-baseline"]:
                  for model in self.pipelines["not-baseline"][pipeline].model_selection.list_of_models:
                        if model == best_model:
                              self.pipelines["not-baseline"][pipeline].model_selection.list_of_models[model].tuning_states["post"].model_sklearn = self.pipelines["not-baseline"][pipeline].model_selection.list_of_models[model].tuning_states["in"].assesment["model_sklearn"]
                              self.best_performing_model["pipelineName"] = pipeline

            return best_model, best_score
      
      def fit_final_models(self):
            # Fit models 
            self.pipelines["not-baseline"][self.best_performing_model["pipelineName"]].model_selection.fit_models(current_phase="post", 
                                                                                                                  best_model_name=self.best_performing_model["modelName"],
                                                                                                                  baseline_model_name=None)
            for pipeline in self.pipelines["baseline"]:
                  for model in self.pipelines["baseline"][pipeline].model_selection.list_of_models:
                        self.pipelines["baseline"][pipeline].model_selection.fit_models(current_phase="post", 
                                                                                  best_model_name=None, 
                                                                                  baseline_model_name=model)
      
      def evaluate_store_final_models(self):
            self.pipelines["not-baseline"][self.best_performing_model["pipelineName"]].model_selection.evaluate_and_store_models(
                  current_phase="post", 
                  comments=None,
                  best_model_name=self.best_performing_model["modelName"], 
                  baseline_model_name=None)
            
            for pipeline in self.pipelines["baseline"]:
                  for model in self.pipelines["baseline"][pipeline].model_selection.list_of_models:
                        self.pipelines["baseline"][pipeline].model_selection.evaluate_and_store_models(
                              current_phase="post", 
                              comments=None,
                              best_model_name=None, 
                              baseline_model_name=model)
                        
      def store_best_performing_sklearn_model_object(self, fileName: str="best_performing_model.pkl", includeModelObject: bool = False):
            """
            Stores the best performing model object
            """
            assert self.best_performing_model is not None, "Best performing model not found"
            assert self.best_performing_model["pipelineName"] is not None, "Pipeline name not found"
            assert self.best_performing_model["modelName"] is not None, "Model name not found"
            if includeModelObject:
                  pipelineName = self.best_performing_model["pipelineName"]
                  modelName = self.best_performing_model["modelName"]
                  joblib.dump(self.pipelines["not-baseline"][pipelineName].model_selection.list_of_models[modelName], fileName) # stores the model object
                  joblib.dump(self.best_performing_model["modelName"].tuning_states["post"].assesment["model_sklearn"], fileName) # store the sklearn model object
            else:
                  joblib.dump(self.best_performing_model["modelName"].tuning_states["post"].assesment["model_sklearn"], fileName)

      def store_pipelines(self, fileName: str="pipelines.pkl"):
            """
            Stores the pipelines
            """
            os.makedirs("stored_objects", exist_ok=True)

            joblib.dump(self.pipelines, fileName)


