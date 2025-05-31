from efficient_classifier.pipeline.pipeline import Pipeline

from copy import deepcopy
import concurrent.futures
import threading
from typing import Any
import os

from efficient_classifier.pipeline.analysis.pipelines_analysis import PipelinesAnalysis
from efficient_classifier.pipeline.serialization_and_deserialization.serializer import SerializationPickle, SerializationJoblib
from efficient_classifier.pipeline.serialization_and_deserialization.deserializer import DeserializationPickle, DeserializationJoblib
from efficient_classifier.utils.miscellaneous.dag import DAG

class PipelineManager:
      """
      Trains all pipelines. 
      Evaluates all pipelines
      
      """
      def __init__(self, pipelines: dict[str, dict[str, Pipeline]], dag: DAG, serializer_type: str="joblib", variables: dict = None):
            """
            Initializes the pipeline manager.

            Parameters
            ----------
            pipelines: dict[str, dict[str, Pipeline]]
                  The pipelines to train. The keys are the category names, the values are the pipeline names and the pipelines.
            serializer_type: str
                  The type of serializer to use. Can be "pickle" or "joblib". Same action with different methods.
            variables: dict
                  The variables to use.

            Returns
            -------
            None
            """
            self.pipelines = pipelines
            self._pipeline_state = None # Can only take upon "pre", "in", "post"
            self.best_performing_model = None
            self.all_models = None
            self.variables = variables
            self.dag = dag

            # Sub-objects
            self.pipelines_analysis = PipelinesAnalysis(pipelines)
            if serializer_type == "pickle":
                  self.serializer = SerializationPickle()
                  self.deserializer = DeserializationPickle()
            elif serializer_type == "joblib":
                  self.serializer = SerializationJoblib()
                  self.deserializer = DeserializationJoblib()
            else:
                  raise ValueError(f"Invalid serializer type: {serializer_type}")

      @property
      def pipeline_state(self):
            return self._pipeline_state

      @pipeline_state.setter
      def pipeline_state(self, pipeline_state: str) -> None:
            """
            Every time we change the pipeline state, we also update the pipelines analysis phase automatically

            Parameters
            ----------
            pipeline_state: str
                  The pipeline state to set. Can be "pre", "in", "post".

            Returns
            -------
            None
            """
            assert pipeline_state in ["pre", "in", "post"], "Pipeline state must be one of the following: pre, in, post"
            self._pipeline_state = pipeline_state
            self.pipelines_analysis.phase = pipeline_state

      # 1) General functions 
      def create_pipeline_divergence(self, category: str, pipelineName: str, print_results: bool = False) -> Pipeline:
            """
            Originally all pipelines point to the same object. This function creates a copy at the moment and creates a new indepedent pipeline object. Changes to this pipeline now only affect this copy.

            Parameters
            ----------
            category: str
                  The category to create a divergence for.
            pipelineName: str
                  The pipeline name to create a divergence for.
            print_results: bool
                  Whether to print the results.

            Returns
            -------
            newPipeline: Pipeline
                  The new pipeline object.
            """
            assert category in self.pipelines, "Category not found"
            assert pipelineName in self.pipelines[category], "Pipeline name not found"

            priorPipeline = self.pipelines[category][pipelineName]
            newPipeline = deepcopy(priorPipeline)
            self.pipelines[category][pipelineName] = newPipeline
            if print_results:
                  print(f"Pipeline {pipelineName} in category {category} has diverged\n Pipeline schema is now: {self.pipelines}")
            return newPipeline
      
      def all_pipelines_execute(self, methodName: str, 
                                verbose: bool = False, 
                                exclude_category: str = "", 
                                exclude_pipeline_names: list[str] = [], 
                                **kwargs):
            """
            Executes a method for all pipelines using threading for parallelization.
            Method name can include dot notation for nested attributes (e.g. "model.fit")

            Note for verbose:
            - If u dont see a given pipeline in the results, it is because it has already been processed (its a copy of another pipeline)

            Parameters
            ----------
            methodName: str
                  The method to execute. As per defined in the phases implementation. 
            verbose: bool
                  Whether to print to stdout the results returned by the method.
            exclude_category: str
                  The category to exclude from the execution. (either baseline or not_baseline)
            exclude_pipeline_names: list[str]
                  The pipeline names to exclude from the execution.
            **kwargs: dict
                  Additional keyword arguments that are method-specific.

            Returns
            -------
            results: dict
                  The results of the execution.
            """
            if exclude_category:
                  assert exclude_category in self.pipelines.keys(), f"Exclude category must be one of the following: {self.pipelines.keys()}"
            results = {}
            processed_pipelines = set()
            results_lock = threading.Lock()  

            def execute_pipeline_method(category: str, pipelineName: str, pipeline: Any) -> None:
                  try:
                        obj = pipeline
                        for attr in methodName.split('.')[:-1]:
                              obj = getattr(obj, attr)
                        method = getattr(obj, methodName.split('.')[-1])
                        result = method(**kwargs)

                        with results_lock:
                              if category not in results:
                                    results[category] = {}
                              results[category][pipelineName] = result
                              if verbose:
                                    print(f"Pipeline {pipelineName} in category {category} has executed {methodName}. Result is: {result}")
                  except Exception as e:
                        print(f"Error executing {methodName} on pipeline {pipelineName} in {category}: {str(e)}")
                        raise

            with concurrent.futures.ThreadPoolExecutor() as executor:
                  futures = []
                  for category in self.pipelines.keys():
                        if category == exclude_category:
                              continue

                        if category not in results:
                              results[category] = {}
                      
                        for pipelineName, pipeline in self.pipelines[category].items():
                              if pipelineName in exclude_pipeline_names:
                                    print(f"Skipping pipeline {pipelineName} in category {category} because it is in the exclude list")
                                    continue

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

                  concurrent.futures.wait(futures)
                  for future in futures:
                        if future.exception():
                              raise future.exception()

            return results

      # 2) Final model functions
      def select_best_performing_model(self):
            """
            Selects the best performing model based on the classification report

            Parameters
            ----------
            metric: str
                  The metric to use to select the best performing model.

            Returns
            -------
            best_model_name: str
                  The name of the best performing model.
            best_score: float
                  The score of the best performing model.
            """
            metric = self.variables["phase_runners"]["dataset_runner"]["metrics_to_evaluate"]["preferred_metric"]
            assert metric in self.pipelines_analysis.merged_report_per_phase[self.pipeline_state].columns, f"Metric not found. Columns are: {self.pipelines_analysis.merged_report_per_phase[self.pipeline_state].columns}"
            metric_df = self.pipelines_analysis.merged_report_per_phase[self.pipeline_state][metric]
            model_names = metric_df.loc["modelName"].tolist()  # Last row: model names
            metric_df = metric_df.drop(index='modelName')     # Drop last row
            metric_df.columns = model_names            # Rename columns to model names

            weighted_avg = metric_df.loc['weighted avg']
            filtered = weighted_avg[~weighted_avg.index.str.endswith('_train')] # Remove training models
            best_model_name = filtered.idxmax()
            best_score = filtered.max()
            self.best_performing_model = {
                  "pipelineName": None,
                  "modelName": best_model_name,
            }
            self.pipelines_analysis.best_performing_model = self.best_performing_model
            print(f"Best performing model: {best_model_name} with {metric} {best_score:.4f}")

            # Overwrite the sklearn_model for the post state 
            for pipeline in self.pipelines["not_baseline"]:
                  for modelName in self.pipelines["not_baseline"][pipeline].modelling.list_of_models:
                        if modelName == best_model_name:
                              self.pipelines["not_baseline"][pipeline].modelling.list_of_models[modelName].tuning_states["post"].model_sklearn = self.pipelines["not_baseline"][pipeline].modelling.list_of_models[modelName].tuning_states["in"].assesment["model_sklearn"]
                              self.best_performing_model["pipelineName"] = pipeline

            return best_model_name, best_score
      

      def fit_final_models(self):
            """
            Fits the final models (post-tuning).

            Returns
            -------
            None
            """
            # Best not_baseline model
            self.pipelines["not_baseline"][self.best_performing_model["pipelineName"]].modelling.fit_models(
                  current_phase="post", 
                  best_model_name=self.best_performing_model["modelName"],
                  baseline_model_name=None
            )

            # All baseline models ()
            tasks = []
            for pipeline in self.pipelines["baseline"]:
                  for modelName in self.pipelines["baseline"][pipeline].modelling.list_of_models:
                        if modelName not in self.pipelines["baseline"][pipeline].modelling.models_to_exclude:
                              tasks.append((
                                    self.pipelines["baseline"][pipeline].modelling.fit_models,
                                    modelName
                              ))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                  futures = [
                        executor.submit(
                              fit_func,
                              current_phase="post",
                              best_model_name=None,
                              baseline_model_name=modelName
                        )
                        for fit_func, modelName in tasks
                  ]
                  for future in futures:
                        if future.exception():
                              raise future.exception()
      
      def evaluate_store_final_models(self):
            """
            Evaluates and stores the final models (post-tuning).

            Returns
            -------
            None
            """
            # Best not_baseline model
            self.pipelines["not_baseline"][self.best_performing_model["pipelineName"]].modelling.evaluate_and_store_models(
                  current_phase="post", 
                  comments=None,
                  best_model_name=self.best_performing_model["modelName"], 
                  baseline_model_name=None)
            
            # All baseline models
            tasks = []
            for pipeline in self.pipelines["baseline"]:
                  for modelName in self.pipelines["baseline"][pipeline].modelling.list_of_models:
                        if modelName not in self.pipelines["baseline"][pipeline].modelling.models_to_exclude:
                              tasks.append((
                                    self.pipelines["baseline"][pipeline].modelling.evaluate_and_store_models,
                                    modelName
                              ))
            with concurrent.futures.ThreadPoolExecutor() as executor:
                  futures = [
                        executor.submit(
                              evaluate_func,
                              current_phase="post",
                              comments=None,
                              best_model_name=None,
                              baseline_model_name=modelName
                        )
                        for evaluate_func, modelName in tasks
                  ]
                  for future in futures:
                        if future.exception():
                              raise future.exception()        
      
      # 3) Serialization and deserialization
      def serialize_pipelines(self, pipelines_to_serialize: list[str]) -> None:
            """
            Serializes the pipelines.

            Parameters
            ----------
            pipelines_to_serialize: list[str]
                  The pipelines to serialize.
            """
            self.serializer.serialize_pipelines(self.pipelines, pipelines_to_serialize)
      
      def serialize_models(self, models_to_serialize: list[str]) -> None:
            """
            Out of all the models in all the pipelines, we select the ones we want to serialize only.

            Parameters
            ----------
            models_to_serialize: list[str]
                  The models to serialize.
            """
            if self.all_models is None:
                  self.all_models = {}
                  for category in self.pipelines:
                        for pipeline_name in self.pipelines[category]:
                              for model_name in self.pipelines[category][pipeline_name].modelling.list_of_models:
                                    self.all_models[model_name] = self.pipelines[category][pipeline_name].modelling.list_of_models[model_name]
            self.serializer.serialize_models(self.all_models, models_to_serialize)
      
      def deserialize_pipelines(self, pipelines_to_deserialize: dict[str, str]) -> None:
            """
            Deserializes the pipelines.

            Parameters
            ----------
            pipelines_to_deserialize: dict[str, str]
                  The pipelines to deserialize.
            """
            return self.deserializer.deserialize_pipelines(pipelines_to_deserialize)
      
      def deserialize_models(self, models_to_deserialize: dict[str, str]):
            """
            Deserializes the models.

            Parameters
            ----------
            models_to_deserialize: dict[str, str]
                  The models to deserialize.
            """
            return self.deserializer.deserialize_models(models_to_deserialize)
            



