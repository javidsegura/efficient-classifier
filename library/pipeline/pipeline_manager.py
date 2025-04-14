from library.pipeline.pipeline import Pipeline

from copy import deepcopy
import concurrent.futures
import threading
from typing import Any

from library.pipeline.analysis.pipelines_analysis import PipelinesAnalysis

class PipelineManager:
      """
      Trains all pipelines. 
      Evaluates all pipelines
      
      """
      def __init__(self, pipelines: dict[str, dict[str, Pipeline]]):
            self.pipelines = pipelines
            self.pipelines_analysis = PipelinesAnalysis(pipelines)
      
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
      
      def all_pipelines_execute(self, methodName: str, verbose: bool = False, **kwargs):
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
                  
                  # Submit tasks for each unique pipeline
                  for category in self.pipelines:
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

