from library.pipeline.pipeline import Pipeline

from copy import deepcopy


class PipelineManager:
      """
      Trains all pipelines. 
      Evaluates all pipelines
      
      """
      def __init__(self, pipelines: dict[str, dict[str, Pipeline]]):
            self.pipelines = pipelines
      
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
      
      def all_pipelines_execute(self, methodName: str, verbose: bool = False, **kwargs,):
            """
            Executes a method for all pipelines.
            Method name can include dot notation for nested attributes (e.g. "model.fit")

            Note for verbose:
            - If u dont see a given pipeline in the results, it is because it has already been processed (its a copy of another pipeline)

            To be added:
            - Add multiconccurency to each invokation in the pipeline
            """
            results = {}
            processed_pipelines = set()

            for category in self.pipelines:
                  results[category] = {}
                  for pipelineName in self.pipelines[category]:
                        pipeline = self.pipelines[category][pipelineName]
                        if id(pipeline) not in processed_pipelines:
                              # Handle nested attribute access
                              obj = pipeline
                              for attr in methodName.split('.')[:-1]:
                                    obj = getattr(obj, attr) # Keeping only the last obj in memory (via the loop)
                              method = getattr(obj, methodName.split('.')[-1])
                              results[category][pipelineName] = method(**kwargs)
                              if verbose:
                                    print(f"Pipeline {pipelineName} in category {category} has executed {methodName}. Result is: {results[category][pipelineName]}")
                              processed_pipelines.add(id(pipeline))
            return results

