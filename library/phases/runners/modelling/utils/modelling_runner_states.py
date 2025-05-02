
from abc import ABC, abstractmethod

from library.pipeline.pipeline_manager import PipelineManager

class ModellingRunnerStates(ABC):
      def __init__(self, pipeline_manager: PipelineManager):
            self.pipeline_manager = pipeline_manager
      
      @abstractmethod
      def run(self):
            pass


class PreTuningRunner(ModellingRunnerStates):
      def __init__(self, pipeline_manager: PipelineManager):
            super().__init__(pipeline_manager)

      def run(self):
            print("Pre tuning runner about to start")
            self.pipeline_manager.all_pipelines_execute(methodName="modelling.fit_models",
                                       verbose=False, 
                                       exclude_pipeline_names=["stacking"], # debugging
                                       current_phase="pre")

class InTuningRunner(ModellingRunnerStates):
      def __init__(self, pipeline_manager: PipelineManager):
            super().__init__(pipeline_manager)

      def run(self):
            print("In tuning runner")
      #      self.pipeline_manager.all_pipelines_execute(methodName="modelling.fit_models",
      #                                  verbose=False, 
      #                                  exclude_pipeline_names=["stacking"],
      #                                  current_phase="in")
           
class PostTuningRunner(ModellingRunnerStates):
      def __init__(self, pipeline_manager: PipelineManager):
            super().__init__(pipeline_manager)

      def run(self):
           print("Post tuning runner")
      #      self.pipeline_manager.all_pipelines_execute(methodName="modelling.fit_models",
      #                                  verbose=False, 
      #                                  exclude_pipeline_names=["stacking"],
      #                                  current_phase="post")