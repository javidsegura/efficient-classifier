
from abc import ABC, abstractmethod

from library.pipeline.pipeline_manager import PipelineManager

class ModellingRunnerStates(ABC):
      def __init__(self, pipeline_manager: PipelineManager, save_plots: bool = False, save_path: str = None):
            self.pipeline_manager = pipeline_manager
            self.save_plots = save_plots
            self.save_path = save_path
      
      @abstractmethod
      def run(self):
            pass

      

