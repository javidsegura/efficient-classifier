
from abc import ABC, abstractmethod

from efficient_classifier.pipeline.pipeline_manager import PipelineManager

class ModellingRunnerStates(ABC):
      """
      This is the base class for the modelling runner states. It contains the common methods for all the states. In this case, it forces the run method to be implemented and to have the same 
      initialization parameters.
      """
      def __init__(self, pipeline_manager: PipelineManager, save_plots: bool = False, save_path: str = None):
            self.pipeline_manager = pipeline_manager
            self.save_plots = save_plots
            self.save_path = save_path
      
      @abstractmethod
      def run(self):
            pass

      

