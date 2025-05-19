from abc import ABC, abstractmethod

import os

from efficient_classifier.pipeline.pipeline_manager import PipelineManager


""" Defiens the abstarct class for phase runners """


class PhaseRunner(ABC):
      """
      In the extensions of this class u should
      write the code that you want to run in the notebook.
      
      """
      def __init__(self, pipeline_manager: PipelineManager, include_plots: bool = False, save_path: str = "") -> None:
            if include_plots:
                  assert save_path is not None, "save_path must be provided if include_plots is True"
            self.pipeline_manager = pipeline_manager
            self.include_plots = include_plots
            if not os.path.exists(save_path):
                  os.makedirs(save_path, exist_ok=True)
            self.save_path = save_path

      @abstractmethod
      def run(self) -> None:
            """
            Return something if you want to save it to the logs
            """
            pass