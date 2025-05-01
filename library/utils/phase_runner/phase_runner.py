from abc import ABC, abstractmethod


from library.pipeline.pipeline import Pipeline


""" Defiens the abstarct class for phase runners """


class PhaseRunner(ABC):
      def __init__(self, pipelines: dict[str, dict[str, Pipeline]], include_plots: bool = False, save_path: str = None) -> None:
            if include_plots:
                  assert save_path is not None, "save_path must be provided if include_plots is True"
            self.pipelines = pipelines
            self.include_plots = include_plots
            self.save_path = save_path

      @abstractmethod
      def run(self) -> None:
            pass