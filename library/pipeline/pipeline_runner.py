""""

This file runs the pipeline code. Its the full automation of all the pipelines' code

"""

from library.pipeline.pipeline import Pipeline
from library.phases.runners.dataset_runner import DatasetRunner
from library.utils.decorators.timer import timer

import logging
import time
import os

""" Phases are: 
- Splitting


"""

class PipelineRunner:
      def __init__(self, 
                   dataset_path: str, 
                   results_path: str,
                   model_task: str,
                   pipelines_names: dict[str, list[str]],
                   include_plots: bool = True
                   ) -> None:
            self.dataset_path = dataset_path
            if not os.path.exists(results_path):
                  os.makedirs(results_path, exist_ok=True)
            self.results_path = results_path
            self.model_task = model_task
            self._set_up_pipelines(pipelines_names)
            self._set_up_logger()

            # Phase runners 
            os.makedirs("results/plots/", exist_ok=True)
            self.phase_runners = {
                  "dataset": DatasetRunner(self.pipelines,
                                            include_plots=include_plots,
                                            save_path="results/plots/dataset/")
            }

      
      def _set_up_pipelines(self, pipelines_names: dict[str, list[str]]) -> None:
            print(f"Setting up pipelines for {self.model_task} model task")
            self.pipelines = {}
            default_pipeline = Pipeline(self.dataset_path, self.results_path, self.model_task)
            default_pipeline.dataset.df.drop(columns=["Family", "Hash"], inplace=True) # We have decided to use only category as target variable; Hash is temporary while im debugging (it will be deleted in EDA)
            for category_name, pipelines in pipelines_names.items():
                  self.pipelines[category_name] = {}
                  for pipeline_name in pipelines:
                        self.pipelines[category_name][pipeline_name] = default_pipeline
      
      def _set_up_logger(self) -> None:
            log_file = "results/logs/pipeline_runner.log"
            logger = logging.getLogger("my_logger")
            logger.setLevel(logging.INFO)

            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            logger.info(f"Pipeline runner started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger = logger

      def run(self):
            for phase_name, phase_runner in self.phase_runners.items():
                  @timer(phase_name)
                  def run_phase():
                        phase_result = phase_runner.run()
                        self.logger.info(f"Phase {phase_name} completed in {time.strftime('%Y-%m-%d %H:%M:%S')}")
                        if phase_result is not None:
                              self.logger.info(f"Return result:s {phase_result}")
                  run_phase()
