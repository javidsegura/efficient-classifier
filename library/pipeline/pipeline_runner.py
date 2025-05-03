""""

This file runs the pipeline code. Its the full automation of all the pipelines' code

"""

import logging
import time
import os


from library.pipeline.pipeline import Pipeline
from library.pipeline.pipeline_manager import PipelineManager

# Runners
from library.phases.runners.dataset_runner import DatasetRunner
from library.phases.runners.featureAnalysis_runner import FeatureAnalysisRunner
from library.phases.runners.dataPreprocessing_runner import DataPreprocessingRunner
from library.phases.runners.modelling.modelling_runner import ModellingRunner

# Utils
from library.utils.decorators.timer import timer
from library.utils.slackBot.bot import SlackBot

""" Phases are: 
- Splitting

"""

class PipelineRunner:
      def __init__(self, 
                   dataset_path: str, 
                   model_task: str,
                   pipelines_names: dict[str, list[str]],
                   include_plots: bool = True
                   ) -> None:
            """
            This is some gerat class
            """
            self.dataset_path = dataset_path
            self.model_task = model_task
            self._set_up_folders()
            self._set_up_pipelines(pipelines_names)
            self._set_up_logger()

            self.phase_runners = {
                  "dataset": DatasetRunner(self.pipeline_manager,
                                            include_plots=include_plots,
                                            save_path=self.plots_path + "dataset/"),
                  "data_preprocessing": DataPreprocessingRunner(self.pipeline_manager,
                                                            include_plots=include_plots,
                                                            save_path=self.plots_path + "data_preprocessing/"),
                  "feature_analysis": FeatureAnalysisRunner(self.pipeline_manager,
                                                            include_plots=include_plots,
                                                            save_path=self.plots_path + "feature_analysis/"),
                  "modelling": ModellingRunner(self.pipeline_manager,
                                                include_plots=include_plots,
                                                save_path=self.plots_path + "modelling/")
                  
            }
            self.slack_bot = SlackBot()

      
      def _set_up_pipelines(self, pipelines_names: dict[str, list[str]]) -> None:
            print(f"Setting up pipelines for {self.model_task} model task")
            combined_pipelines = {}
            default_pipeline = Pipeline(self.dataset_path, self.model_results_path, self.model_task)
            # DO GENERAL PIPELINE-WIDE SET-UP (e.g: remove zero day, no category, etc)
            default_pipeline.dataset.df.drop(columns=["Family", "Hash"], inplace=True) # We have decided to use only category as target variable; Hash is temporary while im debugging (it will be deleted in EDA)
            for category_name, pipelines in pipelines_names.items():
                  combined_pipelines[category_name] = {}
                  for pipeline_name in pipelines:
                        combined_pipelines[category_name][pipeline_name] = default_pipeline
            self.pipeline_manager = PipelineManager(combined_pipelines)

      
      def _set_up_logger(self) -> None:
            log_file = self.logs_path + "pipeline_runner.log"
            logger = logging.getLogger("my_logger")
            logger.setLevel(logging.INFO)

            file_handler = logging.FileHandler(log_file, mode="w")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            logger.info(f"Pipeline runner started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger = logger

      def _set_up_folders(self) -> None:
            if not os.path.exists("results/model_evaluation/"):
                  os.makedirs("results/model_evaluation/", exist_ok=True)
            self.model_results_path = "results/model_evaluation/results.csv"

            if not os.path.exists("results/plots/"):
                  os.makedirs("results/plots/", exist_ok=True)
            self.plots_path = "results/plots/"

            if not os.path.exists("results/logs/"):
                  os.makedirs("results/logs/", exist_ok=True)
            self.logs_path = "results/logs/"

      def run(self):
            for phase_name, phase_runner in self.phase_runners.items():
                  @timer(phase_name)
                  def run_phase():
                        start_time = time.time()
                        phase_result = phase_runner.run()
                        self.logger.info(f"Phase '{phase_name}' completed in {time.time() - start_time} seconds at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                        if phase_result is not None:
                              self.logger.info(f"'{phase_name}' returned: {phase_result}")
                              time.sleep(.1)
                              self.slack_bot.send_message(f"Phase '{phase_name}' completed in {time.time() - start_time} seconds at {time.strftime('%Y-%m-%d %H:%M:%S')}\
                                                          Result: {str(phase_result)}",
                                                          channel="#general")

                  run_phase()
            # Send slack bot all the images in the results/plots folder
            for root, dirs, files in os.walk(self.plots_path):
                  for file in files:
                        file_path = os.path.join(root, file)
                        self.slack_bot.send_file(file_path,
                                                 channel="#general",
                                                 title=file,
                                                 initial_comment="")
            


