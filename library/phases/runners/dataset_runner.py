

from library.utils.phase_runner.phase_runner import PhaseRunner
from library.pipeline.pipeline_manager import PipelineManager

class DatasetRunner(PhaseRunner):
      def __init__(self, pipeline_manager: PipelineManager, include_plots: bool = False, save_path: str = "") -> None:
            super().__init__(pipeline_manager, include_plots, save_path)
      
      def _run_feature_engineering_pre_split(self) -> None:
            # FEDE (expand here)
            # pre split
            ...

      def run(self) -> None:
            # Select the first pipeline.
            print(self.pipeline_manager.pipelines)
            pipelines = list(self.pipeline_manager.pipelines["not_baseline"].values())
            default_pipeline = pipelines[0]

            feature_engineering_results = self._run_feature_engineering_pre_split()

            split_df = default_pipeline.dataset.split.asses_split_classifier(
                        p=self.pipeline_manager.variables["dataset_runner"]["split_df"]["p"], 
                        step=self.pipeline_manager.variables["dataset_runner"]["split_df"]["step"],
                        save_plots=self.include_plots,
                        save_path=self.save_path
                        )

            encoding_df = default_pipeline.dataset.split.split_data(
                        y_column=self.pipeline_manager.variables["dataset_runner"]["encoding"]["y_column"],
                        train_size=self.pipeline_manager.variables["dataset_runner"]["encoding"]["train_size"],
                        validation_size=self.pipeline_manager.variables["dataset_runner"]["encoding"]["validation_size"],
                        test_size=self.pipeline_manager.variables["dataset_runner"]["encoding"]["test_size"],
                        save_plots=False, # remove this harcdoded value (currently set to make pipeline run faster)
                        save_path=self.save_path
                  )
            return split_df, encoding_df

