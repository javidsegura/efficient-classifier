

from library.utils.phase_runner.phase_runner import PhaseRunner
from library.pipeline.pipeline_manager import PipelineManager

class DatasetRunner(PhaseRunner):
      def __init__(self, pipeline_manager: PipelineManager, include_plots: bool = False, save_path: str = "") -> None:
            super().__init__(pipeline_manager, include_plots, save_path)
      
      def _run_feature_engineering(self) -> None:
            # FEDE (expand here)
            # pre split
            ...

      def run(self) -> None:
            # Select the first pipeline.
            print(self.pipeline_manager.pipelines)
            pipelines = list(self.pipeline_manager.pipelines["not_baseline"].values())
            default_pipeline = pipelines[0]

            feature_engineering_results = self._run_feature_engineering()

            split_df = default_pipeline.dataset.split.asses_split_classifier(
                        p=.85, 
                        step=.05,
                        save_plots=self.include_plots,
                        save_path=self.save_path
                        )

            encoding_df = default_pipeline.dataset.split.split_data(
                        y_column="Category",
                        train_size=0.8,
                        validation_size=0.1,
                        test_size=0.1,
                        save_plots=False, # remove this harcdoded value (makes pipeline run faster)
                        save_path=self.save_path
                  )
            return split_df, encoding_df

