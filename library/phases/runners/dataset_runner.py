

from library.utils.phase_runner.phase_runner import PhaseRunner
from library.pipeline.pipeline import Pipeline

class DatasetRunner(PhaseRunner):
      def __init__(self, pipelines: dict[str, dict[str, Pipeline]], include_plots: bool = False, save_path: str = "results/dataset_split") -> None:
            super().__init__(pipelines, include_plots, save_path)

      def run(self) -> None:
            # Select the first pipeline
            pipelines = list(self.pipelines["not-baseline"].values())
            default_pipeline = pipelines[0]

            split_df = default_pipeline.dataset.split.asses_split_classifier(
                        p=.85, 
                        step=.05,
                        save_plots=self.include_plots,
                        save_path=self.save_path
                        )

            default_pipeline.dataset.split.split_data(
                        y_column="Category",
                        train_size=0.8,
                        validation_size=0.1,
                        test_size=0.1,
                        save_plots=False, # remove this harcdoded value
                        save_path=self.save_path
                  )
            return split_df

