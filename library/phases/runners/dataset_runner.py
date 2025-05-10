import numpy as np

from library.utils.phase_runner_definition.phase_runner import PhaseRunner
from library.pipeline.pipeline_manager import PipelineManager

class DatasetRunner(PhaseRunner):
      def __init__(self, pipeline_manager: PipelineManager, include_plots: bool = False, save_path: str = "") -> None:
            super().__init__(pipeline_manager, include_plots, save_path)

            

            
      def _ensembled_pipeline_feature_engineering(self) -> None:
            # FEDE (expand here)
            ensembled_pipeline = self.pipeline_manager.pipelines["not_baseline"]["ensembled"]
            df = ensembled_pipeline.dataset.df

            #df["Name_of_some_column"] = np.log(df["Name_of_some_column"])
      
      def _run_feature_engineering_pre_split(self) -> None:
            """

            TO READ BY FEDE:
                  - Each pipelin has its own function (just to make it easier to read)
                  - Above you have an example of applying log transformation for a specific column
                  - Note that currettly all pipelines are the same because divergance is done later in modelling (and juan probably will put it up to data preprocessing). If u want
                  to do somethign here that is pipeline specific you need to create the divergence (below you have a function that is all-commented out on showing how to diverge
                  all pipelines, as an example)
            
            
            """
            # FEDE (expand here)
            # pre split
            self._ensembled_pipeline_feature_engineering()
      


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
                        save_plots=True, 
                        save_path=self.save_path
                  )
            return split_df, encoding_df

