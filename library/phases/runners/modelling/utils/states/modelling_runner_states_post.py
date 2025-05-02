
from library.phases.runners.modelling.utils.states.modelling_runner_states_base import ModellingRunnerStates
from library.pipeline.pipeline_manager import PipelineManager

           
class PostTuningRunner(ModellingRunnerStates):
      def __init__(self, pipeline_manager: PipelineManager, save_plots: bool = False, save_path: str = None):
            super().__init__(pipeline_manager, save_plots, save_path)

      def run(self):
           print("Post tuning runner")
      #      self.pipeline_manager.all_pipelines_execute(methodName="modelling.fit_models",
      #                                  verbose=False, 
      #                                  exclude_pipeline_names=["stacking"],
      #                                  current_phase="post")