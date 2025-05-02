
from library.pipeline.pipeline_runner import PipelineRunner
import matplotlib

include_plots = True
if include_plots:
      matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_pipeline():
      # Setting up the pipeline runner
      pipeline_runner = PipelineRunner(
            dataset_path="./dataset/dynamic_dataset.csv",
            model_task="classification",
            include_plots=include_plots,
            pipelines_names={
                  "not_baseline": ["ensembled", "tree_based", "support_vector_machine",
                              "naive_bayes", "feed_forward_neural_network", "stacking"],
                  "baseline": ["baselines"],
            }
      )

      # Running the pipeline
      pipeline_runner.run()

if __name__ == "__main__":
      run_pipeline()

