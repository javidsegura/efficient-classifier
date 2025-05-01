


from library.pipeline.pipeline_runner import PipelineRunner



pipeline_runner = PipelineRunner(
      dataset_path="./dataset/dynamic_dataset.csv",
      results_path="results/model_evaluation/results.csv",
      model_task="classification",
      include_plots=True,
      pipelines_names={
            "not-baseline": ["enembled", "tree-based", "support-vector-machine",
                             "naive-bayes", "feed-forward-neural-network", "stacking"],
            "baseline": ["baselines"],
      }
)

pipeline_runner.run()



