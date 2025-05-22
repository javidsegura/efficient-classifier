from efficient_classifier.pipeline.pipeline_runner import PipelineRunner
import matplotlib
import yaml

variables = yaml.load(open("efficient-classifier/efficient_classifier/configurations.yaml"), Loader=yaml.FullLoader)
include_plots = variables["PIPELINE_RUNNER"]["include_plots"]

if include_plots:
      matplotlib.use("Agg")
import matplotlib.pyplot as plt



def run_pipeline():
       # Setting up the pipeline runner
       pipeline_runner = PipelineRunner(
             dataset_path=variables["PIPELINE_RUNNER"]["dataset_path"],
            model_task=variables["PIPELINE_RUNNER"]["model_task"],
            include_plots=variables["PIPELINE_RUNNER"]["include_plots"],
            pipelines_names=variables["PIPELINE_RUNNER"]["pipelines_names"],
            serialize_results=variables["PIPELINE_RUNNER"]["serialize_results"],
            variables=variables
       )

       # Running the pipeline
       pipeline_runner.run()

if __name__ == "__main__":
      run_pipeline()


