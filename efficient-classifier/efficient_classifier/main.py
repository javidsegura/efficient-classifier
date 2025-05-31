from efficient_classifier.pipeline.pipeline_runner import PipelineRunner
import matplotlib
import yaml

variables = yaml.load(open("efficient-classifier/efficient_classifier/configurations.yaml"), Loader=yaml.FullLoader)
include_plots = variables["general"]["include_plots"]

if include_plots:
      matplotlib.use("Agg")
import matplotlib.pyplot as plt



def run_pipeline():
       # Setting up the pipeline runner
       pipeline_runner = PipelineRunner(
             dataset_path=variables["general"]["dataset_path"],
            model_task=variables["general"]["model_task"],
            include_plots=variables["general"]["include_plots"],
            pipelines_names=variables["general"]["pipelines_names"],
            variables=variables
       )

       # Running the pipeline
       pipeline_runner.run()

if __name__ == "__main__":
      run_pipeline()


