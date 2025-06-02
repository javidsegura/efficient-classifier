from efficient_classifier.pipeline.pipeline_runner import PipelineRunner
import matplotlib
import yaml






def run_pipeline(variables):
       include_plots = variables["general"]["include_plots"]
       # Setting up the pipeline runner
       if include_plots:
                  matplotlib.use("Agg")
       import matplotlib.pyplot as plt
       
       pipeline_runner = PipelineRunner(
            dataset_path=variables["general"]["dataset_path"],
            model_task=variables["general"]["model_task"],
            include_plots=variables["general"]["include_plots"],
            pipelines_names=variables["general"]["pipelines_names"],
            variables=variables
       )

       # Running the pipeline
       pipeline_runner.run()




