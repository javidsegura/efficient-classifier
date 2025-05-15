import os
import csv
from datetime import datetime

import pandas as pd
import hashlib
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import time, sys

from library.phases.phases_implementation.modelling.shallow.model_definition.model_base import Model
from library.phases.phases_implementation.dataset.dataset import Dataset

import yaml


class ResultsDF:
      def __init__(self, model_results_path: str, dataset: Dataset):
            self.variables = yaml.load(open("library/configurations.yaml"), Loader=yaml.FullLoader)

            if dataset.modelTask == "classification":
                  metrics_to_evaluate = self.variables["dataset_runner"]["metrics_to_evaluate"]["classification"]
            else:
                  metrics_to_evaluate = self.variables["dataset_runner"]["metrics_to_evaluate"]["regression"]
            assert len(metrics_to_evaluate) > 0, "The metrics to evaluate must be a non-empty list"
            self.metrics_to_evaluate = metrics_to_evaluate
            self.model_results_path = model_results_path
            self.dataset = dataset
            header = ["id", "timeStamp", "comments", "modelName", "currentPhase", "features_used", "hyperParameters", "timeToFit", "timeToPredict"]

            header += [f"{metric}_val" for metric in self.metrics_to_evaluate]
            header += [f"{metric}_test" for metric in self.metrics_to_evaluate]
            self.header = header
            columns_to_check_duplicates = ["modelName", "features_used", "hyperParameters", "comments", "currentPhase"]
            self.columns_to_check_duplicates = columns_to_check_duplicates
            self._create_results_file()
            self.results_df = pd.read_csv(self.model_results_path)
      
      def _create_results_file(self):
            if os.path.exists(self.model_results_path):
                  return
            else: 
                  os.makedirs(os.path.dirname(self.model_results_path), exist_ok=True)
                  with open(self.model_results_path, "w", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(self.header)
      
      def check_header_consistency(self, metadata: dict):
            metadata = metadata.copy() # Temporary copy for the check
            header_cols = set(self.header)
            metadata_cols = set(metadata.keys())
            for metric in self.metrics_to_evaluate:
                  metadata_cols.add(metric + "_val")
                  metadata_cols.add(metric + "_test")
                  metadata.pop(metric)
            for col in header_cols:
                  if col not in metadata_cols:
                        raise ValueError(f"The data to write does not match the columns of the results. \n Data to write: {sorted(list(metadata_cols))} \n Data header: {sorted(header_cols)}")
      
      def store_results(self, list_of_models: dict[str, Model], current_phase: str, models_to_exclude: list[str] = None):
            model_logs = []
            for modelName, modelObject in list_of_models.items():
                  if models_to_exclude is not None and modelName in models_to_exclude:
                        continue
         
                  # Extracting the metadata from the assesment
                  using_validation_set = current_phase == "pre" or current_phase == "in"
                  metadata = modelObject.tuning_states[current_phase].assesment

                  self.check_header_consistency(metadata)

                  # Extracting the model_log
                  model_sklearn = metadata["model_sklearn"]      
                  model_log = {
                        "id": "",
                        "timeStamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "comments": self.variables["modelling_runner"]["model_assesment"]["comments"],
                        "modelName": modelName,
                        "currentPhase": current_phase,
                        "features_used": self.dataset.X_train.columns.tolist(),
                        "hyperParameters": self.serialize_params(model_sklearn.get_params()),
                        "timeToFit": metadata["timeToFit"],
                        "timeToPredict": metadata["timeToPredict"],
                  }
                  
                  # Adding remaining data 
                  print(f"METADATA IS: {metadata}")
                  metricsAdded = 0
                  for col in list(metadata.keys()):
                        if col in self.metrics_to_evaluate:
                              print(f"COL IS: {col}")
                              if using_validation_set:
                                    model_log[f"{col}_val"] = metadata["metrics"]["base_metrics"][col] if col in metadata["metrics"]["base_metrics"] else metadata["metrics"]["additional_metrics"]["not_train"][col]
                                    model_log[f"{col}_test"] = -1
                              else:
                                    model_log[f"{col}_test"] = metadata["metrics"]["base_metrics"][col] if col in metadata["metrics"]["base_metrics"] else metadata["metrics"]["additional_metrics"]["train"][col + "_train"]
                                    model_log[f"{col}_val"] = -1
                              metricsAdded += 1
                  assert metricsAdded == len(self.metrics_to_evaluate), f"Not all metrics were added. Model_log is: {model_log}"

                  # Computing hash values 
                  dataForHash = {k: v for k, v in model_log.items() if k in self.columns_to_check_duplicates}
                  hash_value = hashlib.sha256(json.dumps(dataForHash).encode()).hexdigest()
                  isNewModel = hash_value not in self.results_df["id"].values
                  model_log["id"] = hash_value
                  if isNewModel:
                        with open(self.model_results_path, "a", newline='') as f: 
                              writer = csv.writer(f)
                              writer.writerow([str(model_log[col]) for col in self.header])
                  else:
                        print(f"****WARNING****: A model with the same values already exists in the results. Results will not be saved. \n \
                              You tried to write {model_log}")
                  model_logs.append(model_log)
            sys.stdout.flush()
            return model_logs
      
      def serialize_params(self, params):
            def make_serializable(val):
                  if hasattr(val, '__class__'):
                        return str(val)
                  if isinstance(val, dict):
                        return {k: make_serializable(v) for k, v in val.items()}
                  if isinstance(val, (list, tuple)):
                        return [make_serializable(v) for v in val]
                  return val

            return make_serializable(params)


      def plot_results_over_time(self, metric: str):
            """
            Plots the results over time for all models

            Parameters
            ----------
                  metric : str
                  The metric to plot
            """
            assert metric in self.metrics_to_evaluate, f"Metric {metric} not found in the metrics stored. \n Available metrics: {self.metrics_to_evaluate}"
            results_df = pd.read_csv(self.model_results_path)
            results_df = results_df.sort_values(by="timeStamp")
            assert (results_df[metric].shape[0] > 0), f"No results found for metric {metric}"
            
            plt.figure(figsize=(30, 10))
            sns.lineplot(x="timeStamp", y=metric, hue="modelName", data=results_df)
            plt.title(f"Results over time for {metric}")
            plt.xlabel("Time")
            ax = plt.gca()

            # Set locator for ticks (more dense)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())

            plt.xticks(rotation=45, fontsize=8)  # smaller font
            plt.ylabel(metric)
            plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            return results_df

                        
                 
                  
                  
                  
