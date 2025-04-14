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

from library.phases.model_selection.model.model import Model
from library.phases.dataset.dataset import Dataset
class ResultsDF:
      def __init__(self, results_path: str, dataset: Dataset):
            if dataset.modelTask == "classification":
                  metrics_to_evaluate = ["accuracy", "precision", "recall", "f1-score"]
            else:
                  metrics_to_evaluate = ["r2", "mae", "mse"]
            assert len(metrics_to_evaluate) > 0, "The metrics to evaluate must be a non-empty list"
            self.metrics_to_evaluate = metrics_to_evaluate
            self.results_path = results_path
            self.dataset = dataset
            header = ["id", "timeStamp", "comments", "modelName", "currentPhase", "features_used", "hyperParameters", "timeToFit", "timeToPredict"]
            if dataset.modelTask == "classification":
                  header += ["classification_report"]
            header += [f"{metric}_val" for metric in self.metrics_to_evaluate]
            header += [f"{metric}_test" for metric in self.metrics_to_evaluate]
            self.header = header
            columns_to_check_duplicates = ["modelName", "features_used", "hyperParameters", "comments", "status"]
            self.columns_to_check_duplicates = columns_to_check_duplicates
            self._create_results_file()
            self.results_df = pd.read_csv(self.results_path)
      
      def _create_results_file(self):
            if os.path.exists(self.results_path):
                  return
            else: 
                  os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
                  with open(self.results_path, "w", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(self.header)
      
      def check_header_consitency(self, metadata: dict):
            print(f"Metadata is: {metadata}")
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
      
      def store_results(self, list_of_models: dict[str, Model], current_phase: str, comments: str, models_to_exclude: list[str] = None):
            start_time = time.time()
            print(f"[DEBUG] Starting store_results")
            sys.stdout.flush()
            assert current_phase and comments, "Either current_phase and comments must be provided"
            model_logs = []
            model_count = 0
            for modelName, modelObject in list_of_models.items():
                  if models_to_exclude is not None and modelName in models_to_exclude:
                        continue
                  
                  model_count += 1
                  if model_count % 5 == 0:  # Log every 5 models
                      print(f"[DEBUG {time.time() - start_time:.2f}s] Processing model {model_count}: {modelName}")
                      sys.stdout.flush()
                  
                  # Extracting the metadata from the assesment
                  using_validation_set = current_phase == "pre" or current_phase == "in"
                  metadata = modelObject.tuning_states[current_phase].assesment
                  
                  self.check_header_consitency(metadata)
            
                  # Extracting the model_log
                  model_sklearn = metadata["model_sklearn"]      
                  model_log = {
                        "id": "",
                        "timeStamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "comments": comments,
                        "modelName": modelName,
                        "currentPhase": current_phase,
                        "features_used": self.dataset.X_train.columns.tolist(),
                        "hyperParameters": self.extract_model_hyperparameters(model_sklearn),
                        "timeToFit": metadata["timeToFit"],
                        "timeToPredict": metadata["timeToPredict"],
                  }
                  if self.dataset.modelTask == "classification":
                        model_log["classification_report"] = metadata["classification_report"]
                  # Adding remaining data 
                  metricsAdded = 0
                  for col in list(metadata.keys()):
                        print(f"Col is: {col}")
                        if col in self.metrics_to_evaluate:
                              if using_validation_set:
                                    model_log[f"{col}_val"] = metadata[col]
                                    model_log[f"{col}_test"] = -1
                              else:
                                    model_log[f"{col}_test"] = metadata[col]
                                    model_log[f"{col}_val"] = -1
                              metricsAdded += 1
                  assert metricsAdded == len(self.metrics_to_evaluate), f"Not all metrics were added. Model_log is: {model_log}"

                  # Computing hash values 
                  dataForHash = {k: v for k, v in model_log.items() if k in self.columns_to_check_duplicates}
                  hash_value = hashlib.sha256(json.dumps(dataForHash).encode()).hexdigest()
                  isNewModel = hash_value not in self.results_df["id"].values
                  model_log["id"] = hash_value
                  if isNewModel:
                        with open(self.results_path, "a", newline='') as f: 
                              writer = csv.writer(f)
                              writer.writerow([str(model_log[col]) for col in self.header])
                  else:
                        print(f"****WARNING****: A model with the same values already exists in the results. Results will not be saved. \n \
                              You tried to write {model_log}")
                  model_logs.append(model_log)
            print(f"[DEBUG {time.time() - start_time:.2f}s] Completed store_results, processed {model_count} models")
            sys.stdout.flush()


            return model_logs
      


      def extract_model_hyperparameters(self, modelObject: Model):
            return "NA"
      
      def plot_results_over_time(self, metric: str):
            """
            Plots the results over time for all models

            Parameters
            ----------
                  metric : str
                  The metric to plot
            """
            assert metric in self.metrics_to_evaluate, f"Metric {metric} not found in the metrics stored. \n Available metrics: {self.metrics_to_evaluate}"
            results_df = pd.read_csv(self.results_path)
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

                        
                 
                  
                  
                  
