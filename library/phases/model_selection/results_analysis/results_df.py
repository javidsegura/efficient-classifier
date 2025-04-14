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
      def __init__(self, results_path: str, metrics_to_evaluate: list[str], dataset: Dataset):
            assert len(metrics_to_evaluate) > 0, "The metrics to evaluate must be a non-empty list"
            self.results_path = results_path
            self.metrics_to_evaluate = metrics_to_evaluate
            self.dataset = dataset
            self.phases = ["EDA", "DataPreprocessing", "FeatureAnalysis", "HyperParameterOptimization"]
            header = ["id", "timeStamp", "comments", "modelName", "status", "features_used", "hyperParameters", "timeToFit", "timeToMakePredictions"]
            header += [f"is_{phase}_done" for phase in self.phases]
            header += [f"{metric}_val" for metric in self.metrics_to_evaluate]
            header += [f"{metric}_test" for metric in self.metrics_to_evaluate]
            self.header = header
            columns_to_check_duplicates = ["modelName", "features_used", "hyperParameters", "comments", "status"]
            columns_to_check_duplicates += [f"is_{phase}_done" for phase in self.phases]
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
      
      def store_results(self, list_of_models: dict[str, Model], phaseProcess: dict, comments: str, models_to_include: list[str] = None):
            start_time = time.time()
            print(f"[DEBUG] Starting store_results")
            sys.stdout.flush()
            
            model_logs = []
            model_count = 0
            for modelName, modelObject in list_of_models.items():
                  if models_to_include is not None and modelName not in models_to_include:
                        continue
                  
                  model_count += 1
                  if model_count % 5 == 0:  # Log every 5 models
                      print(f"[DEBUG {time.time() - start_time:.2f}s] Processing model {model_count}: {modelName}")
                      sys.stdout.flush()
                  
                  using_validation_set = modelObject.currentPhase == "pre_tuning" or modelObject.currentPhase == "in_tuning"
                  metadata = None
                  if modelObject.currentPhase == "pre_tuning":
                        metadata = modelObject.preTuningState.assesment
                  elif modelObject.currentPhase == "in_tuning":
                        metadata = modelObject.inTuningState.assesment
                  elif modelObject.currentPhase == "post_tuning":
                        metadata = modelObject.postTuningState.assesment
                  else:
                        raise ValueError("Invalid phase")
                  
                  print(f"Metadata: {metadata}")
                  print(f"Current phase: {modelObject.currentPhase}")
                  # Checking same header between model_log and results_df
                  model_log_header = list(metadata.keys())
                  if "model_sklearn" in model_log_header:
                        model_log_header.remove("model_sklearn")
                  if "predictions" in model_log_header:
                        model_log_header.remove("predictions")
                  model_log_cleaned = []
                  for metric in model_log_header:
                        if metric in self.metrics_to_evaluate:
                              model_log_cleaned.append(metric + "_val")
                              model_log_cleaned.append(metric + "_test")
                        else:
                              model_log_cleaned.append(metric)
                  if sorted(model_log_cleaned) != sorted(self.header):
                        raise ValueError(f"The data to write does not match the columns of the results. \n Data to write: {sorted(model_log_cleaned)} \n Data header: {sorted(self.header)}")
                  
                  # Extracting the model_log
                  model_sklearn = modelObject.preTuningState.assesment["model_sklearn"] if using_validation_set else modelObject.postTuningState.assesment["model_sklearn"]      
                  model_log = {
                        "id": "",
                        "timeStamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "comments": comments,
                        "modelName": modelName,
                        "status": metadata["status"],
                        "features_used": self.dataset.X_train.columns.tolist(),
                        "hyperParameters": self.extract_model_hyperparameters(model_sklearn),
                        "timeToFit": metadata["timeToFit"],
                        "timeToMakePredictions": metadata["timeToMakePredictions"],
                  }
                  for phase in self.phases:
                        model_log[f"is_{phase}_done"] = "Yes" if phaseProcess[f"is_{phase}_done"] else "No"
                  for metric in metadata.keys():
                        if metric in self.metrics_to_evaluate:
                              if using_validation_set:
                                    model_log[f"{metric}_val"] = metadata[metric]
                                    model_log[f"{metric}_test"] = -1
                              else:
                                    model_log[f"{metric}_test"] = metadata[metric]
                                    model_log[f"{metric}_val"] = -1

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

                        
                 
                  
                  
                  
