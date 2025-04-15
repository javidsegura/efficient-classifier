from library.pipeline.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
import numpy as np
class PipelinesAnalysis:
      def __init__(self, pipelines: dict[str, dict[str, Pipeline]]):
            self.pipelines = pipelines
            self.encoded_map = None
            self.phase = None
      
      def _compute_classification_report(self, include_training: bool = False):
            """
            Plots the classification report of a given model
            """
            assert self.phase in ["pre", "in", "post"], "Phase must be either pre, in or post"
            classification_reports = []
            for category in self.pipelines:
                  for pipeline in self.pipelines[category]:
                              if pipeline not in ["ensembled", "tree-based"]:
                                    continue
                              for modelName in self.pipelines[category][pipeline].model_selection.list_of_models:
                                    if modelName not in self.pipelines[category][pipeline].model_selection.models_to_exclude:
                                          if self.phase != "post":
                                                      y_pred = self.pipelines[category][pipeline].model_selection.list_of_models[modelName].tuning_states[self.phase].assesment["predictions_val"]
                                                      y_true = self.pipelines[category][pipeline].model_selection.dataset.y_val
                                                      not_training_report = classification_report(y_true, y_pred, output_dict=True)
                                                      not_training_report["modelName"] = modelName
                                                      classification_reports.append(pd.DataFrame(not_training_report))
                                                      if include_training:
                                                            y_pred_train = self.pipelines[category][pipeline].model_selection.list_of_models[modelName].tuning_states[self.phase].assesment["predictions_train"]
                                                            y_true_train = self.pipelines[category][pipeline].model_selection.dataset.y_train
                                                            training_report = classification_report(y_true_train, y_pred_train, output_dict=True)
                                                            training_report["modelName"] = modelName + "_train"
                                                            classification_reports.append(pd.DataFrame(training_report))
                                          else:
                                                      y_pred = self.pipelines[category][pipeline].model_selection.list_of_models[modelName].tuning_states[self.phase].assesment["predictions_test"]
                                                      y_true = self.pipelines[category][pipeline].model_selection.dataset.y_test
                                                      not_training_report = classification_report(y_true, y_pred, output_dict=True)
                                                      not_training_report["modelName"] = modelName
                                                      classification_reports.append(pd.DataFrame(not_training_report))
                                                      if include_training:
                                                            y_pred_train = self.pipelines[category][pipeline].model_selection.list_of_models[modelName].tuning_states[self.phase].assesment["predictions_train"]
                                                            y_true_train = self.pipelines[category][pipeline].model_selection.dataset.y_train
                                                            training_report = classification_report(y_true_train, y_pred_train, output_dict=True)
                                                            training_report["modelName"] = modelName + "_train"
                                                            classification_reports.append(pd.DataFrame(training_report))

            self.classification_report = pd.concat(classification_reports).T
            
            if self.encoded_map is not None:
                  reverse_map = {str(v): k for k, v in self.encoded_map.items()} #{number:name}
                  index = self.classification_report.index.tolist()
                  new_index = []
                  for idx in index:
                        if idx in reverse_map:  
                              new_index.append(reverse_map[idx])
                        else:  
                              new_index.append(idx)
                  self.classification_report.index = new_index
            
            return self.classification_report
      
      def plot_cross_model_comparison(self, metric: list[str], cols:int = 2):
            """
            Plots the classification report of a given model
            """
            assert self.phase in ["pre", "in", "post"], "Phase must be either pre, in or post"
            class_report_df = self._compute_classification_report()

            num_metrics = len(metric)
            cols = cols
            rows =  math.ceil(num_metrics / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 7))
            axes = axes.flatten()  # Flatten to iterate easily, even if 1 row

            for i, metric in enumerate(metric):
                  class_report_cols = class_report_df.columns
                  assert metric in class_report_cols, f"Metric not present in {class_report_cols}"
                  ax = axes[i]
                  metric_df = class_report_df[metric]
                  df_numeric = metric_df.iloc[:-1].astype(float)  
                  model_names = metric_df.iloc[-1].values        

                  # Plotting
                  ax.plot(df_numeric.index, df_numeric.iloc[:, 0], marker='o', label=model_names[0])
                  ax.plot(df_numeric.index, df_numeric.iloc[:, 1], marker='s', label=model_names[1])

                  ax.set_title(f'{metric} by Model')
                  ax.set_xlabel('Class Index')
                  ax.set_ylabel(metric)
                  ax.tick_params(axis='x', rotation=45)
                  ax.legend()
                  ax.grid(True)

            # Hide any unused subplots
            for j in range(i + 1, len(axes)):
                  fig.delaxes(axes[j])

            plt.tight_layout()
            plt.suptitle(f"Cross-model Performance Comparison - {self.phase} phase")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()


      def plot_results_df(self, metrics: list[str]):
            """
            For all the metrics it plots all the trained models 
            """
            assert self.phase in ["pre", "in", "post"], "Phase must be either pre, in or post"
            dataframes = []
            for category in self.pipelines:
                  for pipeline in self.pipelines[category]:
                        df = self.pipelines[category][pipeline].model_selection.results_analysis[self.phase].phase_results_df
                        dataframes.append(df)
            metrics_df = pd.concat(dataframes)

            num_metrics = len(metrics)
            cols = 2
            rows = math.ceil(num_metrics / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
            axes = axes.flatten()  # Flatten to iterate easily, even if 1 row

            for i, metric in enumerate(metrics):
                  ax = axes[i]
                  sns.barplot(data=metrics_df, x='modelName', y=metric, ax=ax, palette="viridis")
                  ax.set_title(f'{metric} by Model')
                  ax.set_xlabel('Model Name')
                  ax.set_ylabel(metric)
                  ax.tick_params(axis='x', rotation=45)

                  # Annotate values
                  for container in ax.containers:
                        ax.bar_label(container, fmt='%.4f', label_type='edge')

            # Hide any unused subplots
            for j in range(i + 1, len(axes)):
                  fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()
      
      def pot_feature_importance(self):
            """
            Plots the feature importance of a given model
            """
            assert self.phase in ["pre", "in", "post"], "Phase must be either pre, in or post"
            importances_dfs = {}
            for category in self.pipelines:
                  for pipeline in self.pipelines[category]:
                              if pipeline not in ["ensembled", "tree-based"]:
                                    continue
                              for modelName in self.pipelines[category][pipeline].model_selection.list_of_models:
                                    if modelName not in self.pipelines[category][pipeline].model_selection.models_to_exclude:
                                          importances = self.pipelines[category][pipeline].model_selection.list_of_models[modelName].tuning_states[self.phase].assesment["model_sklearn"].feature_importances_
                                          feature_importance_df = pd.DataFrame({
                                                                            'Feature': self.pipelines[category][pipeline].dataset.X_train.columns,
                                                                            'Importance': importances
                                                                            }).sort_values(by='Importance', ascending=False)
                                          importances_dfs[pipeline] = feature_importance_df
            for pipeline in importances_dfs:
                  sns.barplot(
                        x="Importance",
                        y="Feature",
                        data=importances_dfs[pipeline]
                        )
                  plt.title(f"Feature Importances for {pipeline} model")
                  plt.show()
            return importances_dfs

      def plot_confusion_matrix(self):
            """
            Plots the confusion matrix of a given model
            """
            assert self.phase in ["pre", "in", "post"], "Phase must be either pre, in or post"
            confusion_matrices = {}
            residuals = {}
            for category in self.pipelines:
                  for pipeline in self.pipelines[category]:
                        for modelName in self.pipelines[category][pipeline].model_selection.list_of_models:
                              if modelName not in self.pipelines[category][pipeline].model_selection.models_to_exclude:
                                    pred = self.pipelines[category][pipeline].model_selection.list_of_models[modelName].tuning_states[self.phase].assesment["predictions_val"]
                                    actual = self.pipelines[category][pipeline].model_selection.dataset.y_val
                                    residuals[pipeline] = self.pipelines[category][pipeline].model_selection.dataset.y_val[pred != actual]
                                    cm = confusion_matrix(actual, pred)
                                    confusion_matrices[modelName] = {
                                                                    "absolute": cm,
                                                                    "relative": cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                                                                    }
            
            fig, axes = plt.subplots(len(confusion_matrices), 2, figsize=(15, 5* len(confusion_matrices)))
            # Convert axes to 2D array if there's only one model
            if len(confusion_matrices) == 1:
                  axes = np.array([axes])
                  
            # Get category labels if encoded_map exists
            labels = None
            if self.encoded_map is not None:
                  # Sort by encoded value to ensure correct order
                  labels = [k for k, v in sorted(self.encoded_map.items(), key=lambda x: x[1])]
            
            for i, (modelName, cm_data) in enumerate(confusion_matrices.items()):
                  # Absolute Confusion Matrix
                  sns.heatmap(cm_data["absolute"], 
                        annot=True, 
                        fmt='d',  # 'd' for integers in absolute matrix
                        cmap='Blues',
                        ax=axes[i, 0],
                        xticklabels=labels,
                        yticklabels=labels)
                  axes[i, 0].set_title(f"Absolute Confusion Matrix for model: {modelName}")
                  axes[i, 0].set_xlabel("Predicted")
                  axes[i, 0].set_ylabel("Actual")

                  # Relative Confusion Matrix
                  sns.heatmap(cm_data["relative"], 
                        annot=True, 
                        fmt='.1f',  # .1f for one decimal place in relative matrix
                        cmap='Blues',
                        ax=axes[i, 1],
                        xticklabels=labels,
                        yticklabels=labels)
                  axes[i, 1].set_title(f"Relative Confusion Matrix for model: {modelName}")
                  axes[i, 1].set_xlabel("Predicted")
                  axes[i, 1].set_ylabel("Actual")
            
            plt.tight_layout()
            plt.suptitle(f"Confusion Matrix - {self.phase} phase")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
            self.residuals = residuals
            return residuals, confusion_matrices
      
      def plot_intra_model_comparison(self, metrics: list[str]):
            """
            3 cols each with two trends. As many rows as unique models
            """
            class_report_df = self._compute_classification_report(include_training=True)
            models = class_report_df.T["modelName"].unique()
            models = {model.split("_")[0] for model in models}
            
            num_metrics = len(metrics)
            cols = num_metrics
            rows = len(models)

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
            
            for i, model in enumerate(models):
                class_report_cols = class_report_df.columns
                for j, metric in enumerate(metrics):
                    assert metric in class_report_cols, f"Metric not present in {class_report_cols}"
                    model_filter = class_report_df.T["modelName"].str.startswith(model)
                    model_df = class_report_df.T[model_filter]
                    ax = axes[i, j]
                    metric_df = model_df.T[metric]
                    df_numeric = metric_df.iloc[:-1].astype(float)  
                    model_names = metric_df.iloc[-1].values

                    # Plotting
                    ax.plot(df_numeric.index, df_numeric.iloc[:, 0], marker='o', label=model_names[0])
                    ax.plot(df_numeric.index, df_numeric.iloc[:, 1], marker='s', label=model_names[1])

                    ax.set_title(f'{metric} - train vs no-train')
                    ax.set_xlabel('Class Index')
                    ax.set_ylabel(metric)
                    ax.tick_params(axis='x', rotation=45)
                    ax.legend()
                    ax.grid(True)

            # Hide any unused subplots
            for j in range(i + 1, len(axes)):
                  fig.delaxes(axes[j])

            plt.tight_layout()
            plt.tight_layout(rect=[0, 0, 1, 0.96])  
            plt.suptitle(f"Intra-model Perfomance Comparison - {self.phase} phase")
            plt.show()

                
            


