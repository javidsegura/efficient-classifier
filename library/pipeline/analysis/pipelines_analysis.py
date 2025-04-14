
from library.pipeline.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
class PipelinesAnalysis:
      def __init__(self, pipelines: dict[str, dict[str, Pipeline]]):
            self.pipelines = pipelines

      def analyze_pipelines(self):
            pass
      
      def plot_results_metrics(self, metrics: list[str], phase: str = "pre"):
            """
            For all the metrics it plots all the trained models 
            """
            assert phase in ["pre", "in", "post"], "Phase must be either pre, in or post"
            dataframes = []
            for category in self.pipelines:
                  for pipeline in self.pipelines[category]:
                        df = self.pipelines[category][pipeline].model_selection.results_analysis[phase].phase_results_df
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
      
      def pot_feature_importance(self, phase: str = "pre"):
            """
            Plots the feature importance of a given model
            """
            assert phase in ["pre", "in", "post"], "Phase must be either pre, in or post"
            importances_dfs = {}
            for category in self.pipelines:
                  for pipeline in self.pipelines[category]:
                              if pipeline not in ["ensembled", "tree-based"]:
                                    continue
                              for modelName in self.pipelines[category][pipeline].model_selection.list_of_models:
                                    if modelName not in self.pipelines[category][pipeline].model_selection.models_to_exclude:
                                          importances = self.pipelines[category][pipeline].model_selection.list_of_models[modelName].tuning_states[phase].assesment["model_sklearn"].feature_importances_
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

            
