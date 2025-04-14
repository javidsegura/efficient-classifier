

from library.phases.dataset.dataset import Dataset
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
import math

from abc import ABC, abstractmethod


class ResultAnalysis(ABC):
      def __init__(self, phase_results_df: pd.DataFrame):
            self.phase_results_df = phase_results_df

      def plot_multiple_model_metrics(self,feature_list):
            num_features = len(feature_list)
            cols = 2
            rows = math.ceil(num_features / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
            axes = axes.flatten()  # Flatten to iterate easily, even if 1 row

            for i, feature in enumerate(feature_list):
                  ax = axes[i]
                  sns.barplot(data=self.phase_results_df, x='modelName', y=feature, ax=ax)
                  ax.set_title(f'{feature} by Model')
                  ax.set_xlabel('Model Name')
                  ax.set_ylabel(feature)
                  ax.tick_params(axis='x', rotation=45)

                  # Annotate values
                  for container in ax.containers:
                        ax.bar_label(container, fmt='%.4f', label_type='edge')

            # Hide any unused subplots
            for j in range(i + 1, len(axes)):
                  fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()


      @abstractmethod
      def plot_results(self):
            """ scatterplot and histogram of the results """
            pass

      @abstractmethod
      def feature_importance(self):
            pass

      @abstractmethod
      def extract_metrics(self):
            pass

class PreTuningResultAnalysis(ResultAnalysis):
      def __init__(self, phase_results_df: pd.DataFrame):
            super().__init__(phase_results_df)

      def plot_results(self):
            pass

      def feature_importance(self):
            pass

      def extract_metrics(self):
            pass

class InTuningResultAnalysis(ResultAnalysis):
      def __init__(self, phase_results_df: pd.DataFrame):
            super().__init__(phase_results_df)

      def plot_results(self):
            pass

      def feature_importance(self):
            pass

      def extract_metrics(self):
            pass

class PostTuningResultAnalysis(ResultAnalysis):
      def __init__(self, phase_results_df: pd.DataFrame):
            super().__init__(phase_results_df)

      def plot_results(self):
            pass

      def feature_importance(self):
            pass

      def extract_metrics(self):
            pass
