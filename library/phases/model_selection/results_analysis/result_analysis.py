

from library.phases.dataset.dataset import Dataset
import pandas as pd 

from abc import ABC, abstractmethod


class ResultAnalysis(ABC):
      def __init__(self, results_df: pd.DataFrame, dataset: Dataset):
            self.results_df = results_df
            self.dataset = dataset

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
      def __init__(self, results_df: pd.DataFrame, dataset: Dataset):
            self.results_df = results_df
            self.dataset = dataset

      def plot_results(self):
            pass

      def feature_importance(self):
            pass

      def extract_metrics(self):
            pass

class InTuningResultAnalysis(ResultAnalysis):
      def __init__(self, results_df: pd.DataFrame, dataset: Dataset):
            self.results_df = results_df
            self.dataset = dataset

      def plot_results(self):
            pass

      def feature_importance(self):
            pass

      def extract_metrics(self):
            pass

class PostTuningResultAnalysis(ResultAnalysis):
      def __init__(self, results_df: pd.DataFrame, dataset: Dataset):
            self.results_df = results_df
            self.dataset = dataset

      def plot_results(self):
            pass

      def feature_importance(self):
            pass

      def extract_metrics(self):
            pass
