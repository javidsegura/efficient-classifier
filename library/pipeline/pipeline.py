from library.phases.dataset.dataset import Dataset
from library.phases.EDA.EDA import EDA
from library.phases.data_preprocessing.data_preprocessing import Preprocessing
from library.phases.feature_analysis.feature_analysis import FeatureAnalysis
from library.phases.model_selection.model_selection import ModelSelection

# Global variables
RANDOM_STATE = 99

class Pipeline:
      def __init__(self, dataset_path: str, 
                   results_path: str,
                   model_type: str, 
                   metrics_to_evaluate: list[str],
                   random_state: int = RANDOM_STATE):
            self.dataset = Dataset(dataset_path, model_type, random_state)
            self.EDA = EDA(self.dataset)
            self.preprocessing = Preprocessing(self.dataset)
            self.feature_analysis = FeatureAnalysis(self.dataset)
            self.model_selection = ModelSelection(self.dataset, results_path, metrics_to_evaluate)