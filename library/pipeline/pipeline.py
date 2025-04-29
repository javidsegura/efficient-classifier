from library.phases.dataset.dataset import Dataset
from library.phases.EDA.EDA import EDA
from library.phases.data_preprocessing.data_preprocessing import DataPreprocessing
from library.phases.feature_analysis.feature_analysis import FeatureAnalysis
from library.phases.modelling.modelling import Modelling

# Global variables
RANDOM_STATE = 99

class Pipeline:
      def __init__(self, dataset_path: str, 
                   results_path: str,
                   model_type: str, 
                   random_state: int = RANDOM_STATE):
            self.dataset = Dataset(dataset_path, model_type, random_state)
            self.EDA = EDA(self.dataset)
            self.data_preprocessing = DataPreprocessing(self.dataset)
            self.feature_analysis = FeatureAnalysis(self.dataset)
            self.modelling = Modelling(self.dataset, results_path)
      
      def speak(self, message: str):
            print(f"{message} from {id(self)}")