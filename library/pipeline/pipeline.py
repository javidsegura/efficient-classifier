from library.phases.phases_implementation.dataset.dataset import Dataset
from library.phases.phases_implementation.EDA.EDA import EDA
from library.phases.phases_implementation.data_preprocessing.data_preprocessing import DataPreprocessing
from library.phases.phases_implementation.feature_analysis.feature_analysis import FeatureAnalysis
from library.phases.phases_implementation.modelling.modelling import Modelling

# Global variables
RANDOM_STATE = 99

class Pipeline:
      def __init__(self, dataset_path: str, 
                   results_path: str,
                   model_task: str, 
                   random_state: int = RANDOM_STATE):
            self.dataset = Dataset(dataset_path, model_task, random_state)
            self.EDA = EDA(self.dataset)
            self.data_preprocessing = DataPreprocessing(self.dataset)
            self.feature_analysis = FeatureAnalysis(self.dataset)
            self.modelling = Modelling(self.dataset, results_path)
      
      def speak(self, message: str) -> None:
            """
            This is just an example function used to illustrate the pipelines functionality.

            Parameters
            ----------
            message : str
                  The message to print
            """
            print(f"{message} from {id(self)}")