
from library.phases.phases_implementation.dataset.dataset import Dataset
from library.phases.phases_implementation.feature_analysis.feature_selection.automatic import AutomaticFeatureSelection
from library.phases.phases_implementation.feature_analysis.feature_selection.manual import ManualFeatureSelection

class FeatureSelection:
      """
      """
      def __init__(self, dataset: Dataset) -> None:
            """
            Initializes the feature selection class. Controls all the feature selection methods.

            Parameters
            ----------
            dataset : Dataset
                  The dataset to select features from
            """
            self.dataset = dataset
            self.automatic_feature_selection = AutomaticFeatureSelection(self.dataset)
            self.manual_feature_selection = ManualFeatureSelection(self.dataset)
      


    
        