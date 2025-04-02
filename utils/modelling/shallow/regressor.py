
from utils.dataset import Dataset
from utils.modelling.shallow.base import ModelAssesment

class RegressorAssesment(ModelAssesment):
  """
  This class is used to assess the performance of a regressor.
  """
  def __init__(self, dataset: Dataset) -> None:
    super().__init__(dataset)
