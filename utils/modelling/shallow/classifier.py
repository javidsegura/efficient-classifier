
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


from utils.dataset import Dataset
from utils.modelling.shallow.base import ModelAssesment

class ClassifierAssesment(ModelAssesment):
  """
  This class is used to assess the performance of a classifier.
  """
  def __init__(self, dataset: Dataset) -> None:
    super().__init__(dataset)
  
  def __set_assesment__(self, y_actual: pd.Series, y_pred: pd.Series, plot: bool = True):
    """
    Assesment of the model in a given set 

    Parameters
    ----------
      y_actual : pd.Series
        The actual labels
      y_pred : pd.Series
        The predicted labels
      plot : bool
        Whether to plot the results

    Returns
    -------
      tuple
      The classification report and the confusion matrix
    """
    class_report = classification_report(y_actual, y_pred, output_dict=True) # F1 score, precision, recall for each class
    conf_matrix = confusion_matrix(y_actual, y_pred)
    if plot:
      print(f"Validation Classification Report: \n{class_report}")
      sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
      plt.xlabel("Predicted Label")
      plt.ylabel("True Label")
      plt.title("Confusion Matrix")
      plt.show()
    return class_report, conf_matrix
  
  def evaluate_classifier(self, plot: bool = True, modelsToExclude: list = []):
    for modelName, model in self.models.items():
      if modelName in modelsToExclude or model["test_predictions"] is None or model["val_predictions"] is None:
        continue
      print(f"Evaluating {modelName}")
      print(f"\t => VALIDATION ASSESMENT:")
      y_actual_val = self.dataset.y_val_encoded
      y_pred_val = model["val_predictions"]
      self.__set_assesment__(y_actual_val, y_pred_val, plot)
      print(f"\t => TEST ASSESMENT:")
      y_actual_test = self.dataset.y_test_encoded
      y_pred_test = model["test_predictions"]
      class_report, confusion_matrix = self.__set_assesment__(y_actual_test, y_pred_test, plot)
      self.models[modelName]["metrics"] = {
        "class_report": class_report,
        "confusion_matrix": confusion_matrix
      }
