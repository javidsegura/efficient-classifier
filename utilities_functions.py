import time

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import concurrent.futures


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, classification_report


def compute_correlation_matrix(figsize: tuple[int, int], dataframe: pd.DataFrame):
  """Computes and plots (efficiently) the correlation heatmap"""
  corr = dataframe.corr()
  mask = np.triu(np.ones_like(corr, dtype=bool)) # avoid redundancy
  f, ax = plt.subplots(figsize=figsize)
  cmap = sns.diverging_palette(230, 20, as_cmap=True)
  vmin, vmax = corr.min().min(), corr.max().max()
  sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8}, vmin=vmin, vmax=vmax)

def basic_distribution_plot(x_axis: pd.Series, title: str, xlabel: str, ylabel: str, bins: int = 30, size: str = "small"):
  """Plots a distribution in a more straight-forward way. Very general subroutine"""
  if size == "s":
    plt.figure(figsize=(5, 3))
  elif size == "m":
    plt.figure(figsize=(10, 6))
  elif size == "l":
    plt.figure(figsize=(20, 15))
  elif size == "xl":
    plt.figure(figsize=(30, 20))
  plt.hist(x_axis, bins=bins, edgecolor='black')
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()

def get_X_y(df: pd.DataFrame, y_column: str, otherColumnsToDrop: list[str] = []) -> tuple[pd.DataFrame, pd.Series]:
  """Splits the dataframe into features and target variable"""
  X = df.drop(columns=[y_column] + otherColumnsToDrop)
  y = df[y_column]
  return X, y

def asses_split(df, p, step):
      computeSE = lambda p, n : np.sqrt((p*(1-p))/n)
      df_split_assesment = pd.DataFrame()
      hold_out_size = step
      p = .85
      priorSE = 0
      while hold_out_size <= .50: # very generous upper-bound split
            assert hold_out_size < 1 
            train_size_percentage  = 1 - hold_out_size
            train_size_count = round(df.shape[0] * train_size_percentage, 0)

            val_size_percentage = hold_out_size / 2
            val_size_count = round(df.shape[0] * (hold_out_size / 2),0)

            test_size_percentage = hold_out_size / 2
            test_size_count = round(df.shape[0] * (hold_out_size / 2),0)


            currentSE = computeSE(p, test_size_count)
            differenceToPriorSE = currentSE - priorSE
            differenceToPriorSE_percentage = (currentSE - priorSE) /  priorSE
            priorSE = currentSE

            new_row = pd.DataFrame([{
            "train_size (%)": train_size_percentage, 
            "train_size_count": train_size_count,
            "validation_size (%)": val_size_percentage ,
            "validation_size_count": val_size_count,
            "test_size (%)": test_size_percentage, 
            "test_size_coount": test_size_count,
            "currentSE": currentSE ,
            "differenceToPriorSE": differenceToPriorSE,
            "differenceToPriorSE (%)": differenceToPriorSE_percentage,
            }])

            # Concatenate the new row with your existing DataFrame
            df_split_assesment = pd.concat([df_split_assesment, new_row], ignore_index=True)
            hold_out_size += step
      return df_split_assesment

def get_split_data(X: pd.DataFrame, y: pd.Series, train_size: float = 0.8, validation_size: float = 0.1, test_size: float = 0.1, random_state: int = 99) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
  """Splits the dataframe into training, validation and test sets"""
  assert train_size + validation_size + test_size == 1, "The sum of the sizes must be 1"
  X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=validation_size + test_size, random_state=random_state) 
  X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(validation_size + test_size), random_state=random_state) 
  return X_train, X_validation, X_test, y_train, y_validation, y_test

def get_X_sets_encoded(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Encodes the categorical features for the training, validation and test sets"""
  encoder = OneHotEncoder(handle_unknown="ignore", 
                        sparse_output=False,
                        dtype=int,
                        drop="first"
                        )
  cat_cols = X_train.select_dtypes(include=["object"]).columns
  # Training set
  encoded_array = encoder.fit_transform(X_train[cat_cols])
  encoded_cols = encoder.get_feature_names_out(cat_cols)
  train_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=X_train.index)
  X_train_encoded = X_train.drop(cat_cols, axis=1).join(train_encoded)
  # Validation set
  encoded_array_val = encoder.transform(X_val[cat_cols])
  val_encoded = pd.DataFrame(encoded_array_val, columns=encoded_cols, index=X_val.index)
  X_val_encoded = X_val.drop(cat_cols, axis=1).join(val_encoded)
  # Test set
  encoded_array_test = encoder.transform(X_test[cat_cols])
  test_encoded = pd.DataFrame(encoded_array_test, columns=encoded_cols, index=X_test.index)
  X_test_encoded = X_test.drop(cat_cols, axis=1).join(test_encoded)
  return X_train_encoded, X_val_encoded, X_test_encoded

def get_y_sets_encoded(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
  labeller = LabelEncoder()
  labeller.fit(y_train)
  y_train_encoded = pd.Series(labeller.transform(y_train), index=y_train.index)
  y_val_encoded = pd.Series(labeller.transform(y_val), index=y_val.index)
  y_test_encoded = pd.Series(labeller.transform(y_test), index=y_test.index)

  encoding_map = dict(zip(labeller.classes_, range(len(labeller.classes_))))

  return y_train_encoded, y_val_encoded, y_test_encoded, encoding_map

def evaluate_classifier(y_actual: pd.Series, y_pred: pd.Series, display_results: bool = False, plot_confusion_matrix: bool = False) -> tuple[float, str, np.ndarray]:
  """ Returns commonc classification metrics, plots confusion matrix if requested, writes to stdout if requested"""
  class_report = classification_report(y_actual, y_pred) # F1 score, precision, recall for each class
  conf_matrix = confusion_matrix(y_actual, y_pred)
  if display_results:
    print(f"Validation Classification Report: \n{class_report}")
  if plot_confusion_matrix:
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
  return class_report, conf_matrix

def eliminate_variables_from_set(listOfSets: list[pd.DataFrame], listOfVariables: list[str]):
  """Eliminates variables from a list of sets"""
  lengthOfListOfSets = [len(set) for set in listOfSets]
  for set in listOfSets:
    set.drop(columns=listOfVariables, 
             inplace=True,
             errors='ignore') # ignore errors if the variable is not in the set
  for i in range(len(listOfSets)):
    if len(listOfSets[i]) == lengthOfListOfSets[i]:
      print(f"No modifications were made to the set {i}")

def __fit_and_predict__(model_item, X_category_train_encoded: pd.DataFrame, y_category_train_encoded: pd.Series, X_category_val_encoded: pd.DataFrame, X_category_test_encoded: pd.DataFrame):
    classifierName, classifier = model_item
    start_time = time.time()
    print(f"!> Started fitting {classifierName}")
    classifier["model"].fit(X_category_train_encoded, y_category_train_encoded)
    end_time = time.time()
    time_to_fit = end_time - start_time
    classifier["timeToFit"] = time_to_fit
    print(f"\t\t => Fitted {classifierName}. Took {time_to_fit} seconds")
    
    classifierName, classifier = model_item
    print(f"!> Started predicting for {classifierName}")
    start_time = time.time()
    classifier["val_predictions"] = classifier["model"].predict(X_category_val_encoded)
    classifier["test_predictions"] = classifier["model"].predict(X_category_test_encoded)
    end_time = time.time()
    classifier["timeToMakePredictions"] = end_time - start_time
    print(f"\t\t => Predicted {classifierName}. Took {classifier['timeToMakePredictions']} seconds")
    return classifierName, classifier
  
def fit_models(models: list[dict], X_category_train_encoded: pd.DataFrame, y_category_train_encoded: pd.Series, X_category_val_encoded: pd.DataFrame, X_category_test_encoded: pd.DataFrame) -> dict:
  """Fits and predicts the models in parallel"""
  with concurrent.futures.ProcessPoolExecutor() as executor:
      # Submit all model fitting tasks to the executor
      future_to_model = {executor.submit(__fit_and_predict__, item, X_category_train_encoded, y_category_train_encoded, X_category_val_encoded, X_category_test_encoded): item for item in models.items()}
      
      for future in concurrent.futures.as_completed(future_to_model):
          classifierName, classifier = future.result()
          models[classifierName] = classifier # this is not doing nothing new. Its just to way for the process pool to return the results
      
  print("All models have been fitted and made predictions in parallel.")
  return models
  
