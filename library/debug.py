""" A file for debugging purposes """

import pandas as pd

PATH_DATASET = "../II/dataset/hour.csv"
PATH_RESULTS = "II/results/results.csv"

dataToWrite = {
      "a": 10,
      "c": 9,
      "d": -1
}

results = pd.read_csv(PATH_RESULTS)

print(results[list(dataToWrite)])
print(pd.Series(dataToWrite))

isNewModel = not(((results[list(dataToWrite)] == pd.Series(dataToWrite)).all(axis=1)).any())

print(isNewModel)


