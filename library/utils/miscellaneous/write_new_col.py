

import pandas as pd

PATH = "results/model_evaluation/results.csv"

df = pd.read_csv(PATH)

print(df.head())

df["your_great_new_metric"] = "N/A"

df.to_csv(PATH, index=False, lineterminator=',\n')

