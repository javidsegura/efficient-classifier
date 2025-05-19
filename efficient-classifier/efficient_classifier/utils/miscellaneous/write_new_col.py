import pandas as pd

PATH = "results/model_evaluation/prior_csvs/april_15.csv"


def add_new_metric(new_metric_name):

      df = pd.read_csv(PATH)

      # Find the position of the first _test column
      first_test_col = next(col for col in df.columns if col.endswith('_test'))
      test_col_idx = df.columns.get_loc(first_test_col)

      # Insert the val before the first _test column
      df.insert(test_col_idx, new_metric_name + "_val", "N/A")
      # Append the test after the last column
      df.insert(len(df.columns), new_metric_name + "_test", "N/A")

      # Debug
      print(df.head())
      print(df.columns)

      # Save the updated DataFrame
      df.to_csv(PATH, index=False, lineterminator=',\n')


add_new_metric("kappa") # Example
