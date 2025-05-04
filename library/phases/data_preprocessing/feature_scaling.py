from library.phases.dataset.dataset import Dataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureScaling:
  def __init__(self, dataset: Dataset) -> None:
    self.dataset = dataset
  
  def scale_features(
        self,
        scaler: str,
        columnsToScale: list[str],
        plot: bool = False,
    ) -> str:
      """
      Scales the features in the dataset

      Parameters:
      -----------
      scaler : str
          The scaler to use ('minmax', 'robust', 'standard')
      columnsToScale : list[str]
          The columns to scale
      plot : bool
          Whether to plot distributions before and after scaling

      Returns:
      --------
      str
          Message indicating the number of features scaled  
      """
      assert len(columnsToScale) > 0, "Columns to scale must be provided"

      if scaler == "minmax":
          scaler_obj = MinMaxScaler()
      elif scaler == "robust":
          scaler_obj = RobustScaler()
      elif scaler == "standard":
          scaler_obj = StandardScaler()
      else:
          raise ValueError(f"Invalid scaler: {scaler}")

      # Optionally store original data for plotting
      if plot:
          original_data = self.dataset.X_train[columnsToScale].copy()

      # Apply transformation
      self.dataset.X_train[columnsToScale] = scaler_obj.fit_transform(self.dataset.X_train[columnsToScale])
      self.dataset.X_val[columnsToScale] = scaler_obj.transform(self.dataset.X_val[columnsToScale])
      self.dataset.X_test[columnsToScale] = scaler_obj.transform(self.dataset.X_test[columnsToScale])

      # Plot only the first 10 columns if requested
      if plot:
          max_plots = 5
          plot_columns = columnsToScale[:max_plots]
          for col in plot_columns:
              fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
              sns.histplot(original_data[col], kde=True, ax=axes[0])
              axes[0].set_title(f"{col} - Before Scaling")
              sns.histplot(self.dataset.X_train[col], kde=True, ax=axes[1])
              axes[1].set_title(f"{col} - After Scaling")
              plt.tight_layout()
              plt.show()

      return f"Succesfully scaled {len(columnsToScale)} features. Plotted distributions for the first {min(10, len(columnsToScale))} features." \
            f"\nTo check the results run: \n your_pipeline.dataset.X_train.head()"