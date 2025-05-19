from efficient_classifier.phases.phases_implementation.dataset.dataset import Dataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from efficient_classifier.utils.miscellaneous.save_or_store_plot import save_or_store_plot

import yaml

class FeatureScaling:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.variables = yaml.load(open("efficient_classifier/configurations.yaml"), Loader=yaml.FullLoader)

    def scale_features(
        self,
        scaler: str,
        columnsToScale: list[str],
        save_plots: bool = False,
        save_path: str = None
    ) -> str:
        """
        Scales the features in the dataset

        Parameters
        ----------
        scaler : str
            The scaler to use ('minmax', 'robust', 'standard')
        columnsToScale : list[str]
            The columns to scale
        save_plots : bool
            Whether to save plots before and after scaling
        save_path : str
            The path to save the plots

        Returns
        -------
        str
            Message indicating the number of features scaled  
        """

        # --- Input validation ---
        if not isinstance(scaler, str):
            raise TypeError("Parameter 'scaler' must be a string.")
        if len(columnsToScale) == 0:
            raise ValueError("At least one column must be provided for scaling.")
        if not isinstance(save_plots, bool):
            raise TypeError("Parameter 'save_plots' must be a boolean.")

        # --- Dataset validation ---
        for attr in ['X_train', 'X_val', 'X_test']:
            if not hasattr(self.dataset, attr):
                raise AttributeError(f"The dataset is missing the attribute '{attr}'.")
            missing_cols = [col for col in columnsToScale if col not in getattr(self.dataset, attr).columns]
            if missing_cols:
                raise ValueError(f"The following columns are missing in '{attr}': {missing_cols}")

        # --- Scaler selection ---
        if scaler == "minmax":
            scaler_obj = MinMaxScaler()
        elif scaler == "robust":
            scaler_obj = RobustScaler()
        elif scaler == "standard":
            scaler_obj = StandardScaler()
        else:
            raise ValueError(f"Invalid scaler: {scaler}. Choose from 'minmax', 'robust', or 'standard'.")

        # --- Optional: store original data for plotting ---
        if save_plots:
            try:
                original_data = self.dataset.X_train[columnsToScale].copy()
            except Exception as e:
                raise RuntimeError(f"Failed to copy original data for plotting: {e}")

        # --- Apply scaling ---
        try:
            self.dataset.X_train[columnsToScale] = scaler_obj.fit_transform(self.dataset.X_train[columnsToScale])
            self.dataset.X_val[columnsToScale] = scaler_obj.transform(self.dataset.X_val[columnsToScale])
            self.dataset.X_test[columnsToScale] = scaler_obj.transform(self.dataset.X_test[columnsToScale])
        except Exception as e:
            raise RuntimeError(f"An error occurred during scaling: {e}")

        # --- Optional: plot distributions ---
        if save_plots:
            try:
                max_plots = self.variables["PIPELINE_RUNNER"]["max_plots_per_function"]
                plot_columns = columnsToScale[:max_plots] if max_plots > 0 else columnsToScale
                for col in plot_columns:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
                    sns.histplot(original_data[col], kde=True, ax=axes[0])
                    axes[0].set_title(f"{col} - Before Scaling")
                    sns.histplot(self.dataset.X_train[col], kde=True, ax=axes[1])
                    axes[1].set_title(f"{col} - After Scaling")
                    plt.tight_layout()
                    save_or_store_plot(fig, save_plots, save_path + "/feature_scaling", f"{col}.png")
            except Exception as e:
                raise RuntimeError(f"An error occurred while plotting: {e}")

        return (
            f"Successfully scaled {len(columnsToScale)} features. "
            f"Plotted distributions for the first {min(5, len(columnsToScale))} features.\n"
            f"To check the results run:\n your_pipeline.dataset.X_train.head()"
        )
