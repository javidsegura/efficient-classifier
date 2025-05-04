from library.phases.phases_implementation.data_preprocessing.bounds_config import BOUNDS
from library.phases.phases_implementation.dataset.dataset import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class OutliersBounds:
    def __init__(self, dataset: Dataset) -> None:
      self.dataset = dataset
    
    def bound_checking(self) -> None:
        """
        Apply numeric *BOUNDS* to *dataset.df* and remove rare violators.

        The global constant :data:`BOUNDS` must map column names to
        (min, max) tuples. For each column, the helper will:
        - Drop rows that lie outside the interval when they represent < 0.5% of the total dataset
        - Keep (but record) them for manual analysis otherwise

        Returns
        -------
        None
        """

        # --- Check dataset validity ---
        if not hasattr(self.dataset, "df"):
            raise AttributeError("The dataset does not contain an attribute named 'df'.")
        if not isinstance(self.dataset.df, pd.DataFrame):
            raise TypeError("self.dataset.df must be a pandas DataFrame.")

        # --- Validate BOUNDS constant ---
        if not isinstance(BOUNDS, dict):
            raise TypeError("BOUNDS must be a dictionary mapping column names to (min, max) tuples.")
        if not all(isinstance(v, tuple) and len(v) == 2 for v in BOUNDS.values()):
            raise ValueError("Each value in BOUNDS must be a (min, max) tuple.")

        self.bound_cols, self.bound_limits = zip(*BOUNDS.items())

        # --- Check all bound columns exist ---
        missing_cols = [col for col in self.bound_cols if col not in self.dataset.df.columns]
        if missing_cols:
            raise ValueError(f"The following columns in BOUNDS are missing from the dataset: {missing_cols}")

        # --- Delegate to helper ---
        try:
            self.outliers_dict = self._bound_checking_helper(
                columnsToCheck=list(self.bound_cols),
                bounds=list(self.bound_limits)
            )
        except Exception as e:
            raise RuntimeError(f"An error occurred during bound checking: {e}")

        return None
  
    def _bound_checking_helper(self, columnsToCheck: list[str] = [], bounds: list[tuple] = []) -> dict[str, pd.DataFrame]:
        """
        Low-level helper that implements the actual bound filtering.

        Parameters
        ----------
        columnsToCheck : list[str]
            Column names to validate.
        bounds : list[tuple[float, float]]
            Sequence of (min, max) intervals for each column.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of column name ➟ offending rows (if any).
        """

        # --- Input validation ---
        if not columnsToCheck or not all(isinstance(col, str) for col in columnsToCheck):
            raise ValueError("Parameter 'columnsToCheck' must be a non-empty list of strings.")
        if not bounds or not all(isinstance(b, tuple) and len(b) == 2 for b in bounds):
            raise ValueError("Parameter 'bounds' must be a non-empty list of (min, max) tuples.")
        if len(columnsToCheck) != len(bounds):
            raise ValueError("Number of columns and bounds must match.")

        if not hasattr(self.dataset, "df"):
            raise AttributeError("The dataset does not contain a 'df' attribute.")
        if not isinstance(self.dataset.df, pd.DataFrame):
            raise TypeError("self.dataset.df must be a pandas DataFrame.")

        out_of_bounds = {}

        for i, column in enumerate(columnsToCheck):
            print(f"\n--- {i + 1}. Checking column '{column}'")
            min_val, max_val = bounds[i]

            if column not in self.dataset.df.columns:
                print(f"⚠️ Warning: Column '{column}' not found in dataset. Skipping.")
                continue

            # Identify out-of-bounds rows
            try:
                out_of_range = self.dataset.df[
                    (self.dataset.df[column] < min_val) |
                    (self.dataset.df[column] > max_val)
                ]
            except Exception as e:
                print(f"❌ Error checking bounds for column '{column}': {e}")
                continue

            if not out_of_range.empty:
                percentage = len(out_of_range) / len(self.dataset.df) * 100
                out_of_bounds[column] = out_of_range
                print(f"Found {len(out_of_range)} values outside bounds [{min_val}, {max_val}]")
                print(f"Percentage: {percentage:.4f}% of data")

                if percentage < 0.5:
                    print("→ Less than 0.5%. Deleting these rows...")
                    try:
                        self.dataset.df.drop(index=out_of_range.index, inplace=True)
                        self.dataset.df.reset_index(drop=True, inplace=True)
                    except Exception as e:
                        print(f"❌ Error deleting rows for column '{column}': {e}")
                else:
                    print("→ More than 0.5%. Keeping them for manual review.")
            else:
                print(f"✅ All values in column '{column}' are within bounds [{min_val}, {max_val}]")

        return out_of_bounds
   
    def compare_distributions_grid(
        self,
        original_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        columns: list[str] | None = None,
        bins: int = 50,
        max_features: int = 20
    ) -> None:
        """
        Side-by-side histograms to compare original vs. cleaned features.

        Parameters
        ----------
        original_df, cleaned_df : pandas.DataFrame
            Pre and post-processing datasets.
        columns : list[str] | None
            Subset of columns to display. Defaults to first *max_features* numeric columns.
        bins : int
            Number of histogram bins.
        max_features : int
            Maximum number of features to plot.

        Returns
        -------
        None
        """

        # --- Validations ---
        if not isinstance(original_df, pd.DataFrame) or not isinstance(cleaned_df, pd.DataFrame):
            raise TypeError("Both original_df and cleaned_df must be pandas DataFrames.")
        if columns is not None and not all(isinstance(c, str) for c in columns):
            raise TypeError("Parameter 'columns' must be a list of strings or None.")
        if not isinstance(bins, int) or bins <= 0:
            raise ValueError("Parameter 'bins' must be a positive integer.")
        if not isinstance(max_features, int) or max_features <= 0:
            raise ValueError("Parameter 'max_features' must be a positive integer.")

        numeric_cols = original_df.select_dtypes(include=np.number).columns.tolist()
        if columns is None:
            columns = numeric_cols[:max_features]
        else:
            columns = [col for col in columns if col in original_df.columns and pd.api.types.is_numeric_dtype(original_df[col])]

        if not columns:
            print("⚠️ No numeric columns to plot.")
            return

        n = len(columns)
        cols = 2
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = axes.flatten()

        for i, col in enumerate(columns):
            try:
                axes[i].hist(original_df[col].dropna(), bins=bins, alpha=0.5, label='Original', color='red')
                axes[i].hist(cleaned_df[col].dropna(), bins=bins, alpha=0.5, label='Cleaned', color='green')
                axes[i].set_title(col)
                axes[i].legend()
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error plotting {col}\n{e}", ha='center')
                axes[i].set_title(col)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def get_outliers_df(
        self,
        detection_type: str = "iqr",
        plot: bool = False,
        threshold: float = 1.5,
        columnsToCheck: list[str] | None = None
    ) -> str:
        """
        Detects outliers, removes them from X_train, and returns a summary.

        Parameters
        ----------
        detection_type : str
            Method used to detect outliers ('iqr' or 'percentile').
        plot : bool
            Whether to show distribution plots of the outlier features.
        threshold : float
            Multiplier for IQR used to define outlier bounds.
        columnsToCheck : list[str] | None
            Columns to check for outliers. If None, uses all numeric columns.

        Returns
        -------
        str
            Summary of the outlier detection operation.
        """

        # --- Validations ---
        if detection_type not in ("iqr", "percentile"):
            raise ValueError("detection_type must be 'iqr' or 'percentile'.")
        if not isinstance(plot, bool):
            raise TypeError("plot must be a boolean.")
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            raise ValueError("threshold must be a positive number.")
        if columnsToCheck is not None:
            if not isinstance(columnsToCheck, list) or not all(isinstance(c, str) for c in columnsToCheck):
                raise TypeError("columnsToCheck must be a list of strings or None.")

        # --- Get numeric columns ---
        if not hasattr(self.dataset, "X_train"):
            raise AttributeError("The dataset is missing 'X_train'.")
        if not isinstance(self.dataset.X_train, pd.DataFrame):
            raise TypeError("self.dataset.X_train must be a pandas DataFrame.")

        only_numerical_features = self.dataset.X_train.select_dtypes(include=["number"]).columns.tolist()
        columns = columnsToCheck if columnsToCheck else only_numerical_features

        outlier_rows = []

        for feature in columns:
            if feature not in self.dataset.X_train.columns:
                print(f"⚠️ Skipping missing column '{feature}'")
                continue

            original_values = self.dataset.X_train[feature]
            if original_values.nunique() < 2:
                continue

            lower_bound, upper_bound = None, None

            if detection_type == "iqr":
                q1 = original_values.quantile(0.25)
                q3 = original_values.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
            elif detection_type == "percentile":
                upper_bound = original_values.quantile(0.99)
                lower_bound = original_values.min()  # Keep lower end untouched

            # Identify outliers
            outlier_mask = (original_values < lower_bound) | (original_values > upper_bound)
            outliersDataset = original_values[outlier_mask]
            outliers_count = outlier_mask.sum()

            if outliers_count > 0:
                outlier_rows.append({
                    "feature": feature,
                    "outlierCount": outliers_count,
                    "percentageOfOutliers": outliers_count / len(original_values) * 100,
                    "descriptiveStatistics": original_values.describe(),
                    "outliersValues": outliersDataset.values
                })

                if plot:
                    plt.title(f"Distribution of '{feature}'")
                    sns.histplot(original_values, kde=True)
                    plt.show()

                if detection_type == "iqr":
                    self.dataset.X_train = self.dataset.X_train[~outlier_mask]
                elif detection_type == "percentile":
                    self.dataset.X_train[feature] = original_values.clip(upper=upper_bound)

        self.dataset.X_train.reset_index(drop=True, inplace=True)

        outlier_df = pd.DataFrame(outlier_rows)

        return (
            f"There are {len(outlier_df)} features with outliers out of "
            f"{len(only_numerical_features)} numerical features "
            f"({len(outlier_df) / len(only_numerical_features) * 100:.2f}%)"
        )
   
    