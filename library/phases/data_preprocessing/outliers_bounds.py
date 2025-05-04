from library.phases.data_preprocessing.bounds_config import BOUNDS
from library.phases.dataset.dataset import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class OutliersBounds:
    def __init__(self, dataset: Dataset) -> None:
      self.dataset = dataset
     
    def bound_checking(self):
      """Apply numeric *BOUNDS* to *dataset.df* and remove rare violators.

      The global constant :data:`BOUNDS` must map column names to
      ``(min, max)`` tuples. For each column the helper will
      * drop rows that lie outside the interval **when** they represent
        < 0.5 % of the total dataset, or
      * keep (but record) them for manual analysis otherwise.

      Returns
      -------
      None
      """
      self.bound_cols, self.bound_limits = zip(*BOUNDS.items())
      self.outliers_dict = self._bound_checking_helper(
          columnsToCheck=list(self.bound_cols),
          bounds=list(self.bound_limits)
      )
      return None
    
    def _bound_checking_helper(self, columnsToCheck: list[str] = [], bounds: list[tuple] = []):
      """
      Low-level helper that implements the actual bound filtering.

      Parameters
      ----------
      columnsToCheck : list[str]
          Column names to validate.
      bounds : list[tuple[float, float]]
          Sequence of *(min, max)* intervals for each *columnsToCheck* entry.

      Returns
      -------
      dict[str, pd.DataFrame]
          Mapping of column name ➟ offending rows (if any).
      """
      assert len(columnsToCheck) > 0, "Columns to check must be provided"
      assert len(bounds) > 0, "Bounds must be provided"
      assert len(columnsToCheck) == len(bounds), "Number of columns and bounds must match"
      
      out_of_bounds = {}
      
      for i, column in enumerate(columnsToCheck):
          print(f"\n--- {i + 1}. Checking column {column}")
          min_val, max_val = bounds[i]
          
          # Check if column exists in the dataset
          if column not in self.dataset.df.columns:
              print(f"Warning: Column '{column}' not found in dataset")
              continue
          
          # Find values outside the bounds
          out_of_range = self.dataset.df[(self.dataset.df[column] < min_val) | 
                                        (self.dataset.df[column] > max_val)]
          
          if len(out_of_range) > 0:
              percentage = len(out_of_range) / len(self.dataset.df) * 100
              out_of_bounds[column] = out_of_range
              print(f"Found {len(out_of_range)} values outside bounds [{min_val}, {max_val}]")
              print(f"Percentage: {percentage:.4f}% of data")
              
              if percentage < 0.5:
                  print(f"→ Less than 0.5%. Deleting these rows...")
                  self.dataset.df = self.dataset.df.drop(out_of_range.index)
                  self.dataset.df.reset_index(drop=True, inplace=True)
              else:
                  print(f"→ More than 0.5%. Keeping them for manual review.")
          else:
              print(f"All values in column '{column}' are within bounds [{min_val}, {max_val}]")
      
      return out_of_bounds
 
    def smart_outlier_handler(self,
          feature_groups: dict[str, list[str]] = None,
          iqr_k: float = 1.5,
          upper_clip: float = 0.995,
          log_feats: list[str] | None = None) -> None:
        """Hybrid outlier removal / winsorisation that *never* alters zeros.

        Parameters
        ----------
        feature_groups : dict[str, list[str]] | None, optional
            Logical grouping of features, e.g. ``{"Memory": ["Mem_RSS", …]}``.
            When *None*, every numeric column is processed independently.
        iqr_k : float, default 1.5
            Whisker length multiplier for Tukey’s IQR rule.
        upper_clip : float, default 0.995
            Percentile to clip extremely heavy-tailed distributions.
        log_feats : list[str] | None, optional
            Columns that benefit from a log1p-transform before IQR bounds.

        Returns
        -------
        None

        Notes
        -----
        * Always keeps the mass at *0* (lower bound).
        * Switches dynamically between IQR and percentile strategies based on
          cardinality.
        """
        original_df = self.dataset.df.copy()
        log_feats = log_feats or []
        
        # If no feature_groups provided, treat all columns as one group
        if not feature_groups:
          feature_groups = {'all_features': self.dataset.df.columns.tolist()}

        # --- helpers -------------------------------------------------------------
        def iqr_bounds(s):
            q1, q3 = s.quantile([.25, .75])
            iqr = q3 - q1
            return q1 - iqr_k * iqr, q3 + iqr_k * iqr

        # --- main loop -----------------------------------------------------------
        # --- main loop -----------------------------------------------------------
        for g, cols in feature_groups.items():
            for col in cols:
                if col not in self.dataset.df.columns:            # silent skip
                    continue

                series = self.dataset.df[col]

                if not np.issubdtype(series.dtype, np.number):
                    continue  # <<< 🚀 add this line to skip non-numeric columns

                # choose strategy
                if col in log_feats:
                    series_log = np.log1p(series)          # keeps zeros at 0
                    lb, ub = iqr_bounds(series_log)
                    ub = np.expm1(ub)                      # back-transform
                    lb = 0                                 # *never* drop / cap zeros
                else:
                    # decide between IQR or percentile based on cardinality
                    if series.nunique() > 1e4:
                        lb, ub = 0, series.quantile(upper_clip)
                    else:
                        lb_tmp, ub_tmp = iqr_bounds(series)
                        lb, ub = 0, ub_tmp                 # protect zeros

                mask_hi = series > ub
                mask_lo = series < lb                      # will always be False (lb==0)

                # drop if very few, otherwise cap
                perc_hi = mask_hi.mean() * 100
                if perc_hi < 0.5:
                    self.dataset.df = self.dataset.df.loc[~mask_hi]
                else:
                    self.dataset.df.loc[mask_hi, col] = ub

                
        self.compare_distributions_grid(original_df, self.dataset.df)

        return None

    def compare_distributions_grid(self, 
        original_df: pd.DataFrame, 
        cleaned_df: pd.DataFrame,
        columns: list[str] | None = None, 
        bins: int = 50, 
        max_features: int = 20):
        """
        Side-by-side histograms to compare original vs. cleaned features.

        Parameters
        ----------
        original_df, cleaned_df : pandas.DataFrame
            Pre and post-processing datasets.
        columns : list[str] | None, optional
            Subset of columns to display. Defaults to the first *max_features*
            numeric ones.
        bins : int, default 50
            Number of histogram bins.
        max_features : int, default 20
            Hard cap to keep the plot grid manageable.

        Returns
        -------
        None
        """
        numeric_cols = original_df.select_dtypes(include=np.number).columns.tolist()
        
        if columns is None:
            columns = numeric_cols[:max_features]

        n = len(columns)
        cols = 2
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = axes.flatten()

        for i, col in enumerate(columns):
            axes[i].hist(original_df[col], bins=bins, alpha=0.5, label='Original', color='red')
            axes[i].hist(cleaned_df[col], bins=bins, alpha=0.5, label='Cleaned', color='green')
            axes[i].set_title(col)
            axes[i].legend()

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])  # remove unused subplots

        plt.tight_layout()
        plt.show()

    def get_outliers_df(self, detection_type: str = "iqr", plot: bool = False, threshold: float = 1.5, columnsToCheck: list[str] | None = None) -> str:
        """
        Detects outliers, removes them from X_train, and returns a DataFrame with outlier statistics.

        Parameters:
        -----------
        pipeline: str
            To determine how to treat the outliers based on the pipeline
        plot : bool
            Whether to plot the outliers
        threshold : float
            Multiplier for the IQR to determine outliers
        columnsToCheck : list[str]
            Specific columns to check for outliers. If empty, all numerical columns are used.

        Returns:
        --------
        str
            Summary of the outlier detection operation 
        """
        outlier_rows = []
        only_numerical_features = self.dataset.X_train.select_dtypes(include=["number"]).columns
        outliers = {}

        for feature in only_numerical_features if not columnsToCheck else columnsToCheck:
            original_values = self.dataset.X_train[feature]
            original_values_size = len(original_values)
            IQR = original_values.quantile(0.75) - original_values.quantile(0.25)
            lower_bound = original_values.quantile(0.25) - threshold * IQR
            upper_bound = original_values.quantile(0.75) + threshold * IQR
            outlier_mask = (original_values < lower_bound) | (original_values > upper_bound)
            outliersDataset = original_values[outlier_mask]
            outliers_count = outlier_mask.sum()

            if outliers_count > 0:
                outliers[feature] = outliersDataset
                outlier_rows.append({
                    "feature": feature,
                    "outlierCount": outliers_count,
                    "percentageOfOutliers": outliers_count / original_values_size * 100,
                    "descriptiveStatistics": original_values.describe(),
                    "outliersValues": outliersDataset.values
                })

                if plot:
                    plt.title(f"Distribution of '{feature}'")
                    sns.histplot(original_values, kde=True)
                    plt.show()

                if detection_type == "iqr":
                  # Remove outliers from X_train
                  self.dataset.X_train = self.dataset.X_train[~outlier_mask]
                elif detection_type == "percentile":
                  # Clip outliers on 99th percentile
                  p99 = original_values.quantile(0.99)
                  self.dataset.X_train[feature] = original_values.clip(upper=p99)
                else:
                  assert("Error: You must introduce a correct value for pipeline. Only 'iqr' and 'percentile' are accepted.")

        # Reset index after removing rows
        self.dataset.X_train.reset_index(drop=True, inplace=True)

        outlier_df = pd.DataFrame(outlier_rows)

        return f"There are {len(outlier_df)} features with outliers out of {len(only_numerical_features)} numerical features ({len(outlier_df) / len(only_numerical_features) * 100:.2f}%)"
   
    