from library.phases.phases_implementation.dataset.dataset import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import LabelEncoder
from library.utils.miscellaneous.save_or_store_plot import save_or_store_plot
class ClassImbalance:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
      
    
    def class_imbalance(self, method: str = "SMOTE", save_plots: bool = False, save_path: str = None) -> str:
        """
        Balances classes via SMOTE and optionally plots the distributions
        before and after resampling.

        Parameters
        ----------
        method : str
            The method to use for balancing classes.
        save_plots : bool
            Whether to save plots of class counts before/after SMOTE
        save_path : str
            The path to save the plots

        Returns
        -------
        str
            Summary of the balancing operation
        """

        # --- Input validation ---
        if not isinstance(save_plots, bool):
            raise TypeError("Parameter 'save_plots' must be a boolean.")

        # --- Attribute checks ---
        for attr in ['X_train', 'y_train']:
            if not hasattr(self.dataset, attr):
                raise AttributeError(f"The dataset is missing the attribute '{attr}'.")

        try:
            counts_before = self.dataset.y_train.value_counts().sort_index()
        except Exception as e:
            raise RuntimeError(f"Could not compute class counts: {e}")

        if counts_before.empty or len(counts_before) < 2:
            raise ValueError("SMOTE requires at least two classes with non-zero samples.")

        try:
            self.imbalance_ratio = counts_before.min() / counts_before.max()
        except ZeroDivisionError:
            raise ValueError("Class count contains zero, cannot compute imbalance ratio.")

        # --- Plot before resampling ---
        fig, ax = plt.subplots(figsize=(6, 4), nrows=1, ncols=2)
        ax = ax.flatten()
        if save_plots:
            try:
                sns.barplot(
                    x=counts_before.index.astype(str),
                    y=counts_before.values,
                    ax=ax[0]
                )
                ax[0].set_title(f"Before {method} (imbalance ratio {self.imbalance_ratio:.2f}:1)")
                ax[0].set_xlabel("Class")
                ax[0].set_ylabel("Count")
                ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha="right")

            except Exception as e:
                raise RuntimeError(f"An error occurred while plotting pre-SMOTE: {e}")

        # --- Apply SMOTE ---
        try:
            if method == "SMOTE":
                smote = SMOTE(random_state=42)
            elif method == "ADASYN":
                smote = ADASYN(random_state=42)
            X_res, y_res = smote.fit_resample(self.dataset.X_train, self.dataset.y_train)
            self.dataset.X_train = X_res
            self.dataset.y_train = y_res
        except Exception as e:
            raise RuntimeError(f"An error occurred during SMOTE resampling: {e}")

        # --- Plot after resampling ---
        if save_plots:
            try:
                counts_after = self.dataset.y_train.value_counts().sort_index()
                sns.barplot(
                    x=counts_after.index.astype(str),
                    y=counts_after.values,
                    ax=ax[1]
                )
                ax[1].set_title(f"After {method} (balanced 1:1)")
                ax[1].set_xlabel("Class")
                ax[1].set_ylabel("Count")
                ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha="right")
                plt.tight_layout(w_pad=3)
                save_or_store_plot(fig, save_plots, save_path + "/class_imbalance", f"class_imbalance.png")
            except Exception as e:
                raise RuntimeError(f"An error occurred while plotting post-SMOTE: {e}")

        return (
            f"Successfully balanced classes via SMOTE. "
            f"Started with a {self.imbalance_ratio:.2f}:1 ratio; now 1:1."
        )

  