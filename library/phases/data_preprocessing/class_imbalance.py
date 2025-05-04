from library.phases.dataset.dataset import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

class ClassImbalance:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
      
    
    def class_imbalance(self, plot: bool = False) -> str:
        """
        Balances classes via SMOTE and optionally plots the distributions
        before and after resampling.

        Parameters
        ----------
        plot : bool
            Whether to show barplots of class counts before/after SMOTE

        Returns
        -------
        str
            Summary of the balancing operation
        """

        # --- Input validation ---
        if not isinstance(plot, bool):
            raise TypeError("Parameter 'plot' must be a boolean.")

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
        if plot:
            try:
                plt.figure(figsize=(6, 4))
                sns.barplot(
                    x=counts_before.index.astype(str),
                    y=counts_before.values
                )
                plt.title(f"Before SMOTE (imbalance ratio {self.imbalance_ratio:.2f}:1)")
                plt.xlabel("Class")
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                raise RuntimeError(f"An error occurred while plotting pre-SMOTE: {e}")

        # --- Encode non-numeric features ---
        try:
            X = self.dataset.X_train.copy()
            for col in X.select_dtypes(include='object').columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        except Exception as e:
            raise RuntimeError(f"An error occurred during encoding of non-numeric features: {e}")

        # --- Apply SMOTE ---
        try:
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, self.dataset.y_train)
            self.dataset.X_train = X_res
            self.dataset.y_train = y_res
        except Exception as e:
            raise RuntimeError(f"An error occurred during SMOTE resampling: {e}")

        # --- Plot after resampling ---
        if plot:
            try:
                counts_after = self.dataset.y_train.value_counts().sort_index()
                plt.figure(figsize=(6, 4))
                sns.barplot(
                    x=counts_after.index.astype(str),
                    y=counts_after.values
                )
                plt.title("After SMOTE (balanced 1:1)")
                plt.xlabel("Class")
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                raise RuntimeError(f"An error occurred while plotting post-SMOTE: {e}")

        return (
            f"Successfully balanced classes via SMOTE. "
            f"Started with a {self.imbalance_ratio:.2f}:1 ratio; now 1:1."
        )

  