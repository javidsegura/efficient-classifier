from library.phases.dataset.dataset import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

class ClassImbalance:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
      
    def class_imbalance(self, plot: bool = False) -> str:
        """
        Balances classes via SMOTE and optionally plots the distributions
        before and after resampling.

        Parameters:
        -----------
        plot : bool
            Whether to show barplots of class counts before/after SMOTE

        Returns:
        --------
        str
            Summary of the balancing operation
        """
        # 1. Record original counts
        counts_before = self.dataset.y_train.value_counts().sort_index()
        self.imbalance_ratio = counts_before.min() / counts_before.max()

        # 2. Optionally plot before
        if plot:
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

        # 3. Apply SMOTE
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(
            self.dataset.X_train, 
            self.dataset.y_train
        )
        self.dataset.X_train, self.dataset.y_train = X_res, y_res

        # 4. Record new counts and plot
        counts_after = self.dataset.y_train.value_counts().sort_index()
        if plot:
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

        return (
            f"Successfully balanced classes via SMOTE. "
            f"Started with a {self.imbalance_ratio:.2f}:1 ratio; now 1:1."
        )
    

  