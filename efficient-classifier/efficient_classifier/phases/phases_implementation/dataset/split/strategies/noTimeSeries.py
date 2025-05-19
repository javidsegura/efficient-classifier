import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates

from sklearn.model_selection import train_test_split

import seaborn as sns
import numpy as np
import pandas as pd
import os

from efficient_classifier.phases.phases_implementation.dataset.split.strategies.base import Split
from efficient_classifier.utils.miscellaneous.save_or_store_plot import save_or_store_plot


class NoTimeSeries(Split):
    def __init__(self, dataset) -> None:
        super().__init__(dataset)

    def split_data(
        self,
        y_column: str,
        otherColumnsToDrop: list[str] = [],
        train_size: float = 0.8,
        validation_size: float = 0.1,
        test_size: float = 0.1,
        save_plots: bool = False,
        save_path: str = None,
    ) -> None:
        """
        Splits the dataframe into training, validation and test sets

        Parameters
        ----------
        y_column : str
              The column name of the target variable
        otherColumnsToDrop : list[str]
              The columns to drop from the dataframe (e.g: record identifiers)
        train_size : float
              The size of the training set
        validation_size : float
              The size of the validation set
        test_size : float
              The size of the test set
        plot_distribution : bool
              Whether to plot the distribution of the features
        Returns
        -------
        X_train : pd.DataFrame
              The training set
        X_val : pd.DataFrame
              The validation set
        X_test : pd.DataFrame
              The test set
        y_train : pd.Series
              The training set
        y_val : pd.Series
              The validation set
        y_test : pd.Series
              The test set
        """
        X, y = self.__get_X_y__(y_column, otherColumnsToDrop)
        assert train_size + validation_size + test_size == 1, (
            "The sum of the sizes must be 1"
        )
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=validation_size + test_size,
            random_state=self.dataset.random_state,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=test_size / (validation_size + test_size),
            random_state=self.dataset.random_state,
        )
        print(f"X_train: {X_train.shape}")
        print(f"X_val: {X_val.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_val: {y_val.shape}")
        print(f"y_test: {y_test.shape}")
        (
            self.dataset.X_train,
            self.dataset.X_val,
            self.dataset.X_test,
            self.dataset.y_train,
            self.dataset.y_val,
            self.dataset.y_test,
        ) = X_train, X_val, X_test, y_train, y_val, y_test
        if save_plots:
            super().plot_per_set_distribution(X.columns, save_plots, save_path)
        return self.dataset.X_train.head().to_dict()

    def asses_split_classifier(
        self,
        p: float,
        step: float,
        upper_bound: float = 0.50,
        save_plots: bool = False,
        save_path: str = None,
    ) -> pd.DataFrame:
        """
        Assesses the split of the dataframe for classification tasks.

        Parameters
        ----------
        p : float
              The percentage of the dataframe to split
        step : float
              The step size for the split
        upper_bound : float
              The upper bound for the split
        plot : bool
              If True, the split assessment will be plotted
        Returns
        -------
        df_split_assesment : pd.DataFrame
              A dataframe with the split assessment
        """

        if self.dataset.modelTask != "classification":
            raise ValueError("The model task must be classification")

        computeSE = lambda p, n: np.sqrt((p * (1 - p)) / n)
        df_split_assesment = pd.DataFrame()
        hold_out_size = step
        priorSE = 0
        differenceToPriorSE_percentage = 0
        while hold_out_size <= upper_bound:
            assert hold_out_size < 1
            train_size_percentage = 1 - hold_out_size
            train_size_count = round(
                self.dataset.df.shape[0] * train_size_percentage, 0
            )

            val_size_percentage = hold_out_size / 2
            val_size_count = round(self.dataset.df.shape[0] * (hold_out_size / 2), 0)

            test_size_percentage = hold_out_size / 2
            test_size_count = round(self.dataset.df.shape[0] * (hold_out_size / 2), 0)

            currentSE = computeSE(p, test_size_count)
            differenceToPriorSE = currentSE - priorSE
            # Avoid division by zero
            if priorSE != 0:
                differenceToPriorSE_percentage = (currentSE - priorSE) / priorSE
            priorSE = currentSE

            new_row = pd.DataFrame(
                [
                    {
                        "train_size (%)": train_size_percentage,
                        "train_size_count": train_size_count,
                        "validation_size (%)": val_size_percentage,
                        "validation_size_count": val_size_count,
                        "test_size (%)": test_size_percentage,
                        "test_size_coount": test_size_count,
                        "currentSE": currentSE,
                        "differenceToPriorSE": differenceToPriorSE,
                        "differenceToPriorSE (%)": differenceToPriorSE_percentage,
                    }
                ]
            )

            # Concatenate the new row with your existing DataFrame
            df_split_assesment = pd.concat(
                [df_split_assesment, new_row], ignore_index=True
            )
            hold_out_size += step
        if save_plots:
            fig, ax1 = plt.subplots()

            color = "tab:blue"
            ax1.set_xlabel("Training Set Percentage")
            ax1.set_ylabel("Current SE", color=color)
            ax1.plot(
                df_split_assesment["train_size (%)"],
                df_split_assesment["currentSE"],
                marker="o",
                color=color,
            )
            ax1.tick_params(axis="y", labelcolor=color)

            ax1.xaxis.set_major_locator(MultipleLocator(0.05))

            ax2 = ax1.twinx()
            color = "tab:red"
            ax2.set_ylabel("Difference to Prior SE (%)", color=color)
            ax2.plot(
                df_split_assesment["train_size (%)"][1:],
                df_split_assesment["differenceToPriorSE (%)"][1:],
                marker="x",
                linestyle="--",
                color=color,
            )
            ax2.tick_params(axis="y", labelcolor=color)

            plt.title("Holdout Split Trade-Off: Training Set vs SE")
            save_or_store_plot(
                fig,
                save_plots,
                save_path + "/split/split_trade_off",
                "split_trade_off.png",
            )

        self.df_split_assesment = df_split_assesment
        return df_split_assesment