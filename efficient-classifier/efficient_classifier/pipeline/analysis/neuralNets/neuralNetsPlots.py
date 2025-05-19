import matplotlib.pyplot as plt
import math

from efficient_classifier.utils.miscellaneous.save_or_store_plot import save_or_store_plot

class NeuralNetsPlots:
      def __init__(self, model_sklearn: object):
            self.model_sklearn = model_sklearn
            self.history = None

      def plot_per_epoch_progress(self, metrics: list[str], phase: str, n_cols: int = 2, save_plots: bool = False, save_path: str = None) -> None:
            """
            Plots the progress of the feedforward NN per epoch. 

            Parameters:
            ----------
            metrics: list[str]
                  The metrics to plot.
            phase: str
                  The phase to plot.
            n_cols: int
                  The number of columns to plot.
            save_plots: bool
                  Whether to save the plots.
            save_path: str
                  The path to save the plots.

            Returns:
            -------
            None
            """
            if self.history is None:
                  self.history = self.model_sklearn.history.history
            assert all(metric in self.history.keys() for metric in metrics), f"Metric must be in {self.history.keys()}"
            if "f1_score" in metrics:
                  raise NotImplementedError("F1 score is not implemented yet. ") # Currently having class-wide analysis
            
            n_rows = math.ceil(len(metrics) / n_cols)
            print(f"There are {n_rows} rows and {n_cols} columns")
            
            fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 5))
            ax = ax.flatten()
            for i, metric in enumerate(metrics):
                  training_metric = self.history[metric]
                  validation_metric = self.history[f"val_{metric}"]
                  epochs = range(1, len(training_metric) + 1)
                  ax[i].plot(epochs, training_metric, 'bo-', label='Training')
                  ax[i].plot(epochs, validation_metric, 'ro-', label='Validation')
                  ax[i].set_title(f'{metric} per epoch')
                  ax[i].set_xlabel('Epochs')
                  ax[i].set_ylabel(metric)
                  ax[i].grid(True)
                  ax[i].legend()
            for i in range(len(metrics), len(ax)):
                  fig.delaxes(ax[i])
            save_or_store_plot(fig, save_plots, directory_path=save_path + f"/{phase}/model_performance", filename=f"per_epoch_progress_{phase}.png")