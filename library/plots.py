
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from library.dataset import Dataset




class Plots:
  """ 
  We will be using 'composition' desing pattern to create plots from the dataframe object that is an instance of the Dataset class
  This design pattern allows for two classes to be able to share data (e.g: dataset object)
  """
  def __init__(self, dataset: Dataset) -> None:
    self.dataset = dataset

  def plot_correlation_matrix(self, size: str = "small"):
    """
    Plots the correlation matrix of the dataframe

    Parameters
    ----------
      size : str
        The size of the plot. Taken on ["s", "m", "l", "auto"]
    """
    only_numerical_df = self.dataset.df.select_dtypes(include=["number"])
    corr = only_numerical_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool)) # avoid redundancy
    if size == "s":
      f, ax = plt.subplots(figsize=(5, 3))
    elif size == "m":
      f, ax = plt.subplots(figsize=(10, 6))
    elif size == "l":
      f, ax = plt.subplots(figsize=(20, 15))
    elif size == "auto":
      f, ax = plt.subplots()
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    vmin, vmax = corr.min().min(), corr.max().max()
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
              square=True, linewidths=.5, cbar_kws={"shrink": .8}, vmin=vmin, vmax=vmax)
  def say_hello(self):
    print("Hello, world!")
