
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def compute_correlation_matrix(figsize, dataframe):
  # Compute the correlation matrix
  corr = dataframe.corr()
  mask = np.triu(np.ones_like(corr, dtype=bool))
  f, ax = plt.subplots(figsize=figsize)
  cmap = sns.diverging_palette(230, 20, as_cmap=True)
  sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
  
def basic_distribution_plot(x_axis,title, xlabel, ylabel, bins=30, size="small"):
  if size == "s":
    plt.figure(figsize=(5, 3))
  elif size == "m":
    plt.figure(figsize=(10, 6))
  elif size == "l":
    plt.figure(figsize=(20, 15))
  elif size == "xl":
    plt.figure(figsize=(30, 20))
  plt.hist(x_axis, bins=bins, edgecolor='black')
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()