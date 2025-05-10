import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from library.phases.phases_implementation.dataset.dataset import Dataset

from library.utils.miscellaneous.save_or_store_plot import save_or_store_plot


class EDA:
  """ 
  We will be using 'composition' desing pattern to create plots from the dataframe object that is an instance of the Dataset class
  This design pattern allows for two classes to be able to share data (e.g: dataset object)
  """
  def __init__(self, dataset: Dataset) -> None:
    self.dataset = dataset
    

  def plot_correlation_matrix(self, size: str = "small", splitted_sets: bool = False, title: str = "", save_plots: bool = False, save_path: str = "", **kwargs) -> None:
    """
    Plots the correlation matrix of the dataframe

    Parameters
    ----------
      size : str
        The size of the plot. Taken on ["s", "m", "l", "auto"]

    Returns
    -------
      None
    """
    if splitted_sets:
      only_numerical_df = self.dataset.X_train.select_dtypes(include=["number"])
      corr = only_numerical_df.corr()
    else:
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
              square=True, linewidths=.5, cbar_kws={"shrink": .8}, vmin=vmin, vmax=vmax,
              xticklabels=corr.columns, yticklabels=corr.index, **kwargs)
    plt.title(f"{title}")
    save_or_store_plot(f, save_plots, save_path + "/feature_selection/manual/multicollinearity", f"{title}.png")

  def plot_categorical_distributions(self, features: list[str], n_cols: int = 2):
    """
    Plots the distribution of categorical features using count plots
    """
    n_rows = len(features)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
      ax_count = axes[i]
      sns.countplot(x=feature, data=self.dataset.df, ax=ax_count)
      ax_count.set_title(f"{feature} - Count Plot")

    for j in range(len(features), len(axes)):
          axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
  
  def count_boxplot_descriptive(self, features: list[str]):
    """
    Plots the count and boxplot of a feature
    """
    n_rows = len(features)
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))

    for i, feature in enumerate(features):
          ax_hist = axes[i, 0]
          ax_box = axes[i, 1]
          ax_text = axes[i, 2]

          self.dataset.df[feature].hist(ax=ax_hist)
          ax_hist.set_title(f"{feature} - Distribution")

          self.dataset.df[feature].plot(kind="box", ax=ax_box)
          ax_box.set_title(f"{feature} - Boxplot")

          # Generate summary statistics text using describe()
          summary_text = self.dataset.df[feature].describe().to_string()

          ax_text.axis('off')
          # Place the text; using a monospaced font helps align the numbers
          ax_text.text(0.5, 0.5, summary_text, ha='center', va='center', fontfamily='monospace', fontsize=10)
          ax_text.set_title(f"{feature} - Summary Statistics")

    plt.tight_layout()
    plt.show()
    
  def lineplot_bivariate(self, features: list[str], target: str, n_cols: int = 3):
      """
      Plots the line plot of a feature against the target with maximized x-axis ticks
      and stretched figure size.
      """
      n_rows = math.ceil(len(features) / n_cols)
      
      fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8 * n_rows))  # Increased figure size
      if n_cols != 1:
        axes = axes.flatten()  # Flatten the array to 1D
      
        for i, feature in enumerate(features):
            sns.lineplot(x=feature, y=target, data=self.dataset.df, ax=axes[i])
            axes[i].set_title(f"{feature} vs {target}")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel(target)
            
            # Increase number of x-axis ticks
            x_values = self.dataset.df[feature]
            n_ticks = min(len(x_values.unique()), 20)  # Cap at 20 ticks to avoid overcrowding
            axes[i].set_xticks(x_values.unique())
            axes[i].tick_params(axis='x', rotation=45)  # Rotate labels for better readability
        
        # Hide any unused subplots
        for j in range(len(features), len(axes)):
            axes[j].set_visible(False)
      else:
        sns.lineplot(x=features[0], y=target, data=self.dataset.df)
        plt.title(f"{features[0]} vs {target}")
        plt.xlabel(features[0])
        plt.ylabel(target)
      
      plt.tight_layout()
      plt.show()

  def scatterplot_bivariate(self, features: list[str], target: str, n_cols: int = 3):
      """
      Plots the line plot of a feature against the target with maximized x-axis ticks
      and stretched figure size.
      """
      n_rows = math.ceil(len(features) / n_cols)
      
      fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8 * n_rows))  # Increased figure size
      if n_cols != 1:
        axes = axes.flatten()  # Flatten the array to 1D
      
        for i, feature in enumerate(features):
            sns.scatterplot(x=feature, y=target, data=self.dataset.df, ax=axes[i])
            axes[i].set_title(f"{feature} vs {target}")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel(target)
            
            # Increase number of x-axis ticks
            x_values = self.dataset.df[feature]
            n_ticks = min(len(x_values.unique()), 20)  # Cap at 20 ticks to avoid overcrowding
            axes[i].set_xticks(x_values.unique())
            axes[i].tick_params(axis='x', rotation=45)  # Rotate labels for better readability
        
        # Hide any unused subplots
        for j in range(len(features), len(axes)):
            axes[j].set_visible(False)
      else:
        sns.scatterplot(x=features[0], y=target, data=self.dataset.df)
        plt.title(f"{features[0]} vs {target}")
        plt.xlabel(features[0])
        plt.ylabel(target)
      
      plt.tight_layout()
      plt.show()

  def barplot_bivariate(self, features: list[str], target: str, n_cols: int = 3):
      """
      Plots the bar plot of a feature against the target with maximized x-axis ticks
      and stretched figure size.
      """
      n_rows = math.ceil(len(features) / n_cols)
      
      fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8 * n_rows))  # Increased figure size
      if n_cols != 1:
        axes = axes.flatten()  # Flatten the array to 1D
      
        for i, feature in enumerate(features):
            # Convert interval data to strings if needed
            if pd.api.types.is_interval_dtype(self.dataset.df[feature]):
                x_values = self.dataset.df[feature].astype(str)
            else:
                x_values = self.dataset.df[feature]
                
            sns.barplot(x=x_values, y=target, data=self.dataset.df, ax=axes[i])
            axes[i].set_title(f"{feature} vs {target}")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel(target)
            
            # Increase number of x-axis ticks
            n_ticks = min(len(x_values.unique()), 20)  # Cap at 20 ticks to avoid overcrowding
            axes[i].set_xticks(range(len(x_values.unique())))
            axes[i].set_xticklabels(x_values.unique())
            axes[i].tick_params(axis='x', rotation=45)  # Rotate labels for better readability
        
        # Hide any unused subplots
        for j in range(len(features), len(axes)):
            axes[j].set_visible(False)
      else:
        # Convert interval data to strings if needed
        if pd.api.types.is_interval_dtype(self.dataset.df[features[0]]):
            x_values = self.dataset.df[features[0]].astype(str)
        else:
            x_values = self.dataset.df[features[0]]
            
        sns.barplot(x=x_values, y=target, data=self.dataset.df)
        plt.title(f"{features[0]} vs {target}")
        plt.xlabel(features[0])
        plt.ylabel(target)
      
      plt.tight_layout()
      plt.show()
  