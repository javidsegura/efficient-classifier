

import matplotlib.pyplot as plt
import os
def save_or_store_plot(fig, save_plots: bool, directory_path: str, filename: str):
      if save_plots and directory_path and filename:
            if not os.path.exists(directory_path):
                  os.makedirs(directory_path)
            fig.savefig(os.path.join(directory_path, filename))
            plt.close(fig)
      else:
            plt.show()