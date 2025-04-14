from abc import ABC, abstractmethod
from library.phases.dataset.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt

class BaseStrategy(ABC):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    @abstractmethod
    def transform_target(self, plot: bool = False):
        pass
    
    @abstractmethod
    def inverse_transform_target(self):
        pass

class LogStrategy(BaseStrategy):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset)

    def transform_target(self, plot: bool = False):
        if plot:
            fig, ax = plt.subplots(2, 3, figsize=(15, 8))
            plt.tight_layout(h_pad=2, w_pad=3)  # Add padding between subplots
            
            # Before transformation plots
            ax[0, 0].hist(self.dataset.y_train, bins=100, edgecolor='#1f77b4', color='#1f77b4', alpha=0.7)
            ax[0, 0].set_title('Distribution of Target Variable (Train)')
            ax[0, 0].set_xlabel('Target Value')
            ax[0, 0].set_ylabel('Frequency')
            
            ax[0, 1].hist(self.dataset.y_val, bins=100, edgecolor='#1f77b4', color='#1f77b4', alpha=0.7)
            ax[0, 1].set_title('Distribution of Target Variable (Validation)')
            ax[0, 1].set_xlabel('Target Value')
            ax[0, 1].set_ylabel('Frequency')
            
            ax[0, 2].hist(self.dataset.y_test, bins=100, edgecolor='#1f77b4', color='#1f77b4', alpha=0.7)
            ax[0, 2].set_title('Distribution of Target Variable (Test)')
            ax[0, 2].set_xlabel('Target Value')
            ax[0, 2].set_ylabel('Frequency')

        # Apply log transformation
        self.dataset.y_train = np.log(self.dataset.y_train)
        self.dataset.y_val = np.log(self.dataset.y_val)
        self.dataset.y_test = np.log(self.dataset.y_test)

        if plot:
            ax[1, 0].hist(self.dataset.y_train, bins=100, edgecolor='#2ca02c', color='#2ca02c', alpha=0.7)
            ax[1, 0].set_title('Log-Transformed Distribution (Train)')
            ax[1, 0].set_xlabel('Log(Target Value)')
            ax[1, 0].set_ylabel('Frequency')
            
            ax[1, 1].hist(self.dataset.y_val, bins=100, edgecolor='#2ca02c', color='#2ca02c', alpha=0.7)
            ax[1, 1].set_title('Log-Transformed Distribution (Validation)')
            ax[1, 1].set_xlabel('Log(Target Value)')
            ax[1, 1].set_ylabel('Frequency')
            
            ax[1, 2].hist(self.dataset.y_test, bins=100, edgecolor='#2ca02c', color='#2ca02c', alpha=0.7)
            ax[1, 2].set_title('Log-Transformed Distribution (Test)')
            ax[1, 2].set_xlabel('Log(Target Value)')
            ax[1, 2].set_ylabel('Frequency')
            
            plt.show() 

    def inverse_transform_target(self):
        self.dataset.y_train = np.exp(self.dataset.y_train)
        self.dataset.y_val = np.exp(self.dataset.y_val)
        self.dataset.y_test = np.exp(self.dataset.y_test)
