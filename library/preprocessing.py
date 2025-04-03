import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats

class Preprocessing:
    def __init__(self):
        pass
    
    # Function to determine if a feature needs standardization or normalization
    def determine_scaling_method(feature_series) -> str:
        """
        Determine whether a feature should be standardized or normalized based on its characteristics.
        
        Args:
            feature_series: Pandas Series containing the feature values
            
        Returns:
            str: 'robust', 'normalize', or 'none'
        """
        # Skip if all values are the same (zero variance)
        if feature_series.std() == 0:
            return 'none'
        
        # Check for outliers using IQR method
        Q1 = feature_series.quantile(0.25)
        Q3 = feature_series.quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold = 1.5 * IQR
        has_outliers = ((feature_series < (Q1 - outlier_threshold)) | 
                        (feature_series > (Q3 + outlier_threshold))).any()
        
        # Check for normality using skewness
        skewness = abs(stats.skew(feature_series.dropna()))
        is_skewed = skewness > 1.0  # Threshold for significant skewness
        
        # Decision logic
        if has_outliers:
            return 'robust'  # Standardization handles outliers better
        elif is_skewed:
            return 'normalize'    # Normalization is better for non-normal distributions
        else:
            return 'robust'  # Default to standardization for most ML algorithms
        
    # Visualize the effect of scaling on a few features
    def plot_before_after_scaling(original_df, scaled_df, features, n_features=4) -> None:
        """
        Plot histograms before and after scaling for selected features.
        
        Args:
            original_df: DataFrame containing the original data
            scaled_df: DataFrame containing the scaled data
            features: List of feature names to plot
            
        Returns:
            None
        """
        # Select a subset of features if there are too many
        if len(features) > n_features:
            features = np.random.choice(features, n_features, replace=False)
        
        fig, axes = plt.subplots(len(features), 2, figsize=(12, 3*len(features)))
        
        for i, feature in enumerate(features):
            # Original distribution
            sns.histplot(original_df[feature], kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f'Original: {feature}')
            
            # Scaled distribution
            sns.histplot(scaled_df[feature], kde=True, ax=axes[i, 1])
            axes[i, 1].set_title(f'Scaled: {feature}')
        
        plt.tight_layout()
        plt.show()