import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import RobustScaler

def analyze_scaled_feature_variances(pipeline_manager, save_path=None):
    """
    Analyzes feature variances after scaling to identify potential threshold for feature selection.
    
    Parameters:
    ----------
    pipeline_manager : PipelineManager
        The pipeline manager containing the pipelines
    save_path : str, optional
        Path to save the plots
    
    Returns:
    -------
    pd.Series
        Sorted feature variances for further analysis
    """
    # Choose a reference pipeline for analysis
    reference_pipeline = pipeline_manager.pipelines["not_baseline"]["tree_based"]
    X_train = reference_pipeline.dataset.X_train.copy()
    
    # Apply RobustScaler (same as in your preprocessing pipeline)
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns
    )
    
    # Calculate variances
    feature_variances = X_scaled.var().sort_values(ascending=False)
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    
    # 1. Histogram of variances
    axes[0].hist(feature_variances, bins=50, edgecolor='black')
    axes[0].set_title('Distribution of Feature Variances after RobustScaling')
    axes[0].set_xlabel('Variance')
    axes[0].set_ylabel('Frequency')
    
    # Add vertical lines at potential threshold points
    for threshold in [0.01, 0.05, 0.1]:
        axes[0].axvline(x=threshold, color='r', linestyle='--', 
                      label=f'Threshold = {threshold}')
    axes[0].legend()
    
    # 2. Sorted bar plot of variances (helps identify the elbow)
    indices = np.arange(len(feature_variances))
    axes[1].bar(indices, feature_variances.values)
    axes[1].set_title('Sorted Feature Variances after RobustScaling')
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('Variance')
    axes[1].set_yscale('log')  # Log scale helps visualize the elbow better
    
    # Add annotation for potential elbow points
    diff = np.diff(feature_variances.values)
    elbow_candidates = np.argsort(diff)[:3]  # Top 3 largest drops
    
    for idx in elbow_candidates:
        if idx > 5:  # Only mark elbows after a few features
            axes[1].annotate(f'Potential elbow: {feature_variances.iloc[idx]:.4f}',
                           xy=(idx, feature_variances.iloc[idx]),
                           xytext=(idx+10, feature_variances.iloc[idx]*2),
                           arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        # Create directory path that matches your project structure
        import os
        plot_dir = os.path.join(save_path, "feature_analysis", "variance_analysis")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Save the plot
        plt.savefig(os.path.join(plot_dir, "feature_variance_analysis.png"))
        print(f"Feature variance analysis saved to {plot_dir}/feature_variance_analysis.png")
    else:
        plt.show()
    
    # Return data for threshold selection
    low_var_counts = {
        0.001: sum(feature_variances < 0.001),
        0.01: sum(feature_variances < 0.01),
        0.05: sum(feature_variances < 0.05),
        0.1: sum(feature_variances < 0.1)
    }
    
    print("Feature count below variance thresholds:")
    for threshold, count in low_var_counts.items():
        print(f"Threshold {threshold}: {count} features ({count/len(feature_variances)*100:.2f}%)")
    
    return feature_variances