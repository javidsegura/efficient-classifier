import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

class FeatureClustering:
    def __init__(self, dataset):
        self.dataset = dataset
        
    def cluster_features(self, method="kmeans", n_clusters=None, correlation_threshold=0.7, 
                         use_representatives=True, save_plots=False, save_path=""):
        """
        Cluster features and optionally replace them with representatives
        
        Parameters:
        -----------
        method : str, default="kmeans"
            Clustering method: "kmeans", "hierarchical", or "correlation"
        n_clusters : int, default=None
            Number of clusters to form. If None, it will be estimated.
        correlation_threshold : float, default=0.7
            Correlation threshold for the correlation-based clustering
        use_representatives : bool, default=True
            If True, replace clusters with representative features
        save_plots : bool, default=False
            Whether to save clustering visualization plots
        save_path : str, default=""
            Path to save plots
            
        Returns:
        --------
        cluster_info : dict
            Dictionary containing clustering information
        """
        if self.dataset.X_train is None:
            raise ValueError("Dataset must be split before clustering features")
            
        X_train = self.dataset.X_train.copy()
        
        # Standardize features for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # Transpose to cluster features instead of samples
        X_features = X_scaled.T
        
        clusters = None
        
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(X_features, max_clusters=min(20, X_features.shape[0]//2))
        
        # Apply the selected clustering method
        if method.lower() == "kmeans":
            clusters = self._kmeans_clustering(X_features, n_clusters)
        elif method.lower() == "hierarchical":
            clusters = self._hierarchical_clustering(X_features, n_clusters)
        elif method.lower() == "correlation":
            clusters = self._correlation_clustering(X_train, correlation_threshold)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Generate cluster information dictionary
        cluster_info = self._generate_cluster_info(X_train, clusters)
        
        # Optional: save visualization of clusters
        if save_plots:
            self._plot_feature_clusters(X_features, clusters, method, save_path)
        
        # Replace clusters with representatives if requested
        if use_representatives:
            self._replace_with_representatives(cluster_info)
        
        return cluster_info
    
    def _find_optimal_clusters(self, X, max_clusters=10):
        """Find optimal number of clusters using silhouette score"""
        best_score = -1
        best_n = 2  # Default to minimum of 2 clusters
        
        for n in range(2, min(max_clusters + 1, X.shape[0])):
            kmeans = KMeans(n_clusters=n, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Skip if we have any single-element clusters
            if len(np.unique(labels, return_counts=True)[1]) != n:
                continue
                
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_n = n
                
        return best_n
    
    def _kmeans_clustering(self, X_features, n_clusters):
        """Apply KMeans clustering to features"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(X_features)
    
    def _hierarchical_clustering(self, X_features, n_clusters):
        """Apply hierarchical clustering to features"""
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        return clustering.fit_predict(X_features)
    
    def _correlation_clustering(self, X_features, correlation_threshold):
        """
        Clusters features based on correlation.
        
        Parameters
        ----------
        X_features : pandas.DataFrame
            DataFrame containing features to cluster
        correlation_threshold : float
            Correlation threshold for clustering
            
        Returns
        -------
        dict
            Dictionary with cluster assignments
        """
        # Calculate the correlation matrix
        corr_matrix = X_features.corr().abs()
        
        # Initialize clusters: each feature starts in its own cluster
        clusters = {i: [feature] for i, feature in enumerate(X_features.columns)}
        cluster_id = len(clusters)
        
        # Create a copy of the correlation matrix to modify
        working_corr = corr_matrix.copy()
        
        # Set diagonal to 0 to avoid self-correlation
        np.fill_diagonal(working_corr.values, 0)
        
        # Merge clusters based on correlation
        while True:
            # Find the pair with the highest correlation
            i, j = np.unravel_index(working_corr.values.argmax(), working_corr.shape)
            max_corr = working_corr.iloc[i, j]
            
            # Stop if no correlation is above threshold
            if max_corr < correlation_threshold:
                break
                
            # Get feature names
            feature_i = X_features.columns[i]
            feature_j = X_features.columns[j]
            
            # Find which clusters these features belong to
            cluster_i, cluster_j = None, None
            for cluster_id, features in clusters.items():
                if feature_i in features:
                    cluster_i = cluster_id
                if feature_j in features:
                    cluster_j = cluster_id
                    
            # Merge clusters if they are different
            if cluster_i != cluster_j:
                clusters[cluster_i].extend(clusters[cluster_j])
                del clusters[cluster_j]
                
            # Set correlation to 0 to avoid reprocessing
            working_corr.iloc[i, j] = 0
            working_corr.iloc[j, i] = 0
        
        # Renumber clusters to be continuous
        new_clusters = {}
        for i, (_, features) in enumerate(clusters.items()):
            new_clusters[i] = features
        
        return new_clusters
    
    def _generate_cluster_info(self, X_features, clusters):
        """
        Generates a dictionary with information about the clusters
        
        Parameters
        ----------
        X_features : pandas.DataFrame
            DataFrame containing features
        clusters : dict or numpy.ndarray
            Cluster assignments
            
        Returns
        -------
        dict
            Dictionary with information about the clusters
        """
        cluster_info = {
            "n_clusters": 0,
            "clusters": {},
            "representatives": {},
            "dropped_features": []
        }
        
        # If clusters is a dictionary (from correlation clustering)
        if isinstance(clusters, dict):
            cluster_info["n_clusters"] = len(clusters)
            for cluster_id, feature_list in clusters.items():
                cluster_info["clusters"][cluster_id] = feature_list
                
                if len(feature_list) > 1:
                    # Find the feature with highest correlation to target or other criterion
                    rep_feature = feature_list[0]  # For simplicity, use first feature as representative
                    cluster_info["representatives"][cluster_id] = rep_feature
                    cluster_info["dropped_features"].extend([f for f in feature_list if f != rep_feature])
        
        # If clusters is a numpy array (from kmeans or hierarchical clustering)
        else:
            unique_clusters = np.unique(clusters)
            cluster_info["n_clusters"] = len(unique_clusters)
            features = X_features.columns
            
            for cluster_id in unique_clusters:
                # Convert to numpy for indexing
                features_array = np.array(features)
                mask = clusters == cluster_id
                cluster_features = features_array[mask].tolist()
                
                cluster_info["clusters"][int(cluster_id)] = cluster_features
                
                if len(cluster_features) > 1:
                    # Find the feature with highest correlation to target or other criterion
                    rep_feature = cluster_features[0]  # For simplicity, use first feature as representative
                    cluster_info["representatives"][int(cluster_id)] = rep_feature
                    cluster_info["dropped_features"].extend([f for f in cluster_features if f != rep_feature])
        
        return cluster_info
    
    def _replace_with_representatives(self, cluster_info):
        """Replace clustered features with representatives in the dataset"""
        for cluster_id, features in cluster_info['clusters'].items():
            if len(features) <= 1:
                continue  # Skip singleton clusters
                
            representative = cluster_info['representatives'][cluster_id]
            features_to_drop = [f for f in features if f != representative]
            
            # Keep only representative features
            if self.dataset.X_train is not None:
                self.dataset.X_train = self.dataset.X_train.drop(columns=features_to_drop)
            if self.dataset.X_val is not None:
                self.dataset.X_val = self.dataset.X_val.drop(columns=features_to_drop)
            if self.dataset.X_test is not None:
                self.dataset.X_test = self.dataset.X_test.drop(columns=features_to_drop)
    
    def _plot_feature_clusters(self, X_features, clusters, method, save_path):
        """Plot clusters visualization"""
        # Use PCA to visualize high-dimensional feature space
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.8)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Feature Clusters using {method.capitalize()}')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f}% variance)')
        
        if save_path:
            plt.savefig(f"{save_path}/feature_clusters_{method}.png")
        plt.close()