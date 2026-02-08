import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
import pickle
import matplotlib.pyplot as plt
import gc

class GMModel:
    """
    Fits a Gaussian Mixture Model to a given embedding space and allows sampling from the fitted distribution.
    """
    def __init__(self, n_components=50, covariance_type='diag', seed=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.seed = seed
        self.model = None

    def find_optimal_k(self, data, min_k=10, max_k=200, step=10, plot_path=None):
        """
        Runs a grid search over K components using the Bayesian Information Criterion (BIC).
        """
        print(f"Running BIC analysis on {len(data)} vectors...")
        n_components_range = range(min_k, max_k + 1, step)
        bic_scores = []

        best_bic = float('inf')
        best_k = min_k

        for k in n_components_range:
            print(f"  Fitting GMM with k={k}...", end='\r')
            
            gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=self.seed)
            gmm.fit(data)
            
            bic = gmm.bic(data)
            bic_scores.append(bic)
            del gmm
            gc.collect()
            
            if bic < best_bic:
                best_bic = bic
                best_k = k

        print(f"Optimal K value: {best_k} (BIC: {best_bic:.2f})")

        if plot_path:
            plt.figure(figsize=(10, 6))
            plt.plot(n_components_range, bic_scores, marker='o')
            plt.title('BIC Score vs. Number of Components')
            plt.xlabel('Number of Components (k)')
            plt.ylabel('BIC Score')
            plt.grid(True)
            plt.savefig(plot_path)
            print(f"BIC plot saved to {plot_path}")

        return best_k

    def fit(self, data):
        """Fits the internal GMM to the provided data."""
        print(f"Fitting model (k={self.n_components}, cov={self.covariance_type})...")
        data = np.asanyarray(data, dtype=np.float32) # cast to float32 just in case
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.seed,
            reg_covar=1e-6, # add small regularization
            verbose=1
        )
        self.model.fit(data)

    def save(self, filepath):
        """Saves the entire object (configuration + fitted model) via pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Distribution model saved to {filepath}")

    @staticmethod
    def load(filepath):
        """Loads a pre-fitted model from disk."""
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        print(f"Distribution model loaded from {filepath}")
        return obj

    def sample(self, n_samples, noise_level=0.01):
        """
        Generates synthetic embeddings with added Gaussian noise.
        
        Returns:
            np.ndarray: Matrix of shape (n_samples, dimensions), L2 normalized.
        """
        if not self.model:
            raise ValueError("Model is not fitted. Call .fit() or .load() first.")

        # Sample GMM
        X_sample, _ = self.model.sample(n_samples)
        X_sample += np.random.normal(0, noise_level, X_sample.shape)
        
        # Apply L2 norm
        X_sample = normalize(X_sample, norm='l2', axis=1)
        
        return X_sample.astype(np.float32)