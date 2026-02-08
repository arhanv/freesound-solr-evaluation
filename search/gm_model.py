"""
Gaussian Mixture Model (GMM) wrapper for probability distribution modeling of embedding spaces.

This module provides the `GMModel` class, which encapsulates training a GMM on high-dimensional
vector spaces (like CLAP or PCA-reduced embeddings) and sampling new "synthetic" vectors
from that learned distribution. It includes utilities for determining the optimal number of 
components (K) via the Bayesian Information Criterion (BIC).
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
import pickle
import matplotlib.pyplot as plt
import gc
import sys
import os

class GMModel:
    """Fits a Gaussian Mixture Model to an embedding space and allows sampling new vectors.

    Attributes:
        n_components (int): Number of mixture components.
        covariance_type (str): Type of covariance parameters to use ('full', 'tied', 'diag', 'spherical').
        seed (int): Random seed for reproducibility.
        model (sklearn.mixture.GaussianMixture): The fitted sklearn GMM object.
        rng (np.random.RandomState): Random number generator for noise addition.
    """

    def __init__(self, n_components=50, covariance_type='diag', seed=42, max_iter=100):
        """Initializes the GMModel.

        Args:
            n_components (int): The number of mixture components.
            covariance_type (str): String describing the type of covariance parameters to use.
            seed (int): Random seed.
            max_iter (int): Maximum number of iterations for the E-M algorithm.
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.seed = seed
        self.max_iter = max_iter
        self.model = None
        self.rng = np.random.RandomState(seed)

    def find_optimal_k(self, data, min_k=10, max_k=200, step=20, plot_path=None):
        """Runs a grid search over K components to minimize the Bayesian Information Criterion (BIC).

        Args:
            data (np.ndarray): The input data to fit.
            min_k (int): Minimum number of components to test.
            max_k (int): Maximum number of components to test.
            step (int): Step size for K.
            plot_path (str, optional): If provided, saves a plot of BIC scores to this path.

        Returns:
            int: The K value that resulted in the lowest BIC score.
        """
        data = np.asanyarray(data, dtype=np.float32)
        print(f"Running BIC analysis on {len(data)} vectors (k={min_k} to {max_k})...")
        n_components_range = range(min_k, max_k + 1, step)
        bic_scores = []

        best_bic = float('inf')
        best_k = min_k

        for k in n_components_range:
            print(f"  Fitting GMM with k={k}...", end='\r')
            sys.stdout.write("\033[s") 
            sys.stdout.flush()
            
            gmm = GaussianMixture(
                n_components=k, 
                covariance_type='diag', 
                random_state=self.seed,
                max_iter=self.max_iter,
                reg_covar=1e-5,
                verbose=1
            )
            gmm.fit(data)

            sys.stdout.write("\033[u\033[J\033[A\033[K")
            sys.stdout.flush()
            
            bic = gmm.bic(data)
            bic_scores.append(bic)
            
            if bic < best_bic:
                best_bic = bic
                best_k = k
                # Keep the best model
                if self.model:
                    del self.model
                self.model = gmm
            else:
                del gmm
            
            gc.collect()

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
        """Fits the internal GMM to the provided data.
        
        Args:
            data (np.ndarray): The training data (n_samples, n_features).
        """
        print(f"Fitting model (k={self.n_components}, cov={self.covariance_type})...")
        data = np.asanyarray(data, dtype=np.float32)
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.seed,
            max_iter=self.max_iter,
            reg_covar=1e-5, # add small regularization
            verbose=1
        )
        self.model.fit(data)

    def save(self, filepath):
        """Saves the entire object (configuration + fitted model) via pickle.

        Args:
            filepath (str): The destination file path.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Distribution model saved to {filepath}")

    @staticmethod
    def load(filepath):
        """Loads a pre-fitted GMModel from disk.

        Args:
            filepath (str): Path to the pickled GMModel file.

        Returns:
            GMModel: The loaded object.
        """
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        print(f"Distribution model loaded from {filepath}")
        return obj

    def sample(self, n_samples, noise_level=0.01):
        """Generates synthetic embedding vectors from the fitted GMM.
        
        Samples are drawn from the GMM, perturbed with slight Gaussian noise, and L2-normalized.

        Args:
            n_samples (int): Number of vectors to generate.
            noise_level (float): Standard deviation of Gaussian noise added to samples.

        Returns:
            np.ndarray: Matrix of shape (n_samples, dimensions), L2 normalized.
        
        Raises:
            ValueError: If the model has not been fitted or loaded.
        """
        if not self.model:
            raise ValueError("Model is not fitted. Call .fit() or .load() first.")

        # Sample GMM
        X_sample, _ = self.model.sample(n_samples)
        
        # Add noise using the instance's RNG
        X_sample += self.rng.normal(0, noise_level, X_sample.shape)
        
        # Apply L2 norm
        X_sample = normalize(X_sample, norm='l2', axis=1)
        
        return X_sample.astype(np.float32)