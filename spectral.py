import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eigh
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class SpectralClustering:
    def __init__(self, n_clusters=3, n_neighbors=10):
        self.n_clusters = n_clusters  # Number of clusters to form
        self.n_neighbors = n_neighbors  # Number of neighbors for the graph

    def fit(self, data):
        self.data = data
        n_samples = data.shape[0]
        
        # Step 1: Construct the similarity graph using k-nearest neighbors
        adjacency_matrix = kneighbors_graph(data, self.n_neighbors, mode='connectivity', include_self=True)
        adjacency_matrix = adjacency_matrix.toarray()

        # Step 2: Compute the Laplacian matrix
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        laplacian_matrix = degree_matrix - adjacency_matrix

        # Step 3: Compute the first k eigenvectors of the Laplacian matrix
        _, eigvecs = eigh(laplacian_matrix, subset_by_index=[0, self.n_clusters-1])

        # Step 4: Apply k-means to the eigenvectors (treat them as data points)
        kmeans = KMeans(n_clusters=self.n_clusters)
        initial_clusters = eigvecs.copy()
        kmeans.fit(eigvecs)
        final_clusters = kmeans.labels_

        # Store number of iterations (epochs) in k-means
        self.epochs = kmeans.n_iter_

        # Step 5: Calculate final error rate (Sum of Squared Errors, SSE)
        self.error = kmeans.inertia_  # Inertia represents SSE in k-means

        return initial_clusters, final_clusters, self.epochs, self.error

# Example usage
if __name__ == "__main__":
    # Create a simple dataset
    data = np.random.rand(100, 2)

    # Initialize and fit Spectral Clustering
    spectral_clustering = SpectralClustering(n_clusters=3, n_neighbors=10)
    initial_clusters, final_clusters, epochs, error = spectral_clustering.fit(data)

    # Print results
    print("Initial clusters (from eigenvectors):")
    print(initial_clusters[:10])  # Display first 10 rows for brevity
    print("\nFinal clusters (from k-means):")
    print(final_clusters)
    print(f"\nTotal epochs in k-means: {epochs}")
    print(f"\nFinal error (SSE): {error}")
    
    # Optional: Visualize the final clusters
    plt.scatter(data[:, 0], data[:, 1], c=final_clusters, cmap='viridis')
    plt.title("Final Clusters (Spectral Clustering)")
    plt.show()
