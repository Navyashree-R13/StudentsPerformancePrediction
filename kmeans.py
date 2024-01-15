import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
from sklearn.decomposition import PCA

# Load the dataset
file_path = 'desktop/kmeans.csv'
data = pd.read_csv(file_path)

# Check for highly correlated features (correlation value of 1.0)
correlation_matrix = data.corr().abs()
high_correlation = np.where(correlation_matrix == 1)
high_correlation = [(correlation_matrix.columns[x], correlation_matrix.columns[y])
                    for x, y in zip(*high_correlation) if x != y and x < y]

# Drop one of each pair of highly correlated columns
for col1, col2 in high_correlation:
    data.drop(col2, axis=1, inplace=True)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Determine the optimal number of clusters using the elbow method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# Display the optimal number of clusters
print(optimal_clusters)


# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
data_pca = pca.fit_transform(scaled_data)

# Assuming the optimal number of clusters from the elbow chart is 5
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(data_pca)

# Scatter plot of the two principal components colored by cluster label
plt.figure(figsize=(10, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title('2D PCA of Data Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

data_reduced = data.copy()


# Adding the cluster labels to the original data
data['Cluster'] = clusters

# Get the cluster centers
cluster_centers = pd.DataFrame(kmeans.cluster_centers_)
print("Cluster Centers:\n", cluster_centers)
data_reduced = data.copy()


# Silhouette Score
silhouette_avg = silhouette_score(data_pca, clusters)
print(f'Silhouette Score: {silhouette_avg}')

# Calinski-Harabasz Index
calinski_harabasz = calinski_harabasz_score(data_pca, clusters)
print(f'Calinski-Harabasz Index: {calinski_harabasz}')

# Davies-Bouldin Index
davies_bouldin = davies_bouldin_score(data_pca, clusters)
print(f'Davies-Bouldin Index: {davies_bouldin}')



# Add cluster labels to the original data
data_reduced['Cluster'] = clusters

# Calculate mean values of original variables for each cluster
# Select only numeric columns for mean calculation
numeric_cols = data_reduced.select_dtypes(include=[np.number]).columns
cluster_variable_means = data_reduced.groupby('Cluster')[numeric_cols].mean()
print(cluster_variable_means.T)  # Transposed for vertical display

# Finding the optimal number of clusters
optimal_clusters = np.argmin(np.diff(np.diff(inertia))) + 2


# Generating a heatmap for the mean values of each cluster
cluster_means = data.groupby('Cluster').mean()

# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_means, annot=True, cmap='viridis')
plt.title('Heatmap of Cluster Means')
plt.ylabel('Cluster')
plt.show()

# Calculate Silhouette scores for each sample
silhouette_vals = silhouette_samples(data_pca, clusters)

# Averaging Silhouette scores for each cluster
for i in range(optimal_clusters):
    cluster_silhouette_vals = silhouette_vals[clusters == i]
    cluster_avg_silhouette = np.mean(cluster_silhouette_vals)
    print(f'Cluster {i} Average Silhouette Score: {cluster_avg_silhouette}')

clustered_data = pd.read_csv(file_path)

# Assuming 'kmeans_optimal' is your trained KMeans model
clusters = kmeans.labels_


data_reduced = data.copy()
# Add the cluster labels to your DataFrame
data['Cluster'] = clusters

# Calculate the mean 'overall score' for each cluster
mean_overall_scores = data.groupby('Cluster')['overall score'].mean()

# Print the mean 'overall score' for each cluster
print(mean_overall_scores)



# Assuming that the 'Cluster' column contains the cluster labels
# and the rest of the columns are the features used for clustering
features = data.drop('Cluster', axis=1)
cluster_labels = data['Cluster']

# Fit the KMeans model with the number of clusters equal to the unique values in the 'Cluster' column
kmeans = KMeans(n_clusters=cluster_labels.nunique(), n_init=10, random_state=42)
kmeans.fit(features)

# Calculate the SSE for each cluster
sse = kmeans.inertia_
print(f"Total SSE (inertia) for all clusters: {sse}")

# Calculate the SSE for each cluster
cluster_sse = []
for label in range(cluster_labels.nunique()):
    cluster_points = features[cluster_labels == label]
    center = kmeans.cluster_centers_[label]
    sse = np.sum((cluster_points - center) ** 2)
    cluster_sse.append(sse)

# Create a DataFrame for displaying the results
sse_df = pd.DataFrame({
    'Cluster': range(1, cluster_labels.nunique() + 1),
    'Number of Observations': cluster_labels.value_counts().sort_index()
})

print(sse_df)







