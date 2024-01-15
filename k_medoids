import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_csv('desktop/DAPM_Excel/clustering.csv')



# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Function to calculate the total cost
def calculate_total_cost(X, medoids):
    total_cost = 0
    for i in range(X.shape[0]):
        total_cost += np.min([np.linalg.norm(X[i] - medoids[j]) for j in range(len(medoids))])
    return total_cost

# Finding the optimal number of clusters using the Elbow method
costs = []
K = range(1, 11)
for k in K:
    kmedoids = KMedoids(n_clusters=k, random_state=42)
    kmedoids.fit(data_scaled)
    costs.append(calculate_total_cost(data_scaled, kmedoids.cluster_centers_))

# Plotting the Elbow Chart
plt.figure(figsize=(10,6))
plt.plot(K, costs, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Total Cost')
plt.title('The Elbow Method showing the optimal number of clusters')
plt.show()

# Assume you have already standardized your data and it's named data_scaled


# Choose the optimal number of clusters based on the elbow chart
optimal_clusters = 3  # or 4, based on your judgement

# Initialize K-medoids with the optimal number of clusters
kmedoids = KMedoids(n_clusters=optimal_clusters, random_state=42)
kmedoids.fit(data_scaled)

# Get the cluster labels for each data point
cluster_labels = kmedoids.labels_

# Assign these cluster labels back to the original (or standardized) data
data['cluster'] = cluster_labels

# Now you can analyze the clusters in your dataset
# For example, to get the size of each cluster:
print(data['cluster'].value_counts())

# To examine the medoids:
print(kmedoids.cluster_centers_)

# If you need to validate the clusters, you could use silhouette score or other metrics
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(data_scaled, cluster_labels)
print("The average silhouette_score is :", silhouette_avg)

# Proceed with interpreting results and using the clusters for your specific application

#scatterplot
# Fit the K-medoids
optimal_clusters = 3  # Replace with the number of clusters you determined
kmedoids = KMedoids(n_clusters=optimal_clusters, random_state=42)
kmedoids.fit(data_scaled)

# Perform PCA for 2D visualization if necessary
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data_scaled)

# Plot the clusters
plt.figure(figsize=(8, 6))
unique_labels = set(kmedoids.labels_)
for k in unique_labels:
    cluster_members = data_reduced[kmedoids.labels_ == k]
    plt.scatter(cluster_members[:, 0], cluster_members[:, 1], label=f'Cluster {k}')

# Plot the medoids
medoids = pca.transform(kmedoids.cluster_centers_)
plt.scatter(medoids[:, 0], medoids[:, 1], s=100, c='black', marker='X', label='Medoids')

plt.title('K-medoids Clustering with PCA-reduced Data')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend()
plt.show()

# Apply K-medoids clustering
# Replace 3 with the number of clusters you have determined
kmedoids = KMedoids(n_clusters=3, random_state=42)
kmedoids.fit(data_scaled)
cluster_labels = kmedoids.labels_

# Assign cluster labels to the data
clustered_data = pd.DataFrame(data_scaled, columns=data.columns)
clustered_data['Cluster'] = cluster_labels

# Calculate the mean of each feature for each cluster
cluster_means = clustered_data.groupby('Cluster').mean()

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means, cmap='viridis', annot=True)
plt.title('Heatmap of Cluster Means for Each Feature')
plt.xlabel('Features')
plt.ylabel('Clusters')
plt.show()

pip install scikit-learn-extra matplotlib scikit-learn


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

# Load your data
# Replace 'path_to_your_data.csv' with the path to your dataset
data = pd.read_csv('desktop/DAPM_Excel/clustering.csv')

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply K-medoids clustering
# Replace 3 with the number of clusters you have determined
kmedoids = KMedoids(n_clusters=3, random_state=42)
kmedoids.fit(data_scaled)
cluster_labels = kmedoids.labels_

# Calculate the silhouette score
score = silhouette_score(data_scaled, cluster_labels)

print("Silhouette Score: ", score)


# Assuming 'kmedoids' is your trained K-medoids model and 'data' is your original dataset
cluster_labels = kmedoids.labels_
data['Cluster'] = cluster_labels

# Group the data by cluster and calculate mean (or median) values for each feature
cluster_profiles = data.groupby('Cluster').mean()

from sklearn.metrics import silhouette_samples

silhouette_vals = silhouette_samples(data_scaled, cluster_labels)

# Apply K-medoids clustering
# Replace the number 3 with the optimal number of clusters you determined
kmedoids = KMedoids(n_clusters=3, random_state=42)
kmedoids.fit(data_scaled)
cluster_labels = kmedoids.labels_

# Assign cluster labels to the original data
data['Cluster'] = cluster_labels

# Now you can proceed to analyze, visualize, and interpret the clustering results



# Group the data by cluster and calculate mean values for each feature
cluster_means = data.groupby('Cluster').mean()
print(cluster_means)

# Apply K-medoids clustering
# Replace 3 with the number of clusters you have determined
kmedoids = KMedoids(n_clusters=3, random_state=42)
kmedoids.fit(data_scaled)
cluster_labels = kmedoids.labels_

# Calculate the Davies-Bouldin Index
db_index = davies_bouldin_score(data_scaled, cluster_labels)

print("Davies-Bouldin Index:", db_index)

#to save the labelled clusters to each student
# Replace 3 with your optimal number of clusters
kmedoids = KMedoids(n_clusters=3, random_state=42)
kmedoids.fit(data_scaled)
cluster_labels = kmedoids.labels_

# Assign cluster labels to the original data
data['Cluster'] = cluster_labels

# Save the dataset with cluster labels to a new CSV file
data.to_csv('desktop/DAPM_Excel/clustering_with_kmedolabels.csv', index=False)


#medoids
# Apply K-medoids clustering
kmedoids = KMedoids(n_clusters=5, random_state=42)  # Adjust the number of clusters as needed
kmedoids.fit(data_scaled)

# Perform PCA to reduce the data to two dimensions
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data_scaled)

# Get the medoids and transform them to the two principal component space
medoids_reduced = pca.transform(kmedoids.cluster_centers_)

# Create a DataFrame with the cluster number and the two principal components of the medoids
medoids_df = pd.DataFrame(medoids_reduced, columns=['Centroid Dimension 1', 'Centroid Dimension 2'])
medoids_df.insert(0, 'Cluster', range(1, len(medoids_reduced) + 1))

# Print the DataFrame
print(medoids_df)
