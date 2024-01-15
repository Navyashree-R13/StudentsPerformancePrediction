import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
#for the first 10 rows
# Re-loading the dataset
file_path = 'desktop/exams.csv'
data = pd.read_csv(file_path)

# Selecting only the numerical data (math, reading, and writing scores)
numerical_data = data[['math score', 'reading score', 'writing score']]
euclidean_distance_matrix = squareform(pdist(numerical_data, metric='euclidean'))


# Extracting the Euclidean distance matrix for the first 10 rows
euclidean_distance_matrix_first_10 = euclidean_distance_matrix[:10, :10]

# Displaying the Euclidean distance matrix for the first 10 rows
print (euclidean_distance_matrix_first_10)


#for the entire dataset
# Calculating the Euclidean distance matrix for the entire dataset
euclidean_distance_matrix_full = squareform(pdist(numerical_data, metric='euclidean'))

# Convert the distance matrix to a DataFrame for easier handling
euclidean_distance_df = pd.DataFrame(euclidean_distance_matrix_full)

# Save the DataFrame as a CSV file
output_file_path = 'desktop/euclidean_distance_matrix_full.csv'
euclidean_distance_df.to_csv(output_file_path, index=False)

print(output_file_path)

#heatmap for euclidean distance
# Creating a heatmap for the Euclidean distance matrix (first 5 rows) in shades of blue
plt.figure(figsize=(8, 6))
sns.heatmap(euclidean_distance_matrix_first_5, cmap='Blues', annot=True)
plt.title('Heatmap of Euclidean Distance Matrix (First 5 Rows)')
plt.xlabel('Student Index')
plt.ylabel('Student Index')
plt.show()


#euclidean for ordinal data
# Mapping for 'parental level of education' based on assumed ordinality
education_level_mapping = {
    'some high school': 1,
    'high school': 2,
    'some college': 3,
    'associate\'s degree': 4,
    'bachelor\'s degree': 5,
    'master\'s degree': 6
}

# Encoding the 'parental level of education' column
ordinal_encoded_education = data['parental level of education'].map(education_level_mapping)

# Calculate the pairwise Euclidean distance
ordinal_encoded_education_reshaped = ordinal_encoded_education.values.reshape(-1, 1)
euclidean_distance_matrix_ordinal = squareform(pdist(ordinal_encoded_education_reshaped, metric='euclidean'))

# The Euclidean distance matrix is now stored in `euclidean_distance_matrix_ordinal`
print(euclidean_distance_matrix_ordinal)


# Calculate the pairwise Euclidean distance for the first 10 rows of the ordinal column
euclidean_distance_matrix_ordinal_first_10 = squareform(pdist(ordinal_encoded_education_reshaped[:10], metric='euclidean'))

# Displaying the Euclidean distance matrix for the first 10 rows of the ordinal data
print(euclidean_distance_matrix_ordinal_first_10)

# Generating the linkage matrix for the Euclidean distances of the first 10 rows
Z_ordinal = linkage(euclidean_distance_matrix_ordinal_first_10, method='ward')

# Plotting the Hierarchical Clustering Dendrogram for the first 10 rows
plt.figure(figsize=(10, 6))
dendrogram(Z_ordinal, leaf_rotation=90., leaf_font_size=10.)
plt.title('Hierarchical Clustering Dendrogram (First 10 Rows - Ordinal Data)')
plt.xlabel('Data Point Index')
plt.ylabel('Distance')
plt.show()


#hamming for nominal

# Selecting only the nominal categorical data from the dataset, excluding 'parental level of education'
nominal_columns = [col for col in data.select_dtypes(include=['object']).columns if col != 'parental level of education']
nominal_data_excluding_ordinal = data[nominal_columns]

# Convert the nominal data to a numeric format using label encoding
nominal_numeric_excluding_ordinal = nominal_data_excluding_ordinal.apply(lambda x: x.astype('category').cat.codes)

# Calculate the pairwise Hamming distance
hamming_distance_matrix_excluding_ordinal = squareform(pdist(nominal_numeric_excluding_ordinal, metric='hamming'))

# Displaying the full Hamming distance matrix
hamming_distance_matrix_excluding_ordinal



# Assuming 'hamming_distance_matrix_excluding_ordinal' is already calculated
# Generating the linkage matrix using Ward's method for the first 10 rows
Z_sampled = linkage(hamming_distance_matrix_excluding_ordinal[:10, :10], method='ward')

# Plotting the Hierarchical Clustering Dendrogram for the first 10 rows
plt.figure(figsize=(10, 6))
dendrogram(Z_sampled, leaf_rotation=90., leaf_font_size=10.)
plt.title('Hierarchical Clustering Dendrogram (First 10 Rows)')
plt.xlabel('Data Point Index')
plt.ylabel('Distance')
plt.show()

#for first 10 rows


# Selecting only the nominal categorical data from the dataset, excluding 'parental level of education'
nominal_columns = [col for col in data.select_dtypes(include=['object']).columns if col != 'parental level of education']
nominal_data = data[nominal_columns]

# Convert the nominal data to a numeric format using label encoding for the first 10 rows
nominal_numeric = nominal_data.apply(lambda x: x.astype('category').cat.codes).iloc[:10]

# Calculate the pairwise Hamming distance for the first 10 rows
hamming_distance_matrix = squareform(pdist(nominal_numeric, metric='hamming'))

# Convert the Hamming distance matrix to a DataFrame for display
hamming_distance_matrix_df = pd.DataFrame(hamming_distance_matrix)
hamming_distance_matrix_df.columns = ['Student ' + str(i+1) for i in range(10)]
hamming_distance_matrix_df.index = ['Student ' + str(i+1) for i in range(10)]

hamming_distance_matrix_df
