import pandas as pd

# Load the dataset
file_path = 'desktop/exams.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Calculate the overall score as the mean of the math, reading, and writing scores
df['overall score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

# Remove the individual math, reading, and writing scores
df_reduced = df.drop(columns=['math score', 'reading score', 'writing score'])

# Define the file path for the new CSV with only the overall score
new_file_path = 'desktop/DAPM_Excel/overall.csv'  # Replace with your desired save path

# Save the modified dataframe to a new CSV file
df_reduced.to_csv(new_file_path, index=False)


#adding performance column
file_path = 'desktop/DAPM_Excel/overall.csv'  # Replace with your file path
df = pd.read_csv(file_path)
average_score = df['overall score'].mean()
df['performance_category'] = (df['overall score'] > average_score).astype(int)
columns_to_drop = ['math score', 'reading score', 'writing score']
df.to_csv('desktop/DAPM_Excel/overall_dis.csv', index=False)


#encoding

# Print column names for verification
print("Column names in the dataset:", df.columns.tolist())

# List of categorical columns for one-hot encoding (adjust based on your dataset)
categorical_cols = ['gender', 'race/ethnicity', 'lunch', 'test preparation course']

# Check if all categorical columns are present in the DataFrame
for col in categorical_cols:
    if col not in df.columns:
        print(f"Column not found in DataFrame: {col}")

# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols)


# Ordinal encoding for 'parental level of education' (adjust keys to match your data)
education_levels = {
    'some high school': 1,
    'high school': 2,
    'some college': 3,
    "associate's degree": 4,
    "bachelor's degree": 5,
    "master's degree": 6
}
df_encoded['parental level of education'] = df['parental level of education'].map(education_levels)

# Save the processed dataframe to a new CSV file
df_encoded.to_csv('desktop/DAPM_Excel/overall_dis_encoded.csv', index=False)  # Replace with your save file path
