import pandas as pd

# Load the dataset
df = pd.read_csv('desktop/DAPM_Excel/overall_dis.csv')  # Replace with your file path

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
