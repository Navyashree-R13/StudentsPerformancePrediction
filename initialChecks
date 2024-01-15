import pandas as pd

# Data cleaning involves handling missing values, outliers, and inconsistencies in the data.
# Let's start by reading the CSV file and performing an initial assessment.

# Read the CSV file into a DataFrame
df = pd.read_csv('Desktop/exams.csv')

# Initial assessment of data for missing values, unique counts for potential inconsistencies, and basic statistics for outliers
missing_values = df.isnull().sum()
unique_counts = df.nunique()
basic_statistics = df.describe()

missing_values, unique_counts, basic_statistics


# Outliers detection using IQR for numerical columns
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column]
    return outliers

outliers_math = detect_outliers(data, 'math score')
outliers_reading = detect_outliers(data, 'reading score')
outliers_writing = detect_outliers(data, 'writing score')

outliers_summary = {
    'Math Score Outliers': outliers_math,
    'Reading Score Outliers': outliers_reading,
    'Writing Score Outliers': outliers_writing
}

print(outliers_summary)

#Correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Reload the data due to execution environment reset
file_path = 'Desktop/exams.csv'
data = pd.read_csv(file_path)

# Recreating the correlation matrix for numerical features with different shades of blue
correlation_matrix = data[['math score', 'reading score', 'writing score']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.2f')
plt.title("Correlation Matrix of Score Variables")
plt.show()


# Check if 'overall score' column exists in the DataFrame
if 'overall score' in df.columns:
    correlation_with_overall = df.corr()['overall score']
else:
    # If 'overall score' is not in the DataFrame, create it and then compute the correlation
    df['overall score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
    correlation_with_overall = df.corr()['overall score']

correlation_with_overall

# Re-importing seaborn as sns
import seaborn as sns

# Regenerating the heatmap for the correlation matrix of scores
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.title("Correlation Matrix of Scores (Including Overall Score)")
plt.show()
