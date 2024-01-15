# It appears that the code execution environment has been reset. I will reload the data and then calculate the mode for gender.

import pandas as pd

# Reload the data
file_path = 'Desktop/exams.csv'
data = pd.read_csv(file_path)

# Calculate the mode for 'gender'
gender_mode = data['gender'].mode()[0]
gender_mode

# Calculating the mode for the 'race/ethnicity' variable
race_mode = data['race/ethnicity'].mode()[0]
race_mode


# Calculate the mode for 'parental level of education'
parental_education_mode = data['parental level of education'].mode()[0]
parental_education_mode


# Calculate the mode for 'lunch'
lunch_mode = data['lunch'].mode()[0]

print(lunch_mode)




#race
# Load the data
data = pd.read_csv('desktop/exams.csv')

# Calculate frequency distribution for 'race/ethnicity'
race_freq = data['race/ethnicity'].value_counts()

# Display the frequency distribution
race_freq

#parental education
# Calculating the relative frequency distribution (proportion) for 'parental level of education'
parental_education_proportions = data['parental level of education'].value_counts(normalize=True)

parental_education_proportions

# Calculating the frequency counts for the 'parental level of education' variable
parental_education_counts = data['parental level of education'].value_counts()
parental_education_counts

# Calculating the frequency count for the 'test preparation course' variable
test_prep_counts = data['test preparation course'].value_counts()
test_prep_counts

# Calculate the mode for 'test preparation course'
test_prep_mode = data['test preparation course'].mode()[0]
print(test_prep_mode)
