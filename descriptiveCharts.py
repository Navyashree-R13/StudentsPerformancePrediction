import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#race/ethnicity
# Sorting the data by race/ethnicity groups
sorted_data = data.sort_values('race/ethnicity')

# Defining the color palette with dark blue and light blue
color_palette = ["#00008B", "#3265a8"]  # Dark blue and light blue hex color codes

# Creating a sorted frequency distribution chart for race vs gender with specified colors
plt.figure(figsize=(10, 6))
sns.countplot(x='race/ethnicity', hue='gender', data=sorted_data, order=['group A', 'group B', 'group C', 'group D', 'group E'], palette=color_palette)
plt.title('Sorted Frequency Distribution of Race/Ethnicity by Gender')
plt.xlabel('Race/Ethnicity')
plt.ylabel('Frequency')
plt.legend(title='Gender')
plt.show()

# Counting the frequency of each lunch type
lunch_counts = data['lunch'].value_counts()

# Setting the style of the plot for a cleaner look
sns.set(style="white")

# Defining different shades of blue for the chart
blue_shades = ["#00008B", "#3265a8"]

# Creating the bar chart with different shades of blue
plt.figure(figsize=(6, 4))
sns.barplot(x=lunch_counts.index, y=lunch_counts.values, palette=blue_shades)
plt.title('Frequency of Each Lunch Type')
plt.xlabel('Lunch Type')
plt.ylabel('Frequency')

# Removing the top and right spines
sns.despine()

plt.show()

print("The mode for lunch is:", lunch_mode)

#Race/Ethnicity

# Load the data
file_path = 'Desktop/exams.csv'  # Replace with your file path if different
data = pd.read_csv(file_path)

# Calculating the mode for the 'race/ethnicity' variable
race_mode = data['race/ethnicity'].mode()[0]

# Counting the frequency of each race/ethnicity category
race_counts = data['race/ethnicity'].value_counts()

# Setting the style of the plot for a cleaner look
sns.set(style="white")

# Defining different shades of blue for the chart
blue_shades = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087']

# Creating the bar chart with different shades of blue
plt.figure(figsize=(8, 5))
sns.barplot(x=race_counts.index, y=race_counts.values, palette=blue_shades)
plt.title('Frequency of Each Race/Ethnicity Category')
plt.xlabel('Race/Ethnicity')
plt.ylabel('Frequency')

# Removing the top and right spines
sns.despine()

plt.show()

# Print the mode
print("The mode for race/ethnicity is:", race_mode)

# Counting the frequency of each lunch type
lunch_counts = data['lunch'].value_counts()

# Setting the style of the plot for a cleaner look
sns.set(style="white")

# Defining different shades of blue for the chart
blue_shades = ["#00008B", "#3265a8"]

# Creating the bar chart with different shades of blue
plt.figure(figsize=(6, 4))
sns.barplot(x=lunch_counts.index, y=lunch_counts.values, palette=blue_shades)
plt.title('Frequency of Each Lunch Type')
plt.xlabel('Lunch Type')
plt.ylabel('Frequency')

# Removing the top and right spines
sns.despine()

plt.show()

print("The mode for lunch is:", lunch_mode)


#test preparation course and lunch
# Defining the color palette with dark blue and light blue
color_palette = ["#00008B", "#3265a8"]  # Dark blue and light blue hex color codes

# Creating the frequency distribution chart with the updated colors
plt.figure(figsize=(10, 6))
sns.countplot(x='lunch', hue='test preparation course', data=sorted_data, palette=color_palette)
plt.title('Frequency Distribution of Lunch by Test Preparation Course')
plt.xlabel('Lunch')
plt.ylabel('Frequency')
plt.legend(title='Test Preparation Course')
plt.show()


# Counting the frequency of each category in 'test preparation course'
test_prep_counts = data['test preparation course'].value_counts()

# Setting the style of the plot for a cleaner look
sns.set(style="white")

# Defining different shades of blue for the chart
blue_shades = ["#00008B", "#3265a8"]

# Creating the bar chart with different shades of blue
plt.figure(figsize=(6, 4))
sns.barplot(x=test_prep_counts.index, y=test_prep_counts.values, palette=blue_shades)
plt.title('Frequency of Test Preparation Course Status')
plt.xlabel('Test Preparation Course')
plt.ylabel('Frequency')

# Removing the top and right spines
sns.despine()

plt.show()


#histogram for math, reading and writing score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
data = pd.read_csv('Desktop/exams.csv')  # Replace with the correct path to your file

# Set the aesthetic style of the plots
sns.set_style('whitegrid')

# Create a histogram for math scores
plt.figure(figsize=(6, 4))
sns.histplot(data['math score'], kde=True, color='blue')
plt.title('Histogram for Math Scores')
plt.xlabel('Math Score')
plt.ylabel('Frequency')
plt.show()

# Create a histogram for reading scores
plt.figure(figsize=(6, 4))
sns.histplot(data['reading score'], kde=True, color='green')
plt.title('Histogram for Reading Scores')
plt.xlabel('Reading Score')
plt.ylabel('Frequency')
plt.show()

# Create a histogram for writing scores
plt.figure(figsize=(6, 4))
sns.histplot(data['writing score'], kde=True, color='brown')
plt.title('Histogram for Writing Scores')
plt.xlabel('Writing Score')
plt.ylabel('Frequency')
plt.show()
