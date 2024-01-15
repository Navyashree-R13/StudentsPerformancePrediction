import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = r'C:\Users\Nithyashree\Desktop\DAPM_Excel\ode_normalization.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Convert 'overall score' to a binary variable: 1 if above average, 0 if below average
mean_score = data['overall score'].mean()
data['performance'] = np.where(data['overall score'] >= mean_score, 1, 0)

# Define the features (X) and the target variable (y)
X = data.drop(['overall score', 'performance'], axis=1)
y = data['performance']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Performance metrics for training set
train_report = classification_report(y_train, y_pred_train)
train_confusion_matrix = confusion_matrix(y_train, y_pred_train)

# Performance metrics for test set
test_report = classification_report(y_test, y_pred_test)
test_confusion_matrix = confusion_matrix(y_test, y_pred_test)

# Display the results
print("Training Set Performance Metrics:")
print(train_report)
print("Training Set Confusion Matrix:")
print(train_confusion_matrix)

print("\nTest Set Performance Metrics:")
print(test_report)
print("Test Set Confusion Matrix:")
print(test_confusion_matrix)

# Extracting the coefficients from the model
coefficients = pd.DataFrame(model.coef_[0], X.columns, columns=['Coefficient'])

# Calculating the odds ratio for better interpretability
coefficients['Odds Ratio'] = np.exp(coefficients['Coefficient'])

coefficients.sort_values(by='Odds Ratio', ascending=False)

# Process the data as done previously
mean_score = data['overall score'].mean()
data['performance'] = np.where(data['overall score'] >= mean_score, 1, 0)
X = data.drop(['overall score', 'performance'], axis=1)
y = data['performance']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Compute ROC curve and ROC area for training set
y_train_prob = model.predict_proba(X_train)[:, 1]
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
roc_auc_train = auc(fpr_train, tpr_train)

# Compute ROC curve and ROC area for test set
y_test_prob = model.predict_proba(X_test)[:, 1]
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
roc_auc_test = auc(fpr_test, tpr_test)

# Plotting
plt.figure(figsize=(10, 8))
plt.plot(fpr_train, tpr_train, color='blue', lw=2, label='ROC curve for training set (area = %0.2f)' % roc_auc_train)
plt.plot(fpr_test, tpr_test, color='green', lw=2, label='ROC curve for test set (area = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Load and process the data
file_path = r'C:\Users\Nithyashree\Desktop\DAPM_Excel\ode_normalization.csv'  # Replace with your file path
data = pd.read_csv(file_path)
mean_score = data['overall score'].mean()
data['performance'] = np.where(data['overall score'] >= mean_score, 1, 0)
X = data.drop(['overall score', 'performance'], axis=1)
y = data['performance']

# Splitting the data and training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions for training and test sets
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Confusion matrices
confusion_matrix_train = confusion_matrix(y_train, y_pred_train)
confusion_matrix_test = confusion_matrix(y_test, y_pred_test)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cmap="Blues")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title(title)

# Plotting Confusion Matrix for Training Set
plot_confusion_matrix(confusion_matrix_train, title='Confusion Matrix for Training Set')

# Plotting Confusion Matrix for Test Set
plot_confusion_matrix(confusion_matrix_test, title='Confusion Matrix for Test Set')
plt.show()

