# Full code for the analysis and model training

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = r'C:\Users\Nithyashree\Desktop\DAPM_Excel\ode_normalization.csv'
data = pd.read_csv(file_path)

# Define the threshold for 'above average' and 'below average' performance
average_threshold = data['overall score'].mean()
data['performance'] = np.where(data['overall score'] >= average_threshold, 1, 0)

# Drop the original 'overall score' as it's no longer needed for prediction
data = data.drop(columns=['overall score'])

# Splitting the data into features and target variable
X = data.drop('performance', axis=1)
y = data['performance']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier (a type of non-linear regression model)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions and Performance Evaluation
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Performance Metrics
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
train_report = classification_report(y_train, y_pred_train)
test_report = classification_report(y_test, y_pred_test)

# Output results
print("Training Set Performance:")
print("Accuracy:", train_accuracy)
print("Classification Report:\n", train_report)

print("Test Set Performance:")
print("Accuracy:", test_accuracy)
print("Classification Report:\n", test_report)

# Training the Random Forest model on the entire dataset to evaluate feature importance
model.fit(X, y)

# Extracting feature importance
feature_importance = model.feature_importances_

# Creating a DataFrame for easier visualization
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})

# Sorting the features based on importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

feature_importance_df


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Reload the dataset
file_path = r'C:\Users\Nithyashree\Desktop\DAPM_Excel\ode_normalization.csv'
data = pd.read_csv(file_path)

# Define the threshold for 'above average' and 'below average' performance
average_threshold = data['overall score'].mean()
data['performance'] = np.where(data['overall score'] >= average_threshold, 1, 0)
data = data.drop(columns=['overall score'])

# Splitting the data into features and target variable
X = data.drop('performance', axis=1)
y = data['performance']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Making predictions on both the training and test sets
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Generating confusion matrices
confusion_matrix_train = confusion_matrix(y_train, y_pred_train)
confusion_matrix_test = confusion_matrix(y_test, y_pred_test)

# Function to plot a confusion matrix
def plot_confusion_matrix(cm, title):
    """Function to plot a confusion matrix."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False,
                xticklabels=['Predicted Below Avg', 'Predicted Above Avg'],
                yticklabels=['Actual Below Avg', 'Actual Above Avg'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Plotting confusion matrices for training and test sets
plot_confusion_matrix(confusion_matrix_train, "Training Set Confusion Matrix")
plot_confusion_matrix(confusion_matrix_test, "Test Set Confusion Matrix")

from sklearn.metrics import roc_curve, auc

# Retraining the model on the training set
model.fit(X_train, y_train)

# Predict probabilities for training and test sets
y_train_probs = model.predict_proba(X_train)[:, 1]
y_test_probs = model.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC for training set
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_probs)
roc_auc_train = auc(fpr_train, tpr_train)

# Compute ROC curve and AUC for test set
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_probs)
roc_auc_test = auc(fpr_test, tpr_test)

# Plotting ROC curves
plt.figure(figsize=(10, 6))
plt.plot(fpr_train, tpr_train, color='blue', lw=2, label='Train ROC curve (area = %0.2f)' % roc_auc_train)
plt.plot(fpr_test, tpr_test, color='green', lw=2, label='Test ROC curve (area = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()
