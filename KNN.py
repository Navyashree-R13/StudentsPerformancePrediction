import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the data
file_path = r'C:\Users\Nithyashree\Desktop\DAPM_Excel\ode_normalization.csv'
data = pd.read_csv(file_path)

# Calculate the mean of the overall score
mean_score = data['overall score'].mean()

# Create a binary classification target based on the mean score
data['performance'] = np.where(data['overall score'] >= mean_score, 1, 0)

# Selecting features and target
X = data.drop(['overall score', 'performance'], axis=1)
y = data['performance']

# Splitting the data into training and testing sets (70:30 split)
X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating and training the KNN model
knn_70 = KNeighborsClassifier()
knn_70.fit(X_train_70, y_train_70)

# Predicting and evaluating on the training set
y_pred_train_70 = knn_70.predict(X_train_70)
accuracy_train_70 = accuracy_score(y_train_70, y_pred_train_70)
precision_train_70, recall_train_70, f1_score_train_70, _ = precision_recall_fscore_support(y_train_70, y_pred_train_70)

# Predicting and evaluating on the test set
y_pred_test_30 = knn_70.predict(X_test_30)
accuracy_test_30 = accuracy_score(y_test_30, y_pred_test_30)
precision_test_30, recall_test_30, f1_score_test_30, _ = precision_recall_fscore_support(y_test_30, y_pred_test_30)

# Creating a DataFrame to display the results
results_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision (Below-Average)", "Recall (Below-Average)", "F1-Score (Below-Average)",
               "Precision (Above-Average)", "Recall (Above-Average)", "F1-Score (Above-Average)"],
    "Training Set Score": [accuracy_train_70, precision_train_70[0], recall_train_70[0], f1_score_train_70[0], precision_train_70[1], recall_train_70[1], f1_score_train_70[1]],
    "Test Set Score": [accuracy_test_30, precision_test_30[0], recall_test_30[0], f1_score_test_30[0], precision_test_30[1], recall_test_30[1], f1_score_test_30[1]]
})

results_df


#for 80:20
# Splitting the data into training and testing sets (80:20 split)
X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the KNN model for the 80:20 split
knn_80 = KNeighborsClassifier()
knn_80.fit(X_train_80, y_train_80)

# Predicting and evaluating on the training set
y_pred_train_80 = knn_80.predict(X_train_80)
accuracy_train_80 = accuracy_score(y_train_80, y_pred_train_80)
precision_train_80, recall_train_80, f1_score_train_80, _ = precision_recall_fscore_support(y_train_80, y_pred_train_80)

# Predicting and evaluating on the test set
y_pred_test_20 = knn_80.predict(X_test_20)
accuracy_test_20 = accuracy_score(y_test_20, y_pred_test_20)
precision_test_20, recall_test_20, f1_score_test_20, _ = precision_recall_fscore_support(y_test_20, y_pred_test_20)

# Creating a DataFrame to display the results for 80:20 split
results_df_80_20 = pd.DataFrame({
    "Metric": ["Accuracy", "Precision (Below-Average)", "Recall (Below-Average)", "F1-Score (Below-Average)",
               "Precision (Above-Average)", "Recall (Above-Average)", "F1-Score (Above-Average)"],
    "Training Set Score": [accuracy_train_80, precision_train_80[0], recall_train_80[0], f1_score_train_80[0], precision_train_80[1], recall_train_80[1], f1_score_train_80[1]],
    "Test Set Score": [accuracy_test_20, precision_test_20[0], recall_test_20[0], f1_score_test_20[0], precision_test_20[1], recall_test_20[1], f1_score_test_20[1]]
})

results_df_80_20


#k-value
# Create a binary classification target based on the mean score
data['performance'] = np.where(data['overall score'] >= mean_score, 1, 0)

# Selecting features and target
X = data.drop(['overall score', 'performance'], axis=1)
y = data['performance']

# Splitting the data into training and testing sets (80:20 split)
X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(X, y, test_size=0.2, random_state=42)

# Array to store scores
cross_val_scores = []

# List of different values of k to try
k_values = range(1, 6)

# Perform 10-fold cross-validation with different values of k
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_80, y_train_80, cv=10, scoring='accuracy')
    cross_val_scores.append(scores.mean())

# Find the value of k that gives the highest accuracy
best_k = k_values[cross_val_scores.index(max(cross_val_scores))]
best_score = max(cross_val_scores)

# Create a dataframe to display the cross-validation scores for each k
cv_results_df = pd.DataFrame({
    'k': k_values,
    'Cross-Validated Accuracy': cross_val_scores
})

# Display the best k value and its score along with the dataframe of all results
print("Best K-value is", best_k)
print("Accuracy for that value is", best_score)
print(cv_results_df)

#ROC for 80:20
# Define the feature variables (X) and the target variable (y)
X = data.drop(['overall score', 'performance'], axis=1)
y = data['performance']

# Split the data into training and testing sets (80:20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Predicted probabilities for ROC curve
y_scores_train = knn.predict_proba(X_train)[:, 1]
y_scores_test = knn.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC for training and test sets
fpr_train, tpr_train, _ = roc_curve(y_train, y_scores_train)
roc_auc_train = auc(fpr_train, tpr_train)
fpr_test, tpr_test, _ = roc_curve(y_test, y_scores_test)
roc_auc_test = auc(fpr_test, tpr_test)

# Plot ROC curves with AUC
plt.figure(figsize=(10, 6))
plt.plot(fpr_train, tpr_train, label=f'Training Set AUC = {roc_auc_train:.2f}')
plt.plot(fpr_test, tpr_test, label=f'Test Set AUC = {roc_auc_test:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve for K-Nearest Neighbors')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Confusion matrices for training and test sets
cm_train = confusion_matrix(y_train, knn.predict(X_train))
cm_test = confusion_matrix(y_test, knn.predict(X_test))

# Plotting the confusion matrices
plt.figure(figsize=(12, 5))

# Training set confusion matrix
plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Training Set)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0.5, 1.5], ['Below-Average', 'Above-Average'])
plt.yticks([0.5, 1.5], ['Below-Average', 'Above-Average'])

# Test set confusion matrix
plt.subplot(1, 2, 2)
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0.5, 1.5], ['Below-Average', 'Above-Average'])
plt.yticks([0.5, 1.5], ['Below-Average', 'Above-Average'])

plt.tight_layout()
plt.show()

#correlations
# Calculate the mean overall score to set our threshold
mean_overall_score = data['overall score'].mean()

# Create a binary target variable: 1 if above mean, 0 otherwise
data['is_above_average'] = (data['overall score'] > mean_overall_score).astype(int)

# Drop the 'overall score' as it should not be included in the feature set
data = data.drop(['overall score'], axis=1)

# Calculate the correlation of each feature with the target variable
correlation_with_target = data.corr()['is_above_average'].sort_values(ascending=False)

# Removing the target variable correlation with itself
correlation_with_target = correlation_with_target.drop(labels=['is_above_average'])

# Separating positive and negative correlations
positive_correlations = correlation_with_target[correlation_with_target > 0]
negative_correlations = correlation_with_target[correlation_with_target < 0]

# Display the correlations
print("Positive Correlations\n",positive_correlations)
print("Negative Correlations\n",negative_correlations)
