import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
file_path = r'C:\Users\Nithyashree\Desktop\DAPM_Excel\ode_normalization.csv'
data = pd.read_csv(file_path)

# Creating binary labels based on the mean overall score
mean_score = data['overall score'].mean()
data['label'] = (data['overall score'] >= mean_score).astype(int)

# Defining the features and the label
X = data.drop(['overall score', 'label'], axis=1)
y = data['label']

# Function to split data, train model, and evaluate metrics
def train_evaluate_metrics(X, y, test_size):
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Normalizing the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the SVM model
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # Predictions for training and testing sets
    y_train_pred = svm_model.predict(X_train_scaled)
    y_test_pred = svm_model.predict(X_test_scaled)

    # Metrics calculation
    metrics = {
        'Accuracy': (accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)),
        'Precision (Below Average)': (precision_score(y_train, y_train_pred, pos_label=0), precision_score(y_test, y_test_pred, pos_label=0)),
        'Precision (Above Average)': (precision_score(y_train, y_train_pred, pos_label=1), precision_score(y_test, y_test_pred, pos_label=1)),
        'Recall (Below Average)': (recall_score(y_train, y_train_pred, pos_label=0), recall_score(y_test, y_test_pred, pos_label=0)),
        'Recall (Above Average)': (recall_score(y_train, y_train_pred, pos_label=1), recall_score(y_test, y_test_pred, pos_label=1)),
        'F1 Score (Below Average)': (f1_score(y_train, y_train_pred, pos_label=0), f1_score(y_test, y_test_pred, pos_label=0)),
        'F1 Score (Above Average)': (f1_score(y_train, y_train_pred, pos_label=1), f1_score(y_test, y_test_pred, pos_label=1))
    }

    return metrics

# Perform evaluation for 80:20 and 70:30 splits
metrics_80_20 = train_evaluate_metrics(X, y, test_size=0.20)
metrics_70_30 = train_evaluate_metrics(X, y, test_size=0.30)

# Combining results into a DataFrame for display
metrics_df = pd.DataFrame({
    'Metric': metrics_80_20.keys(),
    'Training 80:20': [m[0] for m in metrics_80_20.values()],
    'Testing 80:20': [m[1] for m in metrics_80_20.values()],
    'Training 70:30': [m[0] for m in metrics_70_30.values()],
    'Testing 70:30': [m[1] for m in metrics_70_30.values()]
})

metrics_df.set_index('Metric', inplace=True)
print(metrics_df)

#ROC Curve

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Function to plot ROC Curve
def plot_roc_curve(model, X_train, y_train, X_test, y_test):
    # Predict probabilities
    probs_train = model.predict_proba(X_train)
    probs_test = model.predict_proba(X_test)

    # Compute ROC curve and AUC for training set
    fpr_train, tpr_train, _ = roc_curve(y_train, probs_train[:, 1])
    auc_train = auc(fpr_train, tpr_train)

    # Compute ROC curve and AUC for testing set
    fpr_test, tpr_test, _ = roc_curve(y_test, probs_test[:, 1])
    auc_test = auc(fpr_test, tpr_test)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, color='blue', label=f'Train AUC = {auc_train:.2f}')
    plt.plot(fpr_test, tpr_test, color='red', label=f'Test AUC = {auc_test:.2f}')
    plt.plot([0, 1], [0, 1], color='darkgray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for SVM')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Initialize the SVM model with probability estimates
svm_model_prob = SVC(kernel='linear', probability=True, random_state=42)

# Train the SVM model
svm_model_prob.fit(X_train_scaled, y_train)

# Plot ROC Curve for both training and test sets
plot_roc_curve(svm_model_prob, X_train_scaled, y_train, X_test_scaled, y_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Function to plot confusion matrix for both sets in a single chart
def plot_confusion_matrix(model, X_train, y_train, X_test, y_test):
    # Predictions for training and testing sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Confusion matrices for training and testing sets
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Confusion matrix for training set
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=ax[0])
    ax[0].set_title("Confusion Matrix - Training Set")
    ax[0].set_xlabel("Predicted Labels")
    ax[0].set_ylabel("True Labels")
    ax[0].set_xticklabels(['Below Average', 'Above Average'])
    ax[0].set_yticklabels(['Below Average', 'Above Average'])

    # Confusion matrix for testing set
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens", ax=ax[1])
    ax[1].set_title("Confusion Matrix - Testing Set")
    ax[1].set_xlabel("Predicted Labels")
    ax[1].set_ylabel("True Labels")
    ax[1].set_xticklabels(['Below Average', 'Above Average'])
    ax[1].set_yticklabels(['Below Average', 'Above Average'])

    plt.tight_layout()
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(svm_model, X_train_scaled, y_train, X_test_scaled, y_test)

# Extracting the coefficients from the trained SVM model
svm_coefficients = svm_model.coef_[0]

# Creating a DataFrame for visualization
feature_names = X.columns
coefficients_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': svm_coefficients
})

# Sorting the DataFrame by the absolute values of the coefficients
coefficients_df = coefficients_df.sort_values(by='Coefficient', key=abs, ascending=False)

# Visualizing the coefficients
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coefficients_df)
plt.title('Feature Coefficients in SVM Model')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()

coefficients_df
