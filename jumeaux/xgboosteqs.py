import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import xgboost as xgb

# Load the dataset
# Handle European decimal format (comma separator)
df = pd.read_csv('awf.csv', sep=';', decimal=',')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nTarget variable distribution:")
print(df['Defect'].value_counts())

# Separate features and target
X = df.drop('Defect', axis=1)
y = df['Defect']

print("\nFeatures:")
print(X.columns.tolist())

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

print(f"\nNumber of classes: {num_classes}")
print(f"Classes: {le.classes_}")

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Build and train the XGBoost model
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=num_classes,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
    verbosity=1
)

print("\nTraining XGBoost model...")
# Train with validation set monitoring
eval_set = [(X_test, y_test)]
model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=False
)

# Track accuracy and loss over iterations
train_losses = []
test_accuracies = []
n_estimators_range = range(1, model.n_estimators + 1)

print("Calculating accuracy at each iteration...")
for n_est in n_estimators_range:
    # Get predictions at current iteration
    y_pred_iter = model.predict(X_test, iteration_range=(0, n_est))
    test_acc = accuracy_score(y_test, y_pred_iter)
    test_accuracies.append(test_acc)

# Make predictions
y_pred = model.predict(X_test)

# Decode predictions back to original labels for display
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n\nXGBoost Model Accuracy: {accuracy:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - XGBoost')
plt.tight_layout()
plt.savefig('xgboost_graph_confusion_matrix.png', dpi=100)
plt.show()

# Plot training history (loss curve)
results = model.evals_result()
epochs = range(len(results['validation_0']['mlogloss']))
train_loss = results['validation_0']['mlogloss']

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, linewidth=2, color='#E63946')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Log Loss', fontsize=12)
plt.title('XGBoost Training Loss Over Epochs', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('xgboost_graph_loss.png', dpi=100)
plt.show()

# Plot accuracy over iterations
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, test_accuracies, linewidth=2, color='#06A77D')
plt.xlabel('Number of Estimators', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('XGBoost Model Accuracy Over Estimators', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim([min(test_accuracies) - 0.01, 1.0])
plt.tight_layout()
plt.savefig('xgboost_graph_accuracy.png', dpi=100)
plt.show()

# Feature importance plot
feature_importance = model.feature_importances_
feature_names = X.columns.tolist()

# Sort features by importance
indices = np.argsort(feature_importance)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title('Feature Importance - XGBoost')
plt.bar(range(X.shape[1]), feature_importance[indices])
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.tight_layout()
plt.savefig('xgboost_graph_feature_importance.png', dpi=100)
plt.show()

print("\n\nFeature Importance:")
for i in range(X.shape[1]):
    print(f"{feature_names[indices[i]]}: {feature_importance[indices[i]]:.4f}")
print(f"Number of classes: {num_classes}")
print(f"Number of estimators used: {model.best_iteration + 1 if hasattr(model, 'best_iteration') else model.n_estimators}")
