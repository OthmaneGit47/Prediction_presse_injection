import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

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

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Build and train the Random Forest model with warm_start to track loss over iterations
model = RandomForestClassifier(
    n_estimators=1,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    oob_score=True,
    warm_start=True,
    verbose=0
)

print("\nTraining Random Forest model...")
# Track out-of-bag error and accuracy as trees are added
oob_errors = []
test_accuracies = []
n_trees_range = range(1, 101)

for n_trees in n_trees_range:
    model.set_params(n_estimators=n_trees)
    model.fit(X_train, y_train)
    oob_error = 1 - model.oob_score_
    oob_errors.append(oob_error)
    
    # Calculate test accuracy
    y_pred_temp = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_temp)
    test_accuracies.append(test_acc)
    
    if n_trees % 20 == 0:
        print(f"Trees: {n_trees}/100, OOB Error: {oob_error:.4f}, Test Accuracy: {test_acc:.4f}")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"\n\nRandom Forest Model Accuracy: {accuracy:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.tight_layout()
plt.savefig('random_forrest_graph_confusion_matrix.png', dpi=100)
plt.show()

# Plot OOB error as loss curve (similar to XGBoost)
plt.figure(figsize=(10, 6))
plt.plot(n_trees_range, oob_errors, linewidth=2, color='#2E86AB')
plt.xlabel('Number of Trees', fontsize=12)
plt.ylabel('Out-of-Bag Error', fontsize=12)
plt.title('Random Forest Training Loss Over Trees', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('random_forrest_graph_loss.png', dpi=100)
plt.show()

# Plot accuracy over number of trees
plt.figure(figsize=(10, 6))
plt.plot(n_trees_range, test_accuracies, linewidth=2, color='#06A77D')
plt.xlabel('Number of Trees', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Random Forest Model Accuracy Over Trees', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim([min(test_accuracies) - 0.01, 1.0])
plt.tight_layout()
plt.savefig('random_forrest_graph_accuracy.png', dpi=100)
plt.show()

# Feature importance
feature_importance = model.feature_importances_
feature_names = X.columns.tolist()

# Sort features by importance
indices = np.argsort(feature_importance)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title('Feature Importance - Random Forest')
plt.bar(range(X.shape[1]), feature_importance[indices])
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.tight_layout()
plt.savefig('random_forrest_graph_feature_importance.png', dpi=100)
plt.show()

print("\n\nFeature Importance:")
for i in range(X.shape[1]):
    print(f"{feature_names[indices[i]]}: {feature_importance[indices[i]]:.4f}")

print("\nRandom Forest model trained successfully!")
print(f"Classes: {model.classes_}")
print(f"Number of trees: {model.n_estimators}")
