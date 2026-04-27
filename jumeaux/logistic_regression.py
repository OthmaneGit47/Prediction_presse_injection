import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
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

# Standardize the features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train the logistic regression model using SGDClassifier
# This allows us to track loss over epochs
model = SGDClassifier(
    loss='log_loss',  # Logistic regression loss
    penalty='l2',
    max_iter=100,
    random_state=42,
    early_stopping=False,
    warm_start=False,
    eta0=0.01  # Learning rate
)

# Track loss over epochs
train_losses = []
learning_rates = []
epoch = 0

print("\nTraining Logistic Regression model...")
for i in range(100):
    model.partial_fit(X_train_scaled, y_train, classes=np.unique(y_train))
    
    # Calculate loss on training data
    train_score = model.score(X_train_scaled, y_train)
    train_losses.append(1 - train_score)  # Convert accuracy to loss
    
    # Track learning rate (eta0 remains constant in SGD)
    learning_rates.append(model.eta0)
    
    epoch += 1
    if (i + 1) % 20 == 0:
        print(f"Epoch {i + 1}/100, Training Loss: {train_losses[-1]:.4f}")

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"\nModel Accuracy: {accuracy:.4f}")
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
plt.title('Confusion Matrix - Logistic Regression')
plt.tight_layout()
plt.savefig('logistic_regression_graph_confusion_matrix.png', dpi=100)
plt.show()

# Plot training loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, linewidth=2, color='#2E86AB')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss (1 - Accuracy)', fontsize=12)
plt.title('Training Loss Over Epochs - Logistic Regression', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('logistic_regression_graph_loss.png', dpi=100)
plt.show()

# Plot feature importance (coefficients)
feature_names = X.columns.tolist()
coefficients = model.coef_[0] if model.coef_.shape[0] == 1 else np.mean(np.abs(model.coef_), axis=0)
indices = np.argsort(np.abs(coefficients))[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importance (Coefficients) - Logistic Regression')
plt.bar(range(len(coefficients)), np.abs(coefficients)[indices])
plt.xticks(range(len(coefficients)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.ylabel('Absolute Coefficient Value')
plt.xlabel('Features')
plt.tight_layout()
plt.savefig('logistic_regression_graph_feature_importance.png', dpi=100)
plt.show()

print("\nModel trained successfully!")
print(f"Classes: {model.classes_}")



