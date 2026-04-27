import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.utils import to_categorical

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
from sklearn.preprocessing import LabelEncoder
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

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert target to categorical for neural network
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# Build the neural network model
model = Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# Train the model
history = model.fit(
    X_train_scaled, y_train_cat,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Make predictions
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"\n\nNeural Network Model Accuracy: {accuracy:.4f}")
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
plt.title('Confusion Matrix - Neural Network')
plt.tight_layout()
plt.savefig('neural_networks_graph_confusion_matrix.png', dpi=100)
plt.show()

# Plot training loss only
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color='#E63946')
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#457B9D')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss - Neural Network', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('neural_networks_graph_loss.png', dpi=100)
plt.show()

# Plot training accuracy only
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='#2A9D8F')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#264653')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Training Accuracy - Neural Network', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('neural_networks_graph_accuracy.png', dpi=100)
plt.show()

# Calculate feature importance using permutation importance
from sklearn.inspection import permutation_importance

class KerasModelWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        # No-op fit to satisfy sklearn estimator interface
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        return np.argmax(preds, axis=1)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

wrapped_model = KerasModelWrapper(model)
feature_importance_result = permutation_importance(
    wrapped_model,
    X_test_scaled,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring='accuracy'
)
feature_names = X.columns.tolist()
indices = np.argsort(feature_importance_result.importances_mean)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importance (Permutation) - Neural Network')
plt.bar(range(len(feature_importance_result.importances_mean)), feature_importance_result.importances_mean[indices])
plt.xticks(range(len(feature_importance_result.importances_mean)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.tight_layout()
plt.savefig('neural_networks_graph_feature_importance.png', dpi=100)
plt.show()

print("\nNeural Network model trained successfully!")
