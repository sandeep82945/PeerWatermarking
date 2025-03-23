import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import json
from collections import Counter

# ---------------------------
# 1. Data Loading and Preprocessing
# ---------------------------
# Load the dataset from the JSON file
with open('result/Outputs_0.4_GD3.0_Greeen_term.json', 'r') as file:
    data = json.load(file)

# Extract features and labels from the dataset
z_scores = []
green_fractions = []
p_values = []
labels = []

for entry in data:
    if "output_without" in entry and "output_with" in entry:
        # Extract features for both watermarked and non-watermarked cases
        z_scores.append(entry["output_without"]["z_score"])
        green_fractions.append(entry["output_without"]["green_fraction"])
        p_values.append(entry["output_without"]["p_value"])
        labels.append(0)  # Label 0 for non-watermarked

        z_scores.append(entry["output_with"]["z_score"])
        green_fractions.append(entry["output_with"]["green_fraction"])
        p_values.append(entry["output_with"]["p_value"])
        labels.append(1)  # Label 1 for watermarked

# Convert lists to numpy arrays
z_scores = np.array(z_scores)
green_fractions = np.array(green_fractions)
p_values = np.array(p_values)
labels = np.array(labels)

# Check class distribution
print("Class distribution:", Counter(labels))

# Combine features into a single array
features = np.column_stack((z_scores, green_fractions, p_values))

# Standardize features across the entire dataset.
# (Alternatively, you can move scaling inside the CV loop to avoid data leakage.)
scaler = StandardScaler()
features = scaler.fit_transform(features)

# ---------------------------
# 2. Define the Neural Network Model
# ---------------------------
class WatermarkClassifier(nn.Module):
    def __init__(self):
        super(WatermarkClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 16)  # Input size is 3 (z_score, green_fraction, p_value)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)   # Output size is 2 (binary classification)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# ---------------------------
# 3. k-Fold Cross Validation Setup
# ---------------------------
k = 5  # number of folds
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# To store metrics for each fold
fold_metrics = []

fold_number = 0
for train_val_index, test_index in skf.split(features, labels):
    fold_number += 1
    print(f"\n===== Fold {fold_number} =====")
    
    # Split data into training+validation and test sets for this fold
    X_train_val, X_test = features[train_val_index], features[test_index]
    y_train_val, y_test = labels[train_val_index], labels[test_index]

    # --- Optionally, re-fit the scaler on the training fold only ---
    # scaler = StandardScaler()
    # X_train_val = scaler.fit_transform(X_train_val)
    # X_test = scaler.transform(X_test)
    
    # Further split training+validation data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # ---------------------------
    # 4. Model Training (with Early Stopping)
    # ---------------------------
    model = WatermarkClassifier()  # reinitialize model for each fold
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    patience = 500       # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    num_epochs = 10000
    best_model_state = None

    for epoch in range(num_epochs):
        # Training step
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
    
        # Early stopping check
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            epochs_without_improvement = 0
            best_model_state = model.state_dict()  # save best model state
        else:
            epochs_without_improvement += 1
    
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load the best model state for evaluation on the test fold
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # ---------------------------
    # 5. Model Evaluation
    # ---------------------------
    model.eval()
    with torch.no_grad():
        # Evaluate on training set (optional)
        y_pred_train = torch.argmax(model(X_train_tensor), dim=1)
        train_accuracy = (y_pred_train == y_train_tensor).float().mean().item()
    
        # Evaluate on validation set (optional)
        y_pred_val = torch.argmax(model(X_val_tensor), dim=1)
        val_accuracy = (y_pred_val == y_val_tensor).float().mean().item()
    
        # Evaluate on test set
        y_pred_test = torch.argmax(model(X_test_tensor), dim=1)
        test_accuracy = (y_pred_test == y_test_tensor).float().mean().item()
        test_precision = precision_score(y_test_tensor.numpy(), y_pred_test.numpy(), zero_division=1)
        test_recall = recall_score(y_test_tensor.numpy(), y_pred_test.numpy(), zero_division=1)
        test_f1 = f1_score(y_test_tensor.numpy(), y_pred_test.numpy(), zero_division=1)
    
    print("\nFold Metrics:")
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Testing Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}")
    
    fold_metrics.append({
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1
    })

# ---------------------------
# 6. Compute Average Metrics Across Folds
# ---------------------------
avg_train_acc = np.mean([m["train_accuracy"] for m in fold_metrics])
avg_val_acc = np.mean([m["val_accuracy"] for m in fold_metrics])
avg_test_acc = np.mean([m["test_accuracy"] for m in fold_metrics])
avg_test_precision = np.mean([m["test_precision"] for m in fold_metrics])
avg_test_recall = np.mean([m["test_recall"] for m in fold_metrics])
avg_test_f1 = np.mean([m["test_f1"] for m in fold_metrics])

print("\n===== Final Cross Validation Metrics =====")
print(f"Average Training Accuracy: {avg_train_acc * 100:.2f}%")
print(f"Average Validation Accuracy: {avg_val_acc * 100:.2f}%")
print(f"Average Testing Accuracy: {avg_test_acc * 100:.2f}%")
print(f"Average Testing Precision: {avg_test_precision:.4f}")
print(f"Average Testing Recall: {avg_test_recall:.4f}")
print(f"Average Testing F1 Score: {avg_test_f1:.4f}")

# Optionally, save the best model from the last fold
# torch.save(model.state_dict(), "watermark_classifier_best_fold.pth")
