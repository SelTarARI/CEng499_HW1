import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from itertools import product
from scipy.stats import sem, t
import matplotlib.pyplot as plt

# Load all datasets for Part 3
x_train, y_train = pickle.load(open("/mnt/c/Users/selta/OneDrive/Documents/hw1/HW 1 Data and Source Codes/datasets/part3_train_dataset.dat", "rb"))
x_validation, y_validation = pickle.load(open("/mnt/c/Users/selta/OneDrive/Documents/hw1/HW 1 Data and Source Codes/datasets/part3_validation_dataset.dat", "rb"))
x_test, y_test = pickle.load(open("/mnt/c/Users/selta/OneDrive/Documents/hw1/HW 1 Data and Source Codes/datasets/part3_test_dataset.dat", "rb"))

# Rescale features
x_train, x_validation, x_test = [x / 255.0 for x in (x_train, x_validation, x_test)]
x_train, x_validation, x_test = [x.astype(np.float32) for x in (x_train, x_validation, x_test)]

# Convert to PyTorch tensors
x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train).long()
x_validation, y_validation = torch.from_numpy(x_validation), torch.from_numpy(y_validation).long()
x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test).long()

# Define Model with Configurable Hyperparameters
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fn):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(activation_fn())
            in_size = h_size
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Hyperparameter Grid
input_size = x_train.shape[1]
output_size = 10  # Assuming 10 classes

hyperparameter_grid = {
    'hidden_sizes': [(128,), (128, 64)],  # Different layer configurations
    'learning_rate': [0.01, 0.001],
    'epochs': [10, 20],  # Number of epochs
    'activation_fn': [nn.Sigmoid, nn.Tanh],  # Activation functions
    'batch_size': [32, 64]  # Batch sizes for minibatch gradient descent
}

# Generate all combinations of hyperparameters
all_configs = list(product(*hyperparameter_grid.values()))

# Function to run training and evaluation
def run_model(x_train, y_train, x_val, y_val, config, run_id=1):
    model = MLP(input_size, config['hidden_sizes'], output_size, config['activation_fn'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses, val_accuracies = [], [], []
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=config['batch_size'], shuffle=True)
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())
            _, val_preds = torch.max(val_outputs, 1)
            val_accuracy = (val_preds == y_val).float().mean().item() * 100
            val_accuracies.append(val_accuracy)
        
        # Print training progress
        print(f"Config {config} | Run {run_id} | Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_accuracy:.2f}%")
    
    return train_losses, val_losses, val_accuracies

# Confidence Interval Function
def compute_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean, se = np.mean(data), sem(data)
    h = se * t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h

# Perform Hyperparameter Search
results = {}
for config_index, config in enumerate(all_configs):
    config_dict = dict(zip(hyperparameter_grid.keys(), config))
    accuracy_scores = []
    
    print(f"\nTesting configuration {config_index+1}/{len(all_configs)}: {config_dict}")
    for run in range(10):
        train_losses, val_losses, val_accuracies = run_model(x_train, y_train, x_validation, y_validation, config_dict, run_id=run+1)
        accuracy_scores.append(val_accuracies[-1])  # Use last epoch's validation accuracy
    
    mean_acc, lower, upper = compute_confidence_interval(accuracy_scores)
    results[config] = {'mean_accuracy': mean_acc, 'confidence_interval': (lower, upper)}
    print(f"Config {config_dict} | Mean Accuracy: {mean_acc:.2f}% | CI: ({lower:.2f}%, {upper:.2f}%)")

# Select Best Configuration Based on Highest Mean Accuracy
best_config = max(results, key=lambda x: results[x]['mean_accuracy'])
best_config_dict = dict(zip(hyperparameter_grid.keys(), best_config))
print(f"\nBest configuration: {best_config_dict} with mean accuracy {results[best_config]['mean_accuracy']:.2f}% and CI: {results[best_config]['confidence_interval']}")

# Combine Training and Validation Data and Train Final Model
x_combined = torch.cat((x_train, x_validation), dim=0)
y_combined = torch.cat((y_train, y_validation), dim=0)
final_accuracies = []

for run in range(10):
    print(f"\nFinal training on combined data - Run {run+1}")
    _, _, val_accuracies = run_model(x_combined, y_combined, x_test, y_test, best_config_dict, run_id=run+1)
    final_accuracies.append(val_accuracies[-1])

# Report Final Test Accuracy with Confidence Interval
final_mean_acc, final_lower, final_upper = compute_confidence_interval(final_accuracies)
print(f"\nFinal test accuracy after combining data: {final_mean_acc:.2f}% with CI: ({final_lower:.2f}%, {final_upper:.2f}%)")

# Generate SRM Plots for Training and Validation Loss
for config_index, config in enumerate(all_configs[:10]):  # Limiting to first 10 configurations for SRM plots
    config_dict = dict(zip(hyperparameter_grid.keys(), config))
    all_train_losses, all_val_losses = [], []
    
    print(f"\nGenerating SRM plots for configuration {config_dict}")
    for run in range(10):
        train_losses, val_losses, _ = run_model(x_train, y_train, x_validation, y_validation, config_dict, run_id=run+1)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
    
    avg_train_loss = np.mean(all_train_losses, axis=0)
    avg_val_loss = np.mean(all_val_losses, axis=0)
    
    plt.figure()
    plt.plot(avg_train_loss, label='Train Loss')
    plt.plot(avg_val_loss, label='Validation Loss')
    plt.title(f"SRM Plot for Config: {config_dict}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
