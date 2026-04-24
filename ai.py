import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# 1. Prunable Linear Layer
# ===============================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Gate scores (learnable)
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)  # values between 0 and 1
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)


# ===============================
# 2. Model
# ===============================
class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32*32*3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ===============================
# 3. Sparsity Loss
# ===============================
def compute_sparsity_loss(model):
    sparsity_loss = 0
    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates = layer.get_gates()
            sparsity_loss += gates.sum()
    return sparsity_loss


# ===============================
# 4. Sparsity Calculation
# ===============================
def calculate_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0
    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates = layer.get_gates().detach()
            total += gates.numel()
            pruned += (gates < threshold).sum().item()
    return 100 * pruned / total


# ===============================
# 5. Training Function
# ===============================
def train(model, device, train_loader, optimizer, lambda_):
    model.train()
    total_loss = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        classification_loss = F.cross_entropy(output, target)

        sparsity_loss = compute_sparsity_loss(model)

        loss = classification_loss + lambda_ * sparsity_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ===============================
# 6. Test Function
# ===============================
def test(model, device, test_loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    return 100. * correct / len(test_loader.dataset)


# ===============================
# 7. Plot Gate Distribution
# ===============================
def plot_gate_distribution(model):
    all_gates = []

    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates = layer.get_gates().detach().cpu().numpy().flatten()
            all_gates.extend(gates)

    plt.figure()
    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Values")
    plt.ylabel("Frequency")
    plt.show()


# ===============================
# 8. Main Execution
# ===============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)

    # Lambda values to test
    lambdas = [1e-5, 1e-4, 1e-3]

    results = []

    for lambda_ in lambdas:
        print(f"\nTraining with lambda = {lambda_}")

        model = PrunableNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Train
        for epoch in range(10):
            loss = train(model, device, train_loader, optimizer, lambda_)
            print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

        # Evaluate
        accuracy = test(model, device, test_loader)
        sparsity = calculate_sparsity(model)

        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Sparsity: {sparsity:.2f}%")

        results.append((lambda_, accuracy, sparsity))

        # Plot for last model
        plot_gate_distribution(model)

    # Print results table
    print("\nFinal Results:")
    print("Lambda\tAccuracy\tSparsity")
    for r in results:
        print(f"{r[0]}\t{r[1]:.2f}\t\t{r[2]:.2f}")


if __name__ == "__main__":
    main()