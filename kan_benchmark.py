import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Import our custom KAN layer
from kan_layer import KANLayer

# ==========================================
# 1. Dataset Generation (Symbolic Regression)
# ==========================================
# Let's try to fit a complex symbolic curve: y = sin(3x) + cos(5x) * exp(-x^2)
def target_function(x):
    return torch.sin(3 * x) + torch.cos(5 * x) * torch.exp(-x**2)

# Generate training data
x_train = torch.linspace(-2, 2, 400).unsqueeze(1)
y_train = target_function(x_train)

# Generate testing data
x_test = torch.linspace(-2.5, 2.5, 200).unsqueeze(1)
y_test = target_function(x_test)

# ==========================================
# 2. Model Definitions
# ==========================================
# Model A: Standard MLP
class MLP(nn.Module):
    def __init__(self, hidden_dim=64):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# Model B: Custom KAN
class SimpleKAN(nn.Module):
    def __init__(self, hidden_dim=8, grid_size=10, spline_order=3):
        super(SimpleKAN, self).__init__()
        # We need a much smaller hidden dimension for KAN to match parameter counts
        self.layer1 = KANLayer(1, hidden_dim, grid_size=grid_size, spline_order=spline_order)
        self.layer2 = KANLayer(hidden_dim, 1, grid_size=grid_size, spline_order=spline_order)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 3. Training Loop Setup
# ==========================================
mlp_model = MLP(hidden_dim=32)
kan_model = SimpleKAN(hidden_dim=4, grid_size=10, spline_order=3)

print(f"MLP Parameters: {count_parameters(mlp_model)}")
print(f"KAN Parameters: {count_parameters(kan_model)}")

def train_model(model, name, epochs=10000, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 200 == 0:
            print(f"[{name}] Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")
            
    # Final eval
    with torch.no_grad():
        test_pred = model(x_test)
        test_loss = criterion(test_pred, y_test).item()
        
    print(f"[{name}] Final Test Loss: {test_loss:.6f}")
    return test_pred, losses

# ==========================================
# 4. Run Benchmark & Plot
# ==========================================
print("\n--- Training MLP ---")
mlp_pred, mlp_losses = train_model(mlp_model, "MLP")

print("\n--- Training KAN ---")
kan_pred, kan_losses = train_model(kan_model, "KAN")

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot True Function
plt.plot(x_test.numpy(), y_test.numpy(), 'k--', label='True Function', linewidth=2)

# Plot MLP
plt.plot(x_test.numpy(), mlp_pred.numpy(), 'm-', label=f'MLP (Params: {count_parameters(mlp_model)})', alpha=0.8)

# Plot KAN
plt.plot(x_test.numpy(), kan_pred.numpy(), 'c-', label=f'KAN (Params: {count_parameters(kan_model)})', alpha=0.8)

plt.title('KAN vs. MLP on Symbolic Regression Task')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig('kan_vs_mlp_benchmark.png', dpi=300, bbox_inches='tight')
print("\nPlot saved successfully as 'kan_vs_mlp_benchmark.png'")
