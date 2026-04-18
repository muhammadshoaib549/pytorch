import torch
import torch.nn as nn

print("=== 05. Advanced: Custom Loss & Hooks ===")

# 1. CUSTOM LOSS FUNCTION
# You can subclass nn.Module to make a reusable loss
class MyCustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        loss = self.mse(pred, target) * self.alpha
        return loss

# 2. HOOKS (Inspecting internal activations)
model = nn.Linear(10, 2)
activations = []

def my_hook(module, input, output):
    print(f"--- Hook Triggered! ---")
    print(f"Layer: {module}")
    print(f"Output shape captured: {output.shape}")
    activations.append(output)

# Attach hook to the layer
handle = model.register_forward_hook(my_hook)

# Run data through
x = torch.randn(1, 10)
y_pred = model(x)

# Cleanup
handle.remove()
print(f"Captured activations count: {len(activations)}\n")
