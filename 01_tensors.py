import torch

print("=== 01. Tensors Reference ===")

# 1. Creation
a = torch.tensor([1, 2, 3])
b = torch.zeros((2, 3))
c = torch.ones((2, 3))
d = torch.rand((2, 3))
e = torch.arange(0, 10, 2)
f = torch.linspace(0, 1, 5)

print(f"Tensor a: {a}")
print(f"Tensor d (random):\n{d}")

# 2. Operations
x = torch.rand((2, 2))
y = torch.rand((2, 2))

print(f"Addition: {x + y}")
print(f"Multiplication (Element-wise): {x * y}")
print(f"Matrix Multiplication: {x @ y}")

# 3. Reshaping
flat = torch.randn(12)
reshaped = flat.view(3, 4) # reshape to 3x4
print(f"Reshaped: {reshaped.shape}")

# 4. Slicing
print(f"First row of reshaped: {reshaped[0, :]}")
print(f"First col of reshaped: {reshaped[:, 0]}")

# 5. Type Casting
int_tensor = torch.tensor([1, 2, 3])
float_tensor = int_tensor.to(torch.float32)
print(f"Dtype: {float_tensor.dtype}")
