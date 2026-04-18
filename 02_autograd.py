import torch

print("=== 02. Autograd Reference ===")

# 1. requires_grad=True tracks operations on the tensor
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 2. Define a function z = x^2 + y^3
z = x**2 + y**3

print(f"Result z: {z}")

# 3. Backward pass (calculates gradients)
z.backward()

# 4. Check gradients (dz/dx and dz/dy)
# dz/dx = 2*x = 2*2 = 4
# dz/dy = 3*y^2 = 3*3^2 = 27
print(f"Gradient of x (dz/dx): {x.grad}")
print(f"Gradient of y (dz/dy): {y.grad}")

# 5. Overriding autograd (e.g. during evaluation)
with torch.no_grad():
    m = x + y
    print(f"m requires grad? {m.requires_grad}")

# 6. Detaching
detached_x = x.detach()
print(f"Detached x requires grad? {detached_x.requires_grad}")
