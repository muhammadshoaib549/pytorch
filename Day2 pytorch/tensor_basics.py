import torch

# 1. Checking Version and GPU Availability
print(f"PyTorch Version: {torch.__version__}")
# Output: PyTorch Version: 2.10.0+cu128

if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available. Using CPU.")
# Output:
# GPU is available!
# Using GPU: Tesla T4

# 2. Tensor Creation Methods
print("\n--- Tensor Creation ---")
# Empty tensor (allocates memory without initializing values)
a = torch.empty(2, 3)
print(f"Empty tensor:\n{a}")
# Output:
# tensor([[2.1715e-18, 2.3081e-12, 2.6302e+20],
#         [6.1943e-04, 3.6022e-12, 6.9989e+22]])

print(f"Type of a: {type(a)}")
# Output: <class 'torch.Tensor'>

# Ones tensor
ones_tensor = torch.ones(2, 3)
print(f"Ones tensor:\n{ones_tensor}")
# Output:
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])

# Random values tensor
rand_tensor = torch.rand(2, 3)
print(f"Random tensor:\n{rand_tensor}")
# Example Output:
# tensor([[0.2627, 0.0428, 0.2080],
#         [0.1180, 0.1217, 0.7356]])

# Manual Seed for reproducibility
torch.manual_seed(100)
random_tensor_seeded = torch.rand(2, 3)
print(f"Seeded Random tensor:\n{random_tensor_seeded}")
# Output:
# tensor([[0.1117, 0.8158, 0.2626],
#         [0.4839, 0.6765, 0.7539]])

# Explicit tensor creation
manual_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Manual tensor:\n{manual_tensor}")
# Output:
# tensor([[1, 2, 3],
#         [4, 5, 6]])

# 3. Range and Distribution Functions
print("\n--- Specialized Initializers ---")

# arange: (start, end, step) -> excludes end
print("Using arange  ->", torch.arange(0, 10, 2))
# Output: tensor([0, 2, 4, 6, 8])

# linspace: (start, end, steps) -> includes end
print("Using linspace ->", torch.linspace(0, 10, 10))
# Output: tensor([ 0.0000, 1.1111, 2.2222, 3.3333, 4.4444, 5.5556, 6.6667, 7.7778, 8.8889, 10.0000])

# eye: Identity Matrix (diagonal of 1s)
print("Using eye      ->\n", torch.eye(5))
# Output:
# tensor([[1., 0., 0., 0., 0.],
#         [0., 1., 0., 0., 0.],
#         [0., 0., 1., 0., 0.],
#         [0., 0., 0., 1., 0.],
#         [0., 0., 0., 0., 1.]])

# full: (shape, value) -> fills everything with the value
print("Using full     ->\n", torch.full((3, 3), 5))
# Output:
# tensor([[5, 5, 5],
#         [5, 5, 5],
#         [5, 5, 5]])

# 4. Tensor Attributes
print("\n--- Tensor Attributes ---")
x = torch.tensor([[1, 2, 3], [4, 5, 6], [1, 4, 5]])

print(f"Shape: {x.shape}")            # Output: torch.Size([3, 3])
print(f"Rank (ndim): {x.ndim}")       # Output: 2
print(f"Total elements: {x.numel()}") # Output: 9
print(f"Data type: {x.dtype}")       # Output: torch.int64
print(f"Device: {x.device}")         # Output: cpu

# 5. Tensors Like (Same shape as x)
print("\n--- Tensors Like ---")
zeros_like_x = torch.zeros_like(x)
print(f"Zeros like x:\n{zeros_like_x}")
# Output:
# tensor([[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]])

ones_like_x = torch.ones_like(x)
print(f"Ones like x:\n{ones_like_x}")
# Output:
# tensor([[1, 1, 1],
#         [1, 1, 1],
#         [1, 1, 1]])

# 6. Data Types and Conversion
print("\n--- Data Types Conversion ---")
x = torch.tensor([[1, 0, 3], [4, 5, 6]], dtype=torch.int32)
print(f"Original Tensor (int32):\n{x}")

# Floating point types
x_f32 = x.to(torch.float32)
x_f64 = x.to(torch.float64)
x_f16 = x.to(torch.float16)
x_bf16 = x.to(torch.bfloat16)

# Integer types
x_i8 = x.to(torch.int8)
x_i16 = x.to(torch.int16)
x_i32 = x.to(torch.int32)
x_i64 = x.to(torch.int64)

# Unsigned 8-bit
x_uint8 = x.to(torch.uint8)

# Boolean and Complex
x_bool = x.to(torch.bool)
x_c64 = x.to(torch.complex64)

# Results Comparison
# 32-bit Float    | Dtype: torch.float32      | Example Value: 1.0
# 64-bit Float    | Dtype: torch.float64      | Example Value: 1.0
# 16-bit Float    | Dtype: torch.float16      | Example Value: 1.0
# BFloat16        | Dtype: torch.bfloat16     | Example Value: 1.0
# 8-bit Int       | Dtype: torch.int8         | Example Value: 1
# 32-bit Int      | Dtype: torch.int32        | Example Value: 1
# 64-bit Int      | Dtype: torch.int64        | Example Value: 1
# Unsigned Int8   | Dtype: torch.uint8        | Example Value: 1
# Boolean         | Dtype: torch.bool         | Example Value: True
# Complex 64      | Dtype: torch.complex64    | Example Value: (1+0j)

# Generating random tensor matching shape of x but different dtype
random_tensor_like = torch.rand_like(x, dtype=torch.float32)
print("\nRandom Tensor (same shape as x, float32):")
print(random_tensor_like)

# 7. Mathematical Operations
print("\n--- Mathematical Operations ---")
x = torch.rand(2, 2)
# Example x: tensor([[0.9535, 0.7064], [0.1629, 0.8902]])

print(f"Addition (x + 10):\n{x + 10}")
# Output: tensor([[10.9535, 10.7064], [10.1629, 10.8902]])

print(f"Subtraction (x - 5):\n{x - 5}")
# Output: tensor([[-4.0465, -4.2936], [-4.8371, -4.1098]])

print(f"Multiplication (x * 2):\n{x * 2}")
# Output: tensor([[1.9071, 1.4128], [0.3258, 1.7804]])

print(f"Division (x / 2):\n{x / 2}")
# Output: tensor([[0.4768, 0.3532], [0.0814, 0.4451]])

print(f"Int Division (x // 0.5):\n{x // 0.5}")
# Output: tensor([[1., 1.], [0., 1.]])

# Power and Modulo
print(f"Power (x^2):\n{x ** 2}")
# Output: tensor([[0.9092, 0.4990], [0.0265, 0.7925]])

logic_gate = (x // 0.5) % 2
print(f"Logic Gate ((x // 0.5) % 2):\n{logic_gate}")
# Output: tensor([[1., 1.], [0., 1.]])

# 8. Element-wise Operations
print("\n--- Element-wise Operations ---")
# Example a: tensor([[0.6017, 0.4234, 0.5224], [0.4175, 0.0340, 0.9157]])
# Example b: tensor([[0.3079, 0.6269, 0.8277], [0.6594, 0.0887, 0.4890]])

# Addition
# Output: tensor([[0.9096, 1.0504, 1.3501], [1.0769, 0.1227, 1.4047]])

# Subtraction
# Output: tensor([[ 0.2938, -0.2035, -0.3052], [-0.2418, -0.0547,  0.4268]])

# Division
# Output: tensor([[1.9541, 0.6754, 0.6312], [0.6332, 0.3834, 1.8728]])

# Multiplication (Hadamard Product)
# Output: tensor([[0.1852, 0.2655, 0.4324], [0.2753, 0.0030, 0.4478]])
