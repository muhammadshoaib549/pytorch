import torch # Import the PyTorch library

if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")  
print("=== 01. Tensors Reference ===") # Print a header for the reference guide

# 1. Creation # Section 1: Different ways to create tensors
a = torch.tensor([1, 2, 3]) # Create a tensor from a Python list
b = torch.zeros((2, 3)) # Create a 2x3 tensor filled with zeros
c = torch.ones((2, 3)) # Create a 2x3 tensor filled with ones
d = torch.rand((2, 3)) # Create a 2x3 tensor with random values from a uniform distribution [0,1)
e = torch.arange(0, 10, 2) # Create a 1D tensor with values from 0 to 8 (step 2)
f = torch.linspace(0, 1, 5) # Create a 1D tensor with 5 evenly spaced values from 0 to 1

print(f"Tensor a: {a}") # Display the contents of tensor a
print(f"Tensor d (random):\n{d}") # Display the contents of tensor d

# 2. Operations # Section 2: Common mathematical operations on tensors
x = torch.rand((2, 2)) # Create a random 2x2 tensor x
y = torch.rand((2, 2)) # Create another random 2x2 tensor y

print(f"Addition: {x + y}") # Add tensors x and y element-wise
print(f"Multiplication (Element-wise): {x * y}") # Multiply x and y element-wise (Hadamard product)
print(f"Matrix Multiplication: {x @ y}") # Perform standard matrix multiplication between x and y

# 3. Reshaping # Section 3: Changing the shape/dimensions of tensors
flat = torch.randn(12) # Create a 1D tensor with 12 random normal values
reshaped = flat.view(3, 4) # Reshape the 1D tensor into a 3x4 tensor (view shares memory)
print(f"Reshaped: {reshaped.shape}") # Print the shape of the reshaped tensor

# 4. Slicing # Section 4: Accessing specific elements or subsets of tensors
print(f"First row of reshaped: {reshaped[0, :]}") # Access all elements in the first row (index 0)
print(f"First col of reshaped: {reshaped[:, 0]}") # Access all elements in the first column (index 0)

# 5. Type Casting # Section 5: Converting tensors between different data types
int_tensor = torch.tensor([1, 2, 3]) # Create a tensor with integer data type
float_tensor = int_tensor.to(torch.float32) # Cast the integer tensor to float32
print(f"Dtype: {float_tensor.dtype}") # Print the data type of the new tensor
