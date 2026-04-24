# PyTorch Operations Guide

This is a comprehensive revision guide containing **every single operation** found in the `tensors.ipynb` notebook. Use this table for quick lookups and exam/project preparation.

---

## 📅 Day 1 - 3: Tensor Fundamentals
*The core concepts and operations learned in the first three days.*

## 1. System & Environment
| Operation Name | Working |
| :--- | :--- |
| `torch.__version__` | Checks the currently installed version of PyTorch. |
| `torch.cuda.is_available()` | Checks if a GPU (NVIDIA CUDA) is available on the system. |
| `torch.cuda.get_device_name(0)` | Returns the name of the GPU (e.g., "Tesla T4"). |
| `torch.cuda.synchronize()` | Forces the CPU to wait for all current GPU tasks to finish (used for timing). |
| `torch.device("cuda" or "cpu")` | Sets a variable to represent the target computing device. |

## 2. Creation & Initialization
| Operation Name | Working |
| :--- | :--- |
| `torch.tensor(data)` | Creates a tensor from a list, array, or other data. |
| `torch.empty(size)` | Allocates memory for a tensor without initializing values (fast, but has random "garbage"). |
| `torch.zeros(size)` | Creates a tensor with all elements set to `0`. |
| `torch.ones(size)` | Creates a tensor with all elements set to `1`. |
| `torch.rand(size)` | Random values from a Uniform distribution between 0 and 1. |
| `torch.randn(size)` | Random values from a Normal distribution (mean=0, std=1). |
| `torch.manual_seed(seed)` | Sets a fixed seed to make random number generation predictable. |
| `torch.arange(start, end, step)` | Creates a sequence from `start` to `end` (excludes end) with a fixed step. |
| `torch.linspace(start, end, steps)` | Creates a sequence of `steps` values between `start` and `end` (includes both). |
| `torch.eye(n)` | Creates an $n \times n$ Identity Matrix (1s on diagonal, 0s elsewhere). |
| `torch.full(size, value)` | Creates a tensor where every element is the specified `value`. |
| `torch.zeros_like(x)` | Creates a zero tensor with the exact same shape as tensor `x`. |
| `torch.ones_like(x)` | Creates a "ones" tensor with the exact same shape as tensor `x`. |
| `torch.rand_like(x)` | Creates a random tensor with the exact same shape as tensor `x`. |

## 3. Tensor Attributes & Inspection
| Operation Name | Working |
| :--- | :--- |
| `type(x)` | Python check to see if the object is a `torch.Tensor`. |
| `x.shape` | Shows the dimensions of the tensor (e.g., `3x3`). |
| `x.ndim` | Shows the "Rank" or number of dimensions (e.g., 2D = 2). |
| `x.numel()` | Returns the total count of elements inside the tensor. |
| `x.dtype` | Shows the data type (e.g., `float32`, `int64`, `bool`). |
| `x.device` | Shows where the tensor is stored (`cpu` or `cuda:0`). |
| `id(x)` | Python check for the memory address (used to check if two tensors share memory). |
| `x.item()` | Extracts a single Python scalar value from a tensor with 1 element. |

## 4. Arithmetic & Basic Math
| Operation Name | Working |
| :--- | :--- |
| `x + y` / `torch.add(x, y)` | Addition (Element-wise). |
| `x - y` / `torch.sub(x, y)` | Subtraction (Element-wise). |
| `x * y` / `torch.mul(x, y)` | Multiplication (Element-wise / Hadamard Product). |
| `x / y` / `torch.div(x, y)` | Division (Element-wise). |
| `x // y` | Floor Division (Integer division). |
| `x ** y` | Power / Exponentiation. |
| `x % y` | Modulo (Remainder). |
| `torch.abs(x)` | Absolute value (removes negative signs). |
| `torch.neg(x)` | Negates values (turns positive to negative and vice versa). |
| `torch.sqrt(x)` | Square root of every element. |
| `torch.log(x)` | Natural Logarithm (base $e$). |
| `torch.exp(x)` | Exponential ($e^x$). |

## 5. Rounding & Clamping
| Operation Name | Working |
| :--- | :--- |
| `torch.round(x)` | Rounds to the nearest integer. |
| `torch.floor(x)` | Always rounds DOWN to the nearest integer. |
| `torch.ceil(x)` | Always rounds UP to the nearest integer. |
| `torch.clamp(x, min, max)` | Forces values to stay inside the `[min, max]` range. |

## 6. Reductions (Statistical)
| Operation Name | Working |
| :--- | :--- |
| `x.sum()` | Sum of all elements. Use `dim=0` (cols) or `dim=1` (rows). |
| `x.mean()` | Average value (requires float tensor). |
| `x.max()` / `x.min()` | Returns the highest / lowest values. |
| `x.argmax()` / `x.argmin()` | Returns the **Index** (position) of the max / min value. |
| `x.std()` | Standard Deviation (measures spread of data). |
| `x.prod()` | Product of all elements multiplied together. |

## 7. Linear Algebra & Matrix Ops
| Operation Name | Working |
| :--- | :--- |
| `torch.matmul(a, b)` | True Matrix Multiplication (Dot product for matrices). |
| `a @ b` | Shorthand symbol for Matrix Multiplication. |
| `torch.transpose(x, 0, 1)` | Swaps two dimensions (e.g., changes Rows to Columns). |
| `x.T` | Shortcut for the transpose of a 2D matrix. |
| `torch.det(x)` | Calculates the Determinant of a square matrix. |
| `torch.inverse(x)` | Calculates the Inverse of a square matrix. |
| `torch.sum(m * n, dim=1)` | Manual Dot Product calculation per row. |

## 8. Logic & Comparisons
| Operation Name | Working |
| :--- | :--- |
| `m > n` | Returns a Boolean tensor (`True/False`) for element-wise check. |
| `x.any(dim)` | Returns `True` if **any** element in that dimension is True. |
| `x.all(dim)` | Returns `True` if **all** elements in that dimension are True. |

## 9. Activation Functions (DL)
| Operation Name | Working |
| :--- | :--- |
| `torch.sigmoid(x)` | Squashes values between 0 and 1 (Probabilities). |
| `torch.relu(x)` | If negative → 0. If positive → stays same. |
| `torch.softmax(x, dim)` | Turns a list into probabilities that sum to exactly 1. |

## 10. Reshaping & Slicing
| Operation Name | Working |
| :--- | :--- |
| `c.flatten()` | Turns a multi-dimensional matrix into a single 1D vector. |
| `c.reshape(shape)` | Changes the shape without changing the data. |
| `c.permute(2, 0, 1)` | Reorders the axes (e.g., `(H, W, C)` to `(C, H, W)`). |
| `c.unsqueeze(dim)` | Adds a "fake" dimension of size 1 (e.g., `[3, 3] -> [1, 3, 3]`). |
| `c.squeeze()` | Removes all dimensions that have a size of 1. |

## 11. Memory & Interoperability
| Operation Name | Working |
| :--- | :--- |
| `x.clone()` | Creates a completely independent copy in a new memory address. |
| `x.to(device)` | Moves the tensor to GPU or CPU. |
| `x.numpy()` | Converts a PyTorch tensor into a NumPy array (CPU only). |
| `x.add_(y)` / `x.relu_()` | **In-place Operations:** Modifies the original tensor directly to save RAM. |

---

## 📅 Day 4: Data Preprocessing & Manual Neural Networks
*The most advanced part: Preparing real data and building a learning brain.*

### 🧹 Part A: The Cleaning & Processing Pipeline
| Step | Code Implementation | Deep Explanation (The "Why") |
| :--- | :--- | :--- |
| **CSV Loading** | `pd.read_csv(link)` | Bringing external datasets (like Football scores) into your project. |
| **Handling Nulls**| `df.fillna(df.mean())` | Fills holes in data using averages. AI cannot calculate `NaN` values. |
| **Team Encoding**| `LabelEncoder.fit_transform()`| Converts words like "Pakistan" into ID number 5. Models only eat numbers. |
| **The Split** | `train_test_split()` | Keeping 20% data hidden until the end to check if the AI is truly smart. |
| **Standardizing**| `StandardScaler()` | Scaling goals (0-5) and city-IDs (0-200) so they all look similar (mean 0). |
| **Tensor Bridge**| `torch.from_numpy()` | Final conversion from cleaned NumPy arrays into PyTorch Tensors. |
| **Batch Reshape**| `y.view(-1, 1)` | Forcing targets into a vertical column shape to match neural output. |

### 🧠 Part B: Building & Training the NN (Autograd)
| Concept | Code Implementation | Deep Explanation (The "Why") |
| :--- | :--- | :--- |
| **Weights Init** | `torch.randn(x, h, grad=True)`| **Pattern Starters**: `grad=True` means PyTorch will remember their errors. |
| **Forward Pass** | `z = torch.mm(X, W) + B`| **The Pattern**: Calculating prediction by multiplying inputs with patterns. |
| **Activation** | `torch.sigmoid(z)` | **Probability**: Squashing numbers into 0-1 range for Win/Loss results. |
| **Loss Scoring** | `Binary Cross Entropy` | **The Penalty**: Log-based math that punishes wrong guesses heavily. |
| **Stability** | `torch.clamp(y_pred, eps, 1-eps)`| **Safety**: Prevents `log(0)` crash. Keeps predictions between 0.000...1 and 0.999...9. |
| **Backward Pass**| `loss.backward()` | **Learning**: Calculating exactly how much each weight contributed to the mistake. |
| **Weight Update**| `w -= lr * w.grad` | **Improvement**: Nudging weights slightly towards the correct answer. |
| **Zeroing Grads**| `w.grad.zero_()` | **Reset**: Wiping yesterday's mistakes so we can start fresh in the next round. |
| **Accuracy** | `(preds == y).float().mean()`| **Final Score**: What percentage of games did the model guess right? |
