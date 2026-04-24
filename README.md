# PyTorch Mega Revision Guide 🚀

This is a comprehensive, A-to-Z revision guide covering every concept and operation found in the `tensors.ipynb` notebook. It is structured to help you revise everything from basic tensor math to building a manual neural network.

---

## 1. System & Environment
| Operation Name | Description |
| :--- | :--- |
| `torch.__version__` | Checks the currently installed version of PyTorch. |
| `torch.cuda.is_available()` | Checks if a GPU (NVIDIA CUDA) is available on the system. |
| `torch.cuda.get_device_name(0)` | Returns the name of the GPU (e.g., "Tesla T4"). |
| `torch.cuda.synchronize()` | Forces CPU to wait for GPU tasks (essential for accurate timing). |
| `torch.device("cuda" or "cpu")` | Creates a device object for target computation. |

## 2. Creation & Initialization
| Operation Name | Description |
| :--- | :--- |
| `torch.tensor(data)` | Creates a tensor from data (list, numpy array). |
| `torch.empty(size)` | Allocates memory without initializing (fast but random garbage). |
| `torch.zeros(size)` | Creates a tensor filled with `0`. |
| `torch.ones(size)` | Creates a tensor filled with `1`. |
| `torch.rand(size)` | Random values from Uniform distribution [0, 1]. |
| `torch.randn(size)` | Random values from Normal distribution (mean 0, std 1). |
| `torch.manual_seed(100)` | Fixes the random numbers so they are the same every time you run. |
| `torch.arange(0, 10, 2)` | Sequence from 0 up to 10 (excludes 10) with step 2 → `[0, 2, 4, 6, 8]`. |
| `torch.linspace(0, 10, 5)`| 5 equally spaced values between 0 and 10 (includes 10). |
| `torch.eye(n)` | Identity Matrix ($n \times n$) with 1s on the diagonal. |
| `torch.full((2, 2), 7)` | Creates a $2 \times 2$ matrix filled entirely with `7`. |
| `torch.zeros_like(x)` | Creates a zero tensor with the same shape/type as `x`. |

## 3. Tensor Attributes & Inspection
| Operation Name | Description |
| :--- | :--- |
| `x.shape` | Returns the dimensions (e.g., `torch.Size([3, 3])`). |
| `x.ndim` / `x.dim()` | Returns the number of dimensions (Rank). |
| `x.numel()` | Returns the total count of elements inside. |
| `x.dtype` | Data type: `float32`, `int64` (Long), `bool`, etc. |
| `x.device` | Shows where tensor lives: `cpu` or `cuda:0`. |
| `x.item()` | Converts a 1-element tensor into a Python number. |

## 4. Arithmetic & Mathematical Ops
| Operation Name | Description |
| :--- | :--- |
| `x + y`, `x - y`, `x * y`, `x / y` | Element-wise basic arithmetic. |
| `x // y` | Floor division (Integer division). |
| `x ** 2` | Exponentiation (Square). |
| `x % y` | Modulo (Remainder). |
| `torch.abs(x)` | Absolute value. |
| `torch.neg(x)` | Negates the values. |
| `torch.sqrt(x)` | Square root. |
| `torch.log(x)` | Natural logarithm (base $e$). |
| `torch.exp(x)` | Exponential ($e^x$). |

## 5. Rounding & Clamping
| Operation Name | Description |
| :--- | :--- |
| `torch.round(x)` | Rounds to the nearest integer. |
| `torch.floor(x)` | Always rounds DOWN (e.g., 3.9 → 3). |
| `torch.ceil(x)` | Always rounds UP (e.g., 3.1 → 4). |
| `torch.clamp(x, min, max)` | Limits values to a range; essential for loss function stability. |

## 6. Matrix & Linear Algebra
| Operation Name | Description |
| :--- | :--- |
| `torch.matmul(a, b)` / `a @ b` | **Matrix Multiplication** (Rows $\times$ Columns). |
| `torch.mm(a, b)` | Strictly for 2D matrix multiplication. |
| `a.T` / `torch.transpose(a, 0, 1)` | Flips rows and columns. |
| `torch.det(x)` | Calculates the Determinant. |
| `torch.inverse(x)` | Finds the inverse matrix. |
| `torch.sum(a * b, dim=1)` | Row-wise Dot Product. |

## 7. Reductions (Aggregation)
| Operation Name | Description |
| :--- | :--- |
| `x.sum(dim=0)` | Sum across columns. |
| `x.mean(dim=1)` | Average across rows (requires float). |
| `x.max().item()` | Finds the global maximum value. |
| `x.argmax(dim=0)` | Finds the **index** of the max value in each column. |
| `x.std()` | Standard Deviation (how spread out the data is). |
| `x.prod()` | Multiplies all elements together. |
| `x.any()` / `x.all()` | Returns True if any/all elements are True. |

## 8. Shape Manipulation
| Operation Name | Description |
| :--- | :--- |
| `x.reshape(rows, cols)` | Changes shape (safer, handles non-contiguous memory). |
| `x.view(rows, cols)` | Changes shape (faster, but requires contiguous memory). |
| `x.flatten()` | Collapses all dimensions into a 1D vector. |
| `x.squeeze()` | Removes all dimensions of size 1. |
| `x.unsqueeze(dim=0)` | Adds a "fake" dimension of size 1 at the specified index. |
| `x.permute(2, 0, 1)` | Reorders axes (e.g., Image HWC → CHW). |

## 9. Data Preprocessing (ML Workflow)
*Used for preparing raw data (like the Football Dataset) for AI models.*

| Step | Tool / Code | Logic |
| :--- | :--- | :--- |
| **Cleaning** | `df.dropna()`, `df.fillna(df.mean())` | Removes or fills missing values. |
| **Encoding** | `LabelEncoder.fit_transform()` | Converts team names ("Brazil") into numbers (0, 1, 2). |
| **Splitting**| `train_test_split(X, y, test_size=0.2)` | Separates data into Training (80%) and Testing (20%). |
| **Scaling** | `StandardScaler.fit_transform()` | Rescales features to have mean=0 and std=1 (helps training). |
| **Pytorch Bridge** | `torch.from_numpy(arr)` | Converts a NumPy array into a PyTorch Tensor. |

## 10. Activation Functions
| Function | Code | Behavior |
| :--- | :--- | :--- |
| **Sigmoid** | `torch.sigmoid(x)` | Squashes values between **0 and 1** (Probabilities). |
| **ReLU** | `torch.relu(x)` | If $x < 0 \to 0$; else stays same. (Removes negatives). |
| **Softmax** | `torch.softmax(x, dim=1)` | Scaled probabilities that **sum to 1** across a row. |

## 11. Manual Neural Network (The Training Loop)
*Revision of the "SimpleNeuralNetwork" class and training logic.*

### Core Components:
- **Weights & Bias**: Initialized with `requires_grad=True` to track gradients.
- **Forward Pass**: $y\_pred = \sigma(X \cdot W + b)$
- **Loss Function**: Binary Cross Entropy calculates error between prediction and reality.
- **Backward Pass**: `loss.backward()` calculates the adjustment needed for each weight.
- **Optimizer Step**: `w -= lr * w.grad` modifies the weights to reduce error.

### The "Must-Know" Logic:
| Concept | Code | Why we use it? |
| :--- | :--- | :--- |
| **Autograd** | `loss.backward()` | Automatically calculates the gradients for us. |
| **No Grad** | `with torch.no_grad():` | Disables tracking during weight updates to save memory. |
| **Zero Grad** | `w.grad.zero_()` | Clears old gradients so they don't add up in the next loop. |
| **Binarizing** | `(y > 0).float()` | Converts raw scores into 0s and 1s for classification. |

---
**Summary for Revision:**
1. Start with **Tensors**.
2. Learn **Math & Matrix** ops.
3. Use **Reductions** to analyze data.
4. Master **Preprocessing** (Numpy ↔ Torch).
5. Build the **Training Loop** (Forward → Loss → Backward → Update).
