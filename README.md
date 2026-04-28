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

---

## 📅 Day 5: PyTorch Built-in Modules (`nn` & `optim`)
*Transitioning from manual math to professional, high-level PyTorch components.*

### 🛠️ Part A: The `torch.nn` & `torchinfo` Toolbox
| Concept / Tool | Code Implementation | Deep Explanation (The "Why") |
| :--- | :--- | :--- |
| **`nn.Module`** | `class Model(nn.Module):` | **The Blueprint**: Inheriting this gives your class access to all PyTorch training features. |
| **`nn.Linear`** | `self.linear = nn.Linear(in, out)` | **The Layer**: Creates a layer with **Weights** and **Bias** automatically. |
| **`super().__init__()`** | `super().__init__()` | **The Link**: Mandatory link that initializes the parent `nn.Module` so PyTorch knows this is a model. |
| **Activation Layers** | `nn.ReLU()`, `nn.Sigmoid()` | **The Logic**: Built-in objects for non-linearity. Can be stored as class variables for reuse. |
| **The `forward` Pass** | `def forward(self, x):` | **The Execution**: This function defines exactly how data flows from input to output. |
| **Model Summary** | `summary(model, input_size)` | **The Auditor**: From `torchinfo`. Shows layers, output shapes, and total parameters. |
| **Parameters** | `model.parameters()` | **The Brain**: A list containing all trainable weights and biases in the entire network. |
| **Safety Check** | `if 'var' in locals():` | **Memory Guard**: Checks if a variable exists in `locals()` dictionary to prevent crashes in Colab. |
| **Features (`X`)** | `X_train` | **The Input**: The data you show the model (e.g., Age, Weight) to learn patterns. |
| **Labels (`y`)** | `y_train` | **The Answer Key**: The ground truth (correct answers) the model tries to match. |
| **Optimizers** | `optim.SGD(params, lr)` | **The Engine**: Professional tools that automatically update weights based on gradients. |
| **`optimizer.step()`** | `optimizer.step()` | **The Update**: The actual command that nudges weights in the right direction to reduce error. |
| **`zero_grad()`** | `optimizer.zero_grad()` | **Clean Slate**: Clears out old gradients before a new calculation starts. |

### ⚖️ Part B: The Concept of "Bias" ($b$)
| Metric | Detail | Detailed Explanation |
| :--- | :--- | :--- |
| **Definition** | Trainable parameter | An extra value added to the output of a neuron, independent of the input data ($x$). |
| **The Formula** | $y = xW + b$ | The complete linear equation for a neuron. |
| **Flexibility** | Shifting | Allows the model to move the "line" up, down, left, or right (not stuck at 0,0). |
| **The Offset** | Zero-input output | Even if $x=0$, the neuron can still output a value ($b$) to represent patterns. |
| **Analogy** | The Threshold | If $W$ is the "Importance" of features, $b$ is the "Starting Point" required to trigger the neuron. |

---

## 📅 Day 6: Datasets & DataLoaders (Detailed Reference)
*The professional way to feed data into models efficiently.*

### 🛠️ 1. Why we use Classes? (Manual Loading vs. PyTorch)
| Headache | Manual Feeding Issue | PyTorch Class Solution |
| :--- | :--- | :--- |
| **Interface** | No standard way to fetch data. | **`Dataset` Class**: Provides a consistent `__getitem__` interface. |
| **Preprocessing** | Hard to apply transforms on the fly. | **Transform Pipeline**: Supports automated cleaning/editing during loading. |
| **Order** | Risk of model memorizing patterns. | **Shuffling**: Automated randomization of indices every epoch. |
| **Performance** | CPU waits for GPU (Bottleneck). | **Parallelization**: `num_workers` load next batches in background. |

### 📦 2. Core Class Responsibilities
| Class | Role / Analogy | Key Responsibility |
| :--- | :--- | :--- |
| **`Dataset`** | **The Organized Storage** | Knowing how to fetch and prepare **one** individual item at a time. |
| **`DataLoader`**| **The Delivery Service** | Bundling items into **batches**, shuffling, and feeding them to the model. |

### 🧬 3. The Custom `Dataset` Blueprint
| Function | Implementation | Deep Explanation |
| :--- | :--- | :--- |
| **`__init__`** | `self.data = data` | **The Setup**: Where you read CSVs or store your tensors in memory initially. |
| **`__len__`** | `return len(self.data)` | **The Boundary**: Tells the loader the total count to avoid "Index Out of Range". |
| **`__getitem__`**| `return x, y` | **The Extractor**: Logic to fetch 1 sample and its label at a specific `index`. |

### ⚡ 4. Mini-Batch Training Strategies
| Strategy | Implementation | Advantage / Disadvantage |
| :--- | :--- | :--- |
| **Batch GD** | All data at once. | Stable but extremely slow and crashes computer memory (RAM). |
| **Stochastic GD**| 1 row at a time. | Fast but very "noisy" and the loss fluctuates wildly. |
| **Mini-Batch GD**| 32-64 samples. | **The Goldilocks**: High speed, stable learning, and memory efficient. |

### 🔄 5. The DataLoader Internal Workflow
| Step | Process | Responsibility |
| :--- | :--- | :--- |
| **1. Sampler** | Index Strategy | Decides which indices to pick (e.g., Random vs Sequential). |
| **2. Dispatch** | Worker Allocation | Sends index lists to multiple CPU Workers for background loading. |
| **3. Fetch** | `__getitem__` Call | Workers pull data from your Dataset for each index in the batch. |
| **4. Collate** | The "Glue" | Combines separate items into a single Batch Tensor. |
| **5. Yield** | training loop | The final batch is sent to the main process for model training. |

### 🎯 6. Sampler & Collate (The Hidden Helpers)
| Helper | Variant | Purpose |
| :--- | :--- | :--- |
| **Sampler** | **Sequential** | Fetches data in order (0, 1, 2...). Best for Evaluation/Testing. |
| **Sampler** | **Random** | Fetches data randomly. Best for Training. |
| **Collate** | `collate_fn` | Custom glue for variable-length data (like different sentence lengths). |

### ⚙️ 7. Full DataLoader Parameters Reference
| Parameter | Default | Deep Impact (The "Why") |
| :--- | :--- | :--- |
| **`batch_size`** | `1` | Affects training speed and the "smoothness" of gradient updates. |
| **`shuffle`** | `False` | Mixes data to ensure the model doesn't memorize the sequence. |
| **`num_workers`** | `0` | Uses multiple CPU cores to load next batches while the GPU trains. |
| **`pin_memory`** | `False` | Moves data to "locked" RAM for faster transfer to the GPU. |
| **`drop_last`** | `False` | Drops the final batch if it's smaller than the batch size. |

---

## 📅 Day 7: Building an Artificial Neural Network (ANN)
*A complete workflow of building, training, and evaluating a neural network using PyTorch.*

### 🚀 Part A: Data Preparation & Preprocessing
| Concept / Tool | Code Implementation | Detailed Explanation (The "Why") |
| :--- | :--- | :--- |
| **Dataset Loading** | `pd.read_csv('fmnist_small.csv')` | Loading the dataset (Fashion MNIST) into a Pandas DataFrame. |
| **Train-Test Split** | `train_test_split(X, y, test_size=0.2)` | Dividing data to keep 20% unseen for testing the model's actual performance. |
| **Data Scaling** | `X_train / 255.0` | Normalizing pixel values (0-255) to a 0-1 range for faster and more stable model training. |
| **Custom Dataset** | `class CustomDataset(Dataset):` | Building a custom Dataset class to properly manage features and labels in PyTorch. |
| **Type Conversion** | `torch.tensor(..., dtype=torch.long)` | Labels must be converted to `torch.long` (integers) for classification loss functions like CrossEntropy. |
| **DataLoader** | `DataLoader(..., batch_size=32, shuffle=True)` | Creating DataLoaders for both training (`shuffle=True`) and testing (`shuffle=False` as order doesn't matter for evaluation). |

### 🧠 Part B: Model Architecture & Training Setup
| Concept / Tool | Code Implementation | Detailed Explanation (The "Why") |
| :--- | :--- | :--- |
| **Model Blueprint** | `class MyNN(nn.Module):` | Inheriting from `nn.Module` to create the Neural Network structure. |
| **`nn.Sequential`** | `self.model = nn.Sequential(...)` | A clean container that automatically passes data sequentially through all the listed layers. |
| **Linear Layer** | `nn.Linear(in_features, out_features)` | A fully connected layer that transforms input dimensions into output dimensions. |
| **Softmax** | `nn.Softmax(dim=1)` | Converts the raw output of the final layer into readable probabilities (summing to 1). |
| **CrossEntropy Loss** | `nn.CrossEntropyLoss()` | The standard loss function used for Multi-Class Classification tasks. |
| **Epoch** | `for epoch in range(epochs):` | **Concept:** One Epoch is when the model sees the entire training dataset exactly once. |

### 🔄 Part C: The Training Loop (Step-by-Step)
| Step | Code Implementation | Detailed Explanation (The "Why") |
| :--- | :--- | :--- |
| **1. Forward Pass** | `outputs = model(batch_features)` | The model makes its best guess based on the current input batch. |
| **2. Calculate Loss** | `loss = criterion(outputs, batch_labels)` | Compares the model's guesses to the actual answers to measure the error. |
| **3. Clear Gradients** | `optimizer.zero_grad()` | Wipes the gradients from the previous batch to prevent them from piling up. |
| **4. Backward Pass** | `loss.backward()` | Calculates how much each weight contributed to the error (Backpropagation). |
| **5. Update Weights** | `optimizer.step()` | Adjusts the weights slightly to reduce the error for the next time. |
| **6. Track Loss** | `loss.item()` | Extracts the pure scalar number from the PyTorch loss tensor to track training progress. |

### 📊 Part D: Model Evaluation & Inference
| Concept / Tool | Code Implementation | Detailed Explanation (The "Why") |
| :--- | :--- | :--- |
| **Evaluation Mode** | `model.eval()` | Switches the model from Training Mode to Evaluation Mode. It deactivates Dropout and stops Batch Normalization updates. |
| **Disable Gradients**| `with torch.no_grad():` | Tells PyTorch to stop tracking operations for gradients, which saves memory and speeds up testing. |
| **Class Prediction** | `_, predicted = torch.max(outputs, 1)` | Finds the index of the highest probability in the output, which corresponds to the predicted class label. |
| **Accuracy Calculation**| `correct += (predicted == batch_labels).sum().item()`| Counts exactly how many predictions matched the ground truth in the current batch. |
| **Final Accuracy** | `100 * correct / total` | The percentage of completely correct predictions over the entire test dataset. |
| **Improve Accuracy** | GPU, Full Dataset, LR, Optimizer | Tweaks like using more data, changing the learning rate, or using a GPU can boost accuracy. |
