# 🗓️ Day 4: PyTorch Neural Networks & Data Preprocessing

This table covers **everything** from today's lecture (Day 4). The goal is to let you revise the entire workflow just by looking at this table, without opening the code.

### 1. Data Cleaning & Preprocessing (Sklearn to PyTorch)
| Concept | Code Implementation | Deep Explanation (The "Why") |
| :--- | :--- | :--- |
| **Data Loading** | `pd.read_csv(url)` | Imports raw data from a GitHub link or local file into a DataFrame. |
| **Cleaning Nulls**| `df.dropna()` or `df.fillna(df.mean())` | Removes empty rows/columns or fills them with averages to prevent calculation errors. |
| **Label Encoding**| `le.fit_transform(column)` | Converts text (e.g., "Scotland") into unique numbers (e.g., 5) so the AI can read it. |
| **Data Splitting**| `train_test_split(X, y, test_size=0.2)` | Breaks data into **Train** (to learn from) and **Test** (to check accuracy later). |
| **Feature Scaling**| `StandardScaler.fit_transform()` | Rescales all numbers (e.g., goals 0-5 and team IDs 0-300) into a similar range (mean 0, std 1). |
| **Numpy to Torch**| `torch.from_numpy(X_train).float()`| Converts standard numbers into **Tensors** so the GPU can process them. |
| **Reshaping (Target)**| `y_tensor.view(-1, 1)` | Changes shape from `[N]` to `[N, 1]` to match the model's output format. |

### 2. Manual Neural Network Architecture
| Component | Code Implementation | Deep Explanation (The "Why") |
| :--- | :--- | :--- |
| **Weights Init** | `torch.randn(inputs, 1, grad=True)` | Creates random starting numbers to multiply with inputs. `grad=True` tells PyTorch to track them. |
| **Bias Init** | `torch.randn(1, grad=True)` | Adds a "starting constant" to the output so the model can shift its threshold. |
| **Linear Layer** | `z = torch.mm(X, W) + B` | The core math: $Input \times Weight + Bias$. This is what finding the pattern looks like. |
| **Activation** | `torch.sigmoid(z)` | Squashes any number into a range between **0 and 1** (making it a probability). |
| **In-place ReLU** | `m.relu_()` | Turns negative numbers to `0` directly in memory without creating a copy (saves RAM). |

### 3. The Training Loop (The Pattern Search)
| Step | Code Implementation | Deep Explanation (The "Why") |
| :--- | :--- | :--- |
| **Forward Pass** | `y_probs = model.forward(X)` | Making a prediction based on current weights. |
| **Loss Function** | `-(y_true * log(y_pred) + ...)` | **Binary Cross Entropy**: Measures how "wrong" the model was. |
| **Log Stability** | `torch.clamp(y_pred, epsilon, 1-epsilon)`| Prevents `log(0)` which would break the math. Forces values to stay tiny above 0. |
| **Backward Pass** | `loss.backward()` | **The Magic**: PyTorch goes through the entire math backward to find who caused the error. |
| **Weight Update** | `with torch.no_grad(): w -= lr * w.grad` | Gradient Descent: Moves the weights slightly in the direction that reduces error. |
| **Zero Gradient** | `w.grad.zero_()` | Clears the old error calculation so it doesn't add up (accumulate) in the next round. |
| **No_Grad Context**| `with torch.no_grad():` | Disables tracking while we are updating weights—it's like telling PyTorch "don't watch me now." |

### 4. Advanced Evaluation
| Concept | Code Implementation | Deep Explanation (The "Why") |
| :--- | :--- | :--- |
| **Binarizing** | `(y_probs > 0.5).float()` | Converts a probability (e.g., 0.8) into a hard classification (e.g., 1.0). |
| **Accuracy** | `(preds == y_true).float().mean()` | Calculates what percentage of games the model guessed correctly. |
| **GPU Sync** | `torch.cuda.synchronize()` | Ensures the GPU has finished all tasks before the CPU starts timing the operation. |
| **Cloning** | `x.clone()` | Creates a deep copy of a tensor. Unlike simple `=` , updating the clone won't change the original. |

---

## 🏗️ Previous Concepts (Day 1-3)
*(Revision from the Tensors Fundamentals)*

| Operation | Code | Quick Note |
| :--- | :--- | :--- |
| **Matrix Mul** | `a @ b` or `torch.matmul` | Rows $\times$ Columns. |
| **Dot Product** | `torch.sum(m * n)` | Multiplying and then adding everything up. |
| **Flattening** | `x.flatten()` | Turning an image/matrix into a single line of numbers. |
| **Squeezing** | `x.squeeze()` | Removing useless "1" dimensions (e.g., `[1, 10] → [10]`). |
| **Permuting** | `x.permute(2, 0, 1)` | Moving rows/cols around (e.g., swapping Width and Height). |
| **Fixed Seed** | `torch.manual_seed(100)` | Ensures random numbers are the same every time for debugging. |

**Shoaib Bhai, ye table Day 4 ki poori revision hai! Ab aapko code dekhne ki zarurat nahi, bas ye steps dimaag mein rakhein.**
