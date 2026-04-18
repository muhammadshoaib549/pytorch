# 🚀 PyTorch Ultimate Mastery Guide
*Your safe-haven for future revision. Look here first to remind yourself "what was what!"*

---

## 🗺️ 1. The Learning Roadmap (Where is what?)
If you forget a concept, find the corresponding file below:

| File Name | Concept | Quick Reminder |
| :--- | :--- | :--- |
| **`01_tensors.py`** | **Tensors** | The "Arrays" of Deep Learning. Multi-dimensional math. |
| **`02_autograd.py`** | **Autograd** | The magic behind how your model calculates errors (gradients). |
| **`03_vision_cnns.py`** | **CNNs** | How computers "see" images using layers that scan pixels. |
| **`04_sequences.py`** | **RNNs/LSTMs** | Handling data that has an order (Time, Sentences, Stock prices). |
| **`05_advanced.py`** | **Loss & Hooks** | Creating your own rules and spying on your model's soul. |
| **`Deep_Dive_...py`** | **Full Pipeline** | The professional way to train on your **Shopping CSV data**. |

---

## 🧠 2. "IDR YE KYA THA?" (What exactly is this?)
| Term | Simple Definition | PyTorch Code |
| :--- | :--- | :--- |
| **Tensor** | A container for numbers. Like a NumPy array but faster. | `torch.tensor([1,2,3])` |
| **Device** | Where the math happens (your CPU or your Graphics Card/GPU). | `tensor.to('cuda')` |
| **`nn.Module`** | The "Mother Class" for all networks. Every model must inherit this. | `class MyModel(nn.Module):` |
| **`forward()`** | The path your data takes to get a prediction. | `def forward(self, x):` |
| **Optimizer** | The "Teacher" that fixes the model when it's wrong. | `optim.Adam()` or `optim.SGD()` |
| **Loss** | The "Scoreboard" showing how far the model is from the truth. | `nn.CrossEntropyLoss()` |
| **`zero_grad()`** | Clearing the memory of past mistakes before learning again. | `optimizer.zero_grad()` |
| **`backward()`** | Using calculus to find out how to fix every single weight. | `loss.backward()` |
| **`step()`** | Actually updating the weights after they've been calculated. | `optimizer.step()` |
| **Batch Size** | How many rows of data the model sees at once. | `DataLoader(bs=32)` |

---

## 🛠️ 3. The 5-Step Pipeline (For your Shopping CSV)
When you look at `Deep_Dive_Shopping_Data.py`, remember these stages:

1.  **Stage 1: The Dataset**: Converting your CSV strings/numbers into Tensors.
2.  **Stage 2: The DataLoader**: Chopping your data into batches so your computer doesn't crash.
3.  **Stage 3: The Architecture**: Defining layers like `Linear` (Standard), `BatchNorm` (Stay stable), and `Dropout` (Don't memorize).
4.  **Stage 4: The Loop**: The cycle of *Predict -> Calculate Error -> Fixed Errors -> Repeat*.
5.  **Stage 5: Evaluation**: Using `model.eval()` to see if the model actually works on new data.

---

## ⚠️ 4. Crucial Reminders (Don't Forget!)
- **NaNs are Killers**: If your CSV has missing data, your model will break. Always use `df.dropna()`.
- **Modes Matter**: Use `model.train()` when learning, and `model.eval()` when testing.
- **Weights**: Always save just the "brain" using `torch.save(model.state_dict())`. It's lighter and safer!

---

## 📦 Environment Setup
If you move to a new machine, run this to get started:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --break-system-packages
```

**Now, go forth and master the AI world! 🚀**
