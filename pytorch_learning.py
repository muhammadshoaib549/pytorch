import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================
# 1. TENSOR BASICS & FUNDAMENTALS
# ==========================================
def learn_tensors():
    print("--- 1. Tensor Basics ---")
    # Creating tensors
    x = torch.tensor([1, 2, 3], dtype=torch.float32)
    y = torch.ones((2, 3))
    z = torch.rand((2, 3))
    
    print(f"Tensor x: {x}")
    print(f"Tensor shape: {z.shape}")
    
    # Operations
    result = torch.matmul(y, z.T) # Matrix multiplication
    print(f"Matmul result shape: {result.shape}\n")

# ==========================================
# 2. DATA PREPROCESSING (The PyTorch Way)
# ==========================================
class ShoppingDataset(Dataset):
    def __init__(self, csv_file):
        # Load data
        df = pd.read_csv(csv_file)
        
        # FIX: Handle missing values (NaNs) found in the dataset
        df = df.dropna(subset=['Age', 'Purchase Amount (USD)', 'Review Rating', 'Subscription Status'])
        
        # Simple task: Predict 'Subscription Status' (Yes/No)
        # Features: Age, Purchase Amount, Review Rating
        features = ['Age', 'Purchase Amount (USD)', 'Review Rating']
        self.X = df[features].values.astype(np.float32)
        
        # Label encoding for target
        le = LabelEncoder()
        self.y = le.fit_transform(df['Subscription Status']).astype(np.float32)
        
        # Normalize features (Crucial for Neural Networks)
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# ==========================================
# 3. NEURAL NETWORK DEFINITION
# ==========================================
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()
        # Layers
        self.layer1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        # Note: We removed nn.Sigmoid() because we'll use BCEWithLogitsLoss 
        # which is more numerically stable and includes the sigmoid inside.

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x) # Output raw numbers (logits)
        return x

# ==========================================
# 4. TRAINING PIPELINE
# ==========================================
def train_model():
    print("--- 2. Training Pipeline ---")
    # Setup
    dataset = ShoppingDataset('customer_shopping_behavior.csv')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = SimpleClassifier(input_dim=3)
    # BCEWithLogitsLoss = Sigmoid + BCELoss (More stable!)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Simple Training Loop
    epochs = 5
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            # 1. Forward pass
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            
            # 2. Backward pass & Optimization
            optimizer.zero_grad() # ALWAYS zero the gradients first
            loss.backward()       # Calculate gradients
            optimizer.step()      # Update weights
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

    # Save model
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Model saved to model_weights.pth\n")

if __name__ == "__main__":
    learn_tensors()
    train_model()
    print("PyTorch Learning Completed! Check README.md for the quick revision table.")
