import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# =================================================================
# 1. ADVANCED DATA PREPROCESSING
# =================================================================
# This class handles everything from loading to encoding and scaling
class ShoppingDataMaster(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        
        # We will predict 'Category' (Multi-class Classification)
        self.target_col = 'Category'
        
        # Features to use (Mixing Numerical and Categorical)
        self.num_features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
        self.cat_features = ['Gender', 'Location', 'Size', 'Color', 'Season', 'Subscription Status']
        
        # FIX: Handle missing values (NaNs) found in the dataset
        df = df.dropna(subset=self.num_features + self.cat_features + [self.target_col])
        
        # Encode Categorical Features
        self.label_encoders = {}
        for col in self.cat_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            
        # Encode Target
        self.target_encoder = LabelEncoder()
        df[self.target_col] = self.target_encoder.fit_transform(df[self.target_col])
        self.num_classes = len(self.target_encoder.classes_)
        
        # Combine all features
        X = df[self.num_features + self.cat_features].values.astype(np.float32)
        y = df[self.target_col].values.astype(np.int64) # LongTensor for CrossEntropy
        
        # Normalization (Standardization)
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(X)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Concept: Returning Tensors directly
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# =================================================================
# 2. FLEXIBLE NEURAL NETWORK ARCHITECTURE
# =================================================================
class AdvancedClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AdvancedClassifier, self).__init__()
        
        # Concept: Using nn.Sequential for modular blocks
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Concept: Batch Normalization
            nn.ReLU(),
            nn.Dropout(0.2),            # Concept: Dropout for Regularization
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Concept: Custom weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight) # Concept: He Initialization
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Concept: Functional API can also be used here (e.g., F.relu)
        return self.feature_extractor(x)

# =================================================================
# 3. COMPREHENSIVE TRAINING & EVALUATION SYSTEM
# =================================================================
def run_deep_learning_experiment():
    # 1. Load Data
    full_dataset = ShoppingDataMaster('customer_shopping_behavior.csv')
    
    # 2. Split Data (Training vs Validation) - Concept: random_split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    # 3. Initialize Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedClassifier(input_dim=10, hidden_dim=64, output_dim=full_dataset.num_classes).to(device)
    
    # Concept: CrossEntropyLoss combines LogSoftmax + NLLLoss
    criterion = nn.CrossEntropyLoss()
    
    # Concept: Weight Decay (L2 Regularization) in Adam
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Concept: Learning Rate Scheduler (Reduces LR when loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    print(f"Starting Training on {device}...")

    # 4. Training Loop
    epochs = 10
    for epoch in range(epochs):
        model.train() # Set to training mode
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Step-by-step optimization
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # 5. Validation Loop
        model.eval() # Set to evaluation mode (disables Dropout/BatchNorm)
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad(): # Concept: Disable gradient tracking for speed/memory
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1) # Get predicted class index
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_val.cpu().numpy())

        avg_val_loss = val_loss/len(val_loader)
        scheduler.step(avg_val_loss) # Update LR based on validation loss
        
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f}")

    # 6. Save Final Model - Concept: Saving state_dict is better than full model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'target_classes': full_dataset.target_encoder.classes_
    }, 'shopping_master_model.pth')
    
    print("\nTraining Complete. Model Saved!")
    print("\nSample Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=full_dataset.target_encoder.classes_))

if __name__ == "__main__":
    run_deep_learning_experiment()
