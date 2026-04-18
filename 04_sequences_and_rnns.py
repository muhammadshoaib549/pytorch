import torch
import torch.nn as nn

print("=== 04. Sequence Concept (RNNs/LSTMs) ===")

# Simulated Sequence Data: [Batch, Seq_Length, Features]
# example: 8 sentences, each 10 words long, each word represented by a 50-dim vector
sequences = torch.randn(8, 10, 50)
print(f"Sequence batch shape: {sequences.shape}")

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleLSTM, self).__init__()
        # batch_first=True means we use [Batch, Seq, Feature]
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # lstm_out: [Batch, Seq, Hidden]
        # (h_n, c_n): final hidden and cell states
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # We only take the last word's output for classification
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

model = SimpleLSTM(50, 64, 5)
output = model(sequences)
print(f"Output shape (8 sentences, 5 categories): {output.shape}\n")
