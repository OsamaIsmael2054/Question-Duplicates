import torch
from torch import nn


class Network(nn.Module):
    def __init__(self,vocab_size, sequence_len=400, n_layers=1, embedding_dim =50, hidden_size = 128, Vout = 64, device="cpu") -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.embed = nn.Embedding(num_embeddings = vocab_size ,embedding_dim = embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.fc = nn.Linear(sequence_len * hidden_size, Vout)

    def forward(self,x):
        x = self.embed(x)

        hidden = (torch.zeros(self.n_layers, x.shape[1], self.hidden_size).to(self.device),
                  torch.zeros(self.n_layers, x.shape[1], self.hidden_size).to(self.device))

        output, _ = self.lstm(x,hidden)

        output = output.reshape(output.shape[0], -1)
        
        out = self.fc(output)
        return out