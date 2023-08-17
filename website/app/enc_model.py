import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, encoder_input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_dim = encoder_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim,
                            self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.lstm = self.lstm.to(self.device)
        self.linear = self.linear.to(self.device)

    def forward(self, X):
        encoder_output, _ = self.lstm(X)
        encoder_output = self.linear(encoder_output)
        return encoder_output
