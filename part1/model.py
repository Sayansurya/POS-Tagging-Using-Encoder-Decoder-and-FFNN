import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from torch import nn
from tqdm import tqdm
import torch.nn as nn
from tqdm import tqdm


class FFNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], output_dim),
        )
        self.model = self.model.to(self.device)

    def forward(self, X):
        y = self.model(X.type(torch.float32).to(self.device))
        return y

    def predict(self, batch):
        batch = batch.type(torch.float32).to(self.device)
        y = self.model(batch)
        prob = nn.Softmax(y)
        return prob.to('cpu')
