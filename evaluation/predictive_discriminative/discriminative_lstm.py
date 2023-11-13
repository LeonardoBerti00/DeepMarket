import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import random

# REFERENCE: Time-series Generative Adversarial Networks

################################### MERGE THE HISTORICAL AND GENERATED DATASETS ###################################
# ho fatto una prova con solo il dataset non generato e l'accuracy è ovviamente di circa il 50%
# 1 merge the two datasets
# 2 add a generated binary feature
###################################################################################################################

class Preprocessor:
    def __init__(self, df):
        self.df = df

    def preprocess(self):
        self._add_generated_feature()
        self._rename_columns()
        self._one_hot_encode()
        self._normalize()
        self._divide_time()
        return self.df

    def _add_generated_feature(self): # this function will be deleted since we add the generated feature during the merging
        random.seed(42)
        self.df['generated'] = [random.randint(0, 1) for _ in range(len(self.df))]

    def _rename_columns(self):
        self.df.columns = ["time", "event_type", "size", "price", "direction", "generated"]

    def _one_hot_encode(self):
        self.df = pd.get_dummies(self.df, columns=['direction', 'event_type'])

    def _normalize(self):
        self.df['price'] = (self.df['price'] - self.df['price'].min()) / (self.df['price'].max() - self.df['price'].min())
        self.df['size'] = (self.df['size'] - self.df['size'].min()) / (self.df['size'].max() - self.df['size'].min())

    def _divide_time(self):
        self.df['time'] = self.df['time'] / 100000


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])
        return out


class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, epochs):
        for epoch in range(epochs):
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.criterion(output, labels.unsqueeze(1))
                loss.backward()
                self.optimizer.step()

    def test(self):
        self.model.eval()
        test_preds = torch.Tensor().to(self.device)
        test_labels = torch.Tensor().to(self.device)
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                output = self.model(inputs)
                test_preds = torch.cat((test_preds, output), dim=0)
                test_labels = torch.cat((test_labels, labels.unsqueeze(1)), dim=0)
        test_preds = torch.sigmoid(test_preds)
        test_preds_binary = (test_preds > 0.5).float()
        accuracy = (test_preds_binary == test_labels).sum().item() / test_labels.numel()
        print(f'Test Accuracy: {accuracy * 100:.2f}%')