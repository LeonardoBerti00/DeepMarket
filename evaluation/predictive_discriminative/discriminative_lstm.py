import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import random


# REFERENCE: Time-series Generative Adversarial Networks

# merge gen and not gen datased adding a binary label column
# train a lstm model to predict the binary label
# test the model on the test set
# if the accuracy is high, the model is able to discriminate between generated and not generated data (the dataset are different) otherwise the model is not able to discriminate between
#    the two datasets (the dataset are similar)
# ho fatto una prova con solo il dataset non generato e l'accuracy è ovviamente di circa il 50% (TODELETE)


class Preprocessor:
    def __init__(self, df): # df in input is the merged dataset with the binary label "generated"
        self.df = df

    def preprocess(self):
        self._remove_columns()
        self._one_hot_encode()
        self.zscore()
        return self.df
    
    def zscore(self):
        self.df['PRICE'] = (self.df['PRICE'] - self.df['PRICE'].mean()) / self.df['PRICE'].std()
        self.df['SIZE'] = (self.df['SIZE'] - self.df['SIZE'].mean()) / self.df['SIZE'].std()
        self.df['ask_price_1'] = (self.df['ask_price_1'] - self.df['ask_price_1'].mean()) / self.df['ask_price_1'].std()
        self.df['ask_size_1'] = (self.df['ask_size_1'] - self.df['ask_size_1'].mean()) / self.df['ask_size_1'].std()
        self.df['bid_price_1'] = (self.df['bid_price_1'] - self.df['bid_price_1'].mean()) / self.df['bid_price_1'].std()
        self.df['bid_size_1'] = (self.df['bid_size_1'] - self.df['bid_size_1'].mean()) / self.df['bid_size_1'].std()
        self.df['ORDER_VOLUME_IMBALANCE'] = (self.df['ORDER_VOLUME_IMBALANCE'] - self.df['ORDER_VOLUME_IMBALANCE'].mean()) / self.df['ORDER_VOLUME_IMBALANCE'].std()
        self.df['VWAP'] = (self.df['VWAP'] - self.df['VWAP'].mean()) / self.df['VWAP'].std()
        self.df['MID_PRICE'] = (self.df['MID_PRICE'] - self.df['MID_PRICE'].mean()) / self.df['MID_PRICE'].std() # not in predictive

    def _one_hot_encode(self):
        self.df = pd.get_dummies(self.df, columns=['TYPE'])

    def _remove_columns(self):
        self.df = self.df.drop(['ORDER_ID', 'SPREAD'], axis=1)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device, non_blocking=True)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device, non_blocking=True)
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
        self.model.train()
        for epoch in range(epochs):
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.criterion(output, labels.unsqueeze(1))
                loss.backward()
                self.optimizer.step()

    def test(self):
        self.model.eval()
        test_preds = torch.Tensor().to(self.device, non_blocking=True)
        test_labels = torch.Tensor().to(self.device, non_blocking=True)
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                output = self.model(inputs)
                test_preds = torch.cat((test_preds, output), dim=0)
                test_labels = torch.cat((test_labels, labels.unsqueeze(1)), dim=0)
        test_preds = torch.sigmoid(test_preds)
        test_preds_binary = (test_preds > 0.5).float()
        accuracy = (test_preds_binary == test_labels).sum().item() / test_labels.numel()
        print(f'Test Accuracy: {accuracy * 100:.2f}%')

#################################################################################################################################################################################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def merge_dataframes_with_labels(d1, d2):
    d1['generated'] = 0
    d2['generated'] = 1
    merged_df = pd.concat([d1, d2])
    # shuffle the dataset
    merged_df = merged_df.sample(frac=1).reset_index(drop=True)
    return merged_df

df1 = pd.read_csv('data/TSLA/TSLA_2015-01-02_2015-01-30/TSLA_2015-01-02_34200000_57600000_message_10.csv') ######################### TO DELETE ######################### insert real data
df2 = pd.read_csv('data/TSLA/TSLA_2015-01-02_2015-01-30/TSLA_2015-01-02_34200000_57600000_message_10.csv') ######################### TO DELETE ######################### insert generated data

df = Preprocessor(merge_dataframes_with_labels(df1, df2)).preprocess()

# Assuming df is already preprocessed
features = df.drop('generated', axis=1).values
labels = df['generated'].values

# Reshape input to be 3D [samples, timesteps, features]
features = features.reshape((features.shape[0], 1, features.shape[1]))

# Split the data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
train_X = torch.tensor(train_X, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
test_X = torch.tensor(test_X, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32)

# Create data loaders
train_data = TensorDataset(train_X, train_y)
train_loader = DataLoader(train_data, batch_size=72)
test_data = TensorDataset(test_X, test_y)
test_loader = DataLoader(test_data, batch_size=72)

model = LSTMModel(input_size=train_X.shape[2], hidden_size=128, num_layers=2, output_size=1, device=device)
model.to(device)

trainer = Trainer(model=model, train_loader=train_loader, test_loader=test_loader, criterion=nn.BCEWithLogitsLoss(), optimizer=torch.optim.Adam(model.parameters(), lr=0.001), device=device)
trainer.train(epochs=100)
trainer.test()