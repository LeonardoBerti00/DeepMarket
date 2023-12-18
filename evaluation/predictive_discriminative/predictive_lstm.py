import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import random

# given real data and generated data
# train a lstm with real data, train a lstm with generated data
# test the two lstm on real data test set


class Preprocessor:
    def __init__(self, df): 
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
                test_labels = torch.cat((test_labels, labels), dim=0)
        mse = nn.functional.mse_loss(test_preds, test_labels.unsqueeze(1)).item()
        print(f'Test MSE: {mse}')

#################################################################################################################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_r = pd.read_csv('data/merged.csv') ######################### TO DELETE ######################### insert real data
df_g = pd.read_csv('data/merged.csv') ######################### TO DELETE ######################### insert generated data
# r'C:\Users\marco\OneDrive\Documenti\AFC\afc_project\Diffusion-Models-for-Time-Series\data\TSLA\TSLA_2015-01-02_2015-01-30\TSLA_2015-01-02_34200000_57600000_message_10.csv'

df_r = Preprocessor(df_r).preprocess()
df_g = Preprocessor(df_g).preprocess()

############ TEST "real" lstm on "real" test set ############

# Assuming df is already preprocessed
features_r = df_r.drop('MID_PRICE', axis=1).values
labels_r = df_r['MID_PRICE'].values

# Reshape input to be 3D [samples, timesteps, features]
features_r = features_r.reshape((features_r.shape[0], 1, features_r.shape[1]))

# Split the data into training and test sets
train_X_r, test_X_r, train_y_r, test_y_r = train_test_split(features_r, labels_r, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
train_X_r = torch.tensor(train_X_r, dtype=torch.float32)
train_y_r = torch.tensor(train_y_r, dtype=torch.float32)
test_X_r = torch.tensor(test_X_r, dtype=torch.float32)
test_y_r = torch.tensor(test_y_r, dtype=torch.float32)

# Create data loaders
train_data_r = TensorDataset(train_X_r, train_y_r)
train_loader_r = DataLoader(train_data_r, batch_size=72)
test_data_r = TensorDataset(test_X_r, test_y_r)
test_loader_r = DataLoader(test_data_r, batch_size=72)

model_r = LSTMModel(input_size=train_X_r.shape[2], hidden_size=128, num_layers=2, output_size=1)
model_r.to(device)

trainer_r = Trainer(model=model_r, train_loader=train_loader_r, test_loader=test_loader_r, criterion=nn.MSELoss(), optimizer=torch.optim.Adam(model_r.parameters(), lr=0.001), device=device)
trainer_r.train(epochs=100)
print("Real data:")
trainer_r.test()

############ TEST "generated" lstm on "real" test set ############

# Assuming df is already preprocessed
features_g = df_g.drop('MID_PRICE', axis=1).values
labels_g = df_g['MID_PRICE'].values

# Reshape input to be 3D [samples, timesteps, features]
features_g = features_g.reshape((features_g.shape[0], 1, features_g.shape[1]))

# Split the data into training and test sets
train_X_g, test_X_g, train_y_g, test_y_g = train_test_split(features_g, labels_g, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
train_X_g = torch.tensor(train_X_g, dtype=torch.float32)
train_y_g = torch.tensor(train_y_g, dtype=torch.float32)
test_X_g = torch.tensor(test_X_g, dtype=torch.float32)
test_y_g = torch.tensor(test_y_g, dtype=torch.float32)

# Create data loaders
train_data_g = TensorDataset(train_X_g, train_y_g)
train_loader_g = DataLoader(train_data_g, batch_size=72)
test_data_g = TensorDataset(test_X_g, test_y_g)
test_loader_g = DataLoader(test_data_g, batch_size=72)

model_g = LSTMModel(input_size=train_X_g.shape[2], hidden_size=128, num_layers=2, output_size=1)
model_g.to(device)

trainer_g = Trainer(model=model_g, train_loader=train_loader_g, test_loader=test_loader_r, criterion=nn.MSELoss(), optimizer=torch.optim.Adam(model_g.parameters(), lr=0.001), device=device)
trainer_g.train(epochs=100)
print("Generated data:")
trainer_g.test()