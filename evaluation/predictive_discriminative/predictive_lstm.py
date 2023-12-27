import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import random
import numpy as np
import constants as cst

# given real data and generated data
# train a lstm with real data, train a lstm with generated data
# test the two lstm on real data test set


class Preprocessor:
    def __init__(self, df): 
        self.df = df

    def preprocess(self):
        self._check_inf()
        self._remove_columns()
        self._one_hot_encode()
        self.zscore()
        self._binarization()
        return self.df
    
    def zscore(self):
        self.df['PRICE'] = (self.df['PRICE'] - self.df['PRICE'].mean()) / self.df['PRICE'].std()
        self.df['SIZE'] = (self.df['SIZE'] - self.df['SIZE'].mean()) / self.df['SIZE'].std()
        self.df['ask_price_1'] = (self.df['ask_price_1'] - self.df['ask_price_1'].mean()) / self.df['ask_price_1'].std()
        self.df['ask_size_1'] = (self.df['ask_size_1'] - self.df['ask_size_1'].mean()) / self.df['ask_size_1'].std()
        self.df['bid_price_1'] = (self.df['bid_price_1'] - self.df['bid_price_1'].mean()) / self.df['bid_price_1'].std()
        self.df['bid_size_1'] = (self.df['bid_size_1'] - self.df['bid_size_1'].mean()) / self.df['bid_size_1'].std()
        #self.df['ORDER_VOLUME_IMBALANCE'] = (self.df['ORDER_VOLUME_IMBALANCE'] - self.df['ORDER_VOLUME_IMBALANCE'].mean()) / self.df['ORDER_VOLUME_IMBALANCE'].std()
        self.df['VWAP'] = self.df['VWAP'].fillna(0)
        self.df['VWAP'] = (self.df['VWAP'] - self.df['VWAP'].mean()) / self.df['VWAP'].std()

    def _one_hot_encode(self):
        self.df = pd.get_dummies(self.df, columns=['TYPE'])

    def _remove_columns(self):
        self.df = self.df.drop(['ORDER_ID', 'SPREAD', 'ORDER_VOLUME_IMBALANCE'], axis=1)
        self.df = self.df.drop(['Unnamed: 0'], axis=1)

    def _binarization(self):
        self.df['BUY_SELL_FLAG'] = self.df['BUY_SELL_FLAG'].apply(lambda x: 1 if x == 'True' else 0)

    def _check_inf(self):
        #self.df[self.df.ORDER_VOLUME_IMBALANCE != np.inf]
        self.df[self.df.MID_PRICE != np.inf]


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
                #print(inputs)
                output = self.model(inputs)
                test_preds = torch.cat((test_preds, output), dim=0)
                test_labels = torch.cat((test_labels, labels), dim=0)
                #print("Predicted Values:", output)
        mse = nn.functional.mse_loss(test_preds, test_labels.unsqueeze(1)).item()
        print(f'Test MSE: {mse}')
        #mae = nn.functional.l1_loss(test_preds, test_labels.unsqueeze(1)).item()
        #print(f'Test MAE: {mae}')

#################################################################################################################################################################################################
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_r = pd.read_csv(cst.REAL_PATH)
    df_g = pd.read_csv(cst.GENERATED_PATH)

    # remove the first 15 minutes of the generated dataset
    df_g["Time"] = df_g['Unnamed: 0'].str.slice(11, 19)
    df_g = df_g.query("Time >= '09:45:00'")
    df_g = df_g.drop(['Time'], axis=1)

    # undersampling on the real dataset
    n_remove = len(df_r) - len(df_g)
    drop_indices = np.random.choice(df_r.index, n_remove, replace=False)
    df_r = df_r.drop(drop_indices)


    df_r = Preprocessor(df_r).preprocess()
    df_g = Preprocessor(df_g).preprocess()

    '''
    # Check for NaN, null, inf, and missing values in df_g
    nan_columns = df_g.columns[df_g.isna().any()].tolist()
    null_columns = df_g.columns[df_g.isnull().any()].tolist()
    inf_columns = df_g.columns[(df_g == np.inf).any()].tolist()
    missing_columns = df_g.columns[df_g.isin([np.nan, np.inf, -np.inf, None]).any()].tolist()

    # Print the columns with NaN values
    print("Columns with NaN values:", nan_columns)

    # Print the columns with null values
    print("Columns with null values:", null_columns)

    # Print the columns with inf values
    print("Columns with inf values:", inf_columns)

    # Print the columns with missing values (NaN, null, inf)
    print("Columns with missing values:", missing_columns)


    # stampa colonne di df_r e df_g e i loro tipi
    # print(df_r.columns, df_g.columns)
    '''

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
    train_loader_r = DataLoader(train_data_r, batch_size=48)
    test_data_r = TensorDataset(test_X_r, test_y_r)
    test_loader_r = DataLoader(test_data_r, batch_size=48)

    model_r = LSTMModel(input_size=train_X_r.shape[2], hidden_size=128, num_layers=2, output_size=1)
    model_r.to(device)

    trainer_r = Trainer(model=model_r, train_loader=train_loader_r, test_loader=test_loader_r, criterion=nn.MSELoss(), optimizer=torch.optim.Adam(model_r.parameters(), lr=0.001), device=device)
    trainer_r.train(epochs=300)
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
    train_loader_g = DataLoader(train_data_g, batch_size=48)
    test_data_g = TensorDataset(test_X_g, test_y_g)
    test_loader_g = DataLoader(test_data_g, batch_size=48)

    model_g = LSTMModel(input_size=train_X_g.shape[2], hidden_size=128, num_layers=2, output_size=1)
    model_g.to(device)

    trainer_g = Trainer(model=model_g, train_loader=train_loader_g, test_loader=test_loader_r, criterion=nn.MSELoss(), optimizer=torch.optim.Adam(model_g.parameters(), lr=0.001), device=device)
    trainer_g.train(epochs=300)
    print("Generated data:")
    trainer_g.test()

if __name__ == '__main__':
    main()
