import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
import random
import numpy as np
from tqdm import tqdm
import constants as cst

# REFERENCE: Time-series Generative Adversarial Networks

# merge gen and not gen datased adding a binary label column
# train a lstm model to predict the binary label 
# test the model on the test set
# if the accuracy is high, the model is able to discriminate between generated and not generated data (the dataset are different) otherwise the model is not able to discriminate between
#    the two datasets (the dataset are similar)
# ho fatto una prova con solo il dataset non generato e l'accuracy è ovviamente di circa il 50% (TODELETE)
class TDataset(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return len(self.data1) + len(self.data2) - 200
    
    def __getitem__(self, index):
        if index % 2 == 0:
            index = index // 2
            return self.data1[index:index+100], torch.zeros(1).to(cst.DEVICE)
        else:
            index = index // 2
            return self.data2[index:index+100], torch.ones(1).to(cst.DEVICE)


class Preprocessor:
    def __init__(self, df): # df in input is the merged dataset with the binary label "generated"
        self.df = df

    def preprocess(self):
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
        self.df['MID_PRICE'] = (self.df['MID_PRICE'] - self.df['MID_PRICE'].mean()) / self.df['MID_PRICE'].std() # not in predictive

    def _one_hot_encode(self):
        self.df = pd.get_dummies(self.df, columns=['TYPE'])
        self.df['TYPE_LIMIT_ORDER'] = self.df['TYPE_LIMIT_ORDER'].apply(lambda x: 1 if x == 'True' else 0)
        self.df['TYPE_ORDER_CANCELLED'] = self.df['TYPE_ORDER_CANCELLED'].apply(lambda x: 1 if x == 'True' else 0)
        self.df['TYPE_ORDER_EXECUTED'] = self.df['TYPE_ORDER_EXECUTED'].apply(lambda x: 1 if x == 'True' else 0)

    def _remove_columns(self):
        self.df = self.df.drop(['ORDER_ID', 'SPREAD', 'ORDER_VOLUME_IMBALANCE'], axis=1)
        self.df = self.df.drop(['Unnamed: 0'], axis=1)

    def _binarization(self):
        self.df['BUY_SELL_FLAG'] = self.df['BUY_SELL_FLAG'].apply(lambda x: 1 if x == 'True' else -1)

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
        self.model.train()
        last_loss = 1000000
        for epoch in tqdm(range(epochs)):
            losses = []
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            print(f'Epoch {epoch+1}, Loss: {np.mean(losses)}')
            if np.mean(losses) >= last_loss:
                break
            last_loss = np.mean(losses)


    def test(self):
        self.model.eval()
        test_preds = torch.Tensor().to(self.device, non_blocking=True)
        test_labels = torch.Tensor().to(self.device, non_blocking=True)
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                output = self.model(inputs)
                test_preds = torch.cat((test_preds, output), dim=0)
                test_labels = torch.cat((test_labels, labels), dim=0)
        test_preds = torch.sigmoid(test_preds)
        test_preds_binary = (test_preds > 0.5).float()
        total = (test_preds_binary.flatten() == test_labels.flatten()).sum().item()
        accuracy = total / test_labels.shape[0]
        print(f'Test Accuracy: {accuracy * 100:.2f}%')

#################################################################################################################################################################################################

def main(real_path, generated_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def merge_dataframes_with_labels(d1, d2):
        d1['generated'] = 0
        d2['generated'] = 1
        merged_df = pd.concat([d1, d2])
        # shuffle the dataset
        merged_df = merged_df.sample(frac=1).reset_index(drop=True)
        return merged_df

    df_r = pd.read_csv(real_path)
    df_g = pd.read_csv(generated_path)

    # remove the first 15 minutes of the generated dataset
    df_g["Time"] = df_g['Unnamed: 0'].str.slice(11, 19)
    df_g = df_g.query("Time >= '09:45:00'")
    df_g = df_g.drop(['Time'], axis=1)
    
    df_g = df_g.query("ask_price_1 < 9999999")
    df_g = df_g.query("bid_price_1 < 9999999")
    df_g = df_g.query("ask_price_1 > -9999999")
    df_g = df_g.query("bid_price_1 > -9999999")
    df_r = df_r.query("ask_price_1 < 9999999")
    df_r = df_r.query("bid_price_1 < 9999999")
    df_r = df_r.query("ask_price_1 > -9999999")
    df_r = df_r.query("bid_price_1 > -9999999")

    if len(df_r) > len(df_g):
        n_remove = len(df_r) - len(df_g)
        drop_indices = np.random.choice(df_r.index, n_remove, replace=False)
        df_r = df_r.drop(drop_indices)
    elif len(df_g) > len(df_r):
        n_remove = len(df_g) - len(df_r)
        drop_indices = np.random.choice(df_g.index, n_remove, replace=False)
        df_g = df_g.drop(drop_indices)

    df_r = Preprocessor(df_r).preprocess()
    df_g = Preprocessor(df_g).preprocess()

    # Split the data into training and test sets
    train_X_r, test_X_r = train_test_split(df_r.values, test_size=0.2, random_state=42)
    train_X_g, test_X_g = train_test_split(df_g.values, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    train_X_r = torch.tensor(train_X_r, dtype=torch.float32)
    test_X_r = torch.tensor(test_X_r, dtype=torch.float32)
    train_X_g = torch.tensor(train_X_g, dtype=torch.float32)
    test_X_g = torch.tensor(test_X_g, dtype=torch.float32)
    train_X_g = torch.concat((train_X_g[:, :7], train_X_g[:, -5:]), dim=1)
    test_X_g = torch.concat((test_X_g[:, :7], test_X_g[:, -5:]), dim=1)
    train_X_r = torch.concat((train_X_r[:, :7], train_X_r[:, -5:]), dim=1)
    test_X_r = torch.concat((test_X_r[:, :7], test_X_r[:, -5:]), dim=1)
    # Create data loaders
    train_data = TDataset(train_X_r, train_X_g)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_data = TDataset(test_X_r, test_X_g)
    test_loader = DataLoader(test_data, batch_size=64)

    model = LSTMModel(input_size=train_X_r.shape[1], hidden_size=128, num_layers=2, output_size=1)
    model.to(device)
    print("Discriminative score: ")
    trainer = Trainer(model=model, train_loader=train_loader, test_loader=test_loader, criterion=nn.BCEWithLogitsLoss(), optimizer=torch.optim.Adam(model.parameters(), lr=0.001), device=device)
    trainer.train(epochs=50)
    trainer.test()


if __name__ == '__main__':
    main()