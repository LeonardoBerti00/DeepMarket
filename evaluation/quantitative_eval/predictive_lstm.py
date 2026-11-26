import copy
import math
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import random
import numpy as np
from tqdm import tqdm

import constants as cst
from utils.utils import calculate_ftsd, plot, compute_prd_from_embedding, compute_prdc, IPR

# Predictive score protocol (Appendix of TRADES):
# a 2-layer LSTM sees the last SEQ_LEN market observations (last order + first LOB level)
# and forecasts the mid-price change HORIZON events ahead. One LSTM is trained on real data and
# one on generated data; both are tested on the same real test set (train on synthetic, test on real).
# All splits are chronological: rows are never shuffled before windowing.
# The LSTMs are trained with an L1 loss, consistent with the MAE metric. Since the mid-price rarely moves
# within HORIZON events, each MAE is also reported relative to the trivial "zero change" predictor
# (relative score < 1 means the model beats predicting no change).
SEQ_LEN = 100
HORIZON = 10
REAL_TEST_RATE = 0.2    # last 20% of the real day is the test set
VAL_RATE = 0.1          # last 10% of each training series is used for early stopping
PATIENCE = 5
MAX_EPOCHS = 100
BATCH_SIZE = 48
FEATURES = ['PRICE', 'SIZE', 'BUY_SELL_FLAG', 'ask_price_1', 'ask_size_1', 'bid_price_1', 'bid_size_1',
            'VWAP', 'TYPE_LIMIT_ORDER', 'TYPE_ORDER_CANCELLED', 'TYPE_ORDER_EXECUTED']
NUMERICAL_FEATURES = ['PRICE', 'SIZE', 'ask_price_1', 'ask_size_1', 'bid_price_1', 'bid_size_1', 'VWAP']


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def set_seed(seed: int = cst.SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class TDataset(Dataset):
    """Contiguous windows of SEQ_LEN observations; the target is the mid-price change
    HORIZON events after the last observation of the window: mid[t + HORIZON] - mid[t]."""
    def __init__(self, data, mid_price):
        self.data = data
        self.mid_price = mid_price

    def __len__(self):
        return len(self.data) - SEQ_LEN - HORIZON + 1

    def __getitem__(self, index):
        t = index + SEQ_LEN - 1     # last observation of the window
        return self.data[index:t + 1], self.mid_price[t + HORIZON] - self.mid_price[t]


class Preprocessor:
    """Encodes the categorical columns and z-scores the numerical ones.
    The z-score statistics are fitted on the training split only and reused on val/test."""
    def __init__(self, df):
        self.df = df

    def preprocess(self):
        self._check_inf()
        self._one_hot_encode()
        self._binarization()
        self.df['VWAP'] = self.df['VWAP'].fillna(0)
        return self.df

    def _one_hot_encode(self):
        types = self.df['TYPE']
        for t in ['LIMIT_ORDER', 'ORDER_CANCELLED', 'ORDER_EXECUTED']:
            self.df['TYPE_' + t] = (types == t).astype(np.float32)

    def _binarization(self):
        # the column can be parsed as bool or as a "True"/"TRUE" string depending on how the csv was written
        self.df['BUY_SELL_FLAG'] = (self.df['BUY_SELL_FLAG'].astype(str).str.lower() == 'true').astype(np.float32)

    def _check_inf(self):
        self.df = self.df[np.isfinite(self.df['MID_PRICE'])]


def fit_zscore(df):
    mean = df[NUMERICAL_FEATURES].mean()
    std = df[NUMERICAL_FEATURES].std().clip(lower=1e-5)
    return mean, std


def to_tensors(df, mean, std, device):
    features = df[FEATURES].copy()
    features[NUMERICAL_FEATURES] = (features[NUMERICAL_FEATURES] - mean) / std
    x = torch.tensor(features.values.astype(np.float32), dtype=torch.float32).to(device, non_blocking=True)
    mid = torch.tensor(df['MID_PRICE'].values, dtype=torch.float32).to(device, non_blocking=True)
    return x, mid


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
        out, (h, c) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, c[-1, :, :]


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.hidden_states = []

    def _evaluate(self, loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for inputs, labels in loader:
                output, _ = self.model(inputs)
                losses.append(self.criterion(output, labels.unsqueeze(1)).item())
        return np.mean(losses)

    def train(self, epochs):
        """Early stopping on the validation loss with patience PATIENCE; restores the best weights."""
        best_val_loss = float('inf')
        best_state = copy.deepcopy(self.model.state_dict())
        epochs_without_improvement = 0
        for epoch in tqdm(range(epochs)):
            self.model.train()
            losses = []
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                output, _ = self.model(inputs)
                loss = self.criterion(output, labels.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            val_loss = self._evaluate(self.val_loader)
            print(f'Epoch {epoch+1}, Train Loss: {np.mean(losses)}, Val Loss: {val_loss}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    break
        self.model.load_state_dict(best_state)

    def test(self):
        self.model.eval()
        test_preds = torch.Tensor().to(self.device, non_blocking=True)
        test_labels = torch.Tensor().to(self.device, non_blocking=True)
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                output, h = self.model(inputs)
                test_preds = torch.cat((test_preds, output), dim=0)
                test_labels = torch.cat((test_labels, labels), dim=0)
                self.hidden_states.append(h.cpu().numpy())
        mae = nn.functional.l1_loss(test_preds, test_labels.unsqueeze(1)).item()
        print(f'Test MAE: {mae}')
        return mae


def _drop_warmup(df):
    # remove the first 15 minutes of the simulation; skipped when the index does not hold full
    # timestamps (e.g. the released TRADES-LOB csv files, which already start at 09:45)
    index = df['Unnamed: 0'].astype(str)
    if not index.str.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}').all():
        return df
    return df[index.str.slice(11, 19) >= '09:45:00']


def _drop_invalid_prices(df):
    for col in ['ask_price_1', 'bid_price_1']:
        df = df[(df[col] < 9999999) & (df[col] > -9999999)]
    return df


def _make_trainer(train_df, val_df, test_df, device):
    """Fits the z-score on train_df, then builds an LSTM trainer whose test set is test_df normalized
    with the same statistics (the whole pipeline is fitted on the training data only)."""
    set_seed()
    mean, std = fit_zscore(train_df)
    loaders = []
    for df, shuffle in [(train_df, True), (val_df, False), (test_df, False)]:
        x, mid = to_tensors(df, mean, std, device)
        loaders.append(DataLoader(TDataset(x, mid), batch_size=BATCH_SIZE, shuffle=shuffle))
    model = LSTMModel(input_size=len(FEATURES), hidden_size=128, num_layers=2, output_size=1).to(device)
    return Trainer(model=model, train_loader=loaders[0], val_loader=loaders[1], test_loader=loaders[2],
                   criterion=nn.L1Loss(), optimizer=torch.optim.Adam(model.parameters(), lr=0.001), device=device)


def zero_change_mae(test_df):
    """MAE of always predicting a zero mid-price change, on the same windows used by TDataset."""
    mid = test_df['MID_PRICE'].values
    t = np.arange(SEQ_LEN - 1, len(mid) - HORIZON)
    return float(np.abs(mid[t + HORIZON] - mid[t]).mean())


def _chronological_split(df, rate):
    cut = int(len(df) * (1 - rate))
    return df.iloc[:cut], df.iloc[cut:]

#################################################################################################################################################################################################
def main(real_data_path, generated_data_path):
    print(generated_data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_r = pd.read_csv(real_data_path)
    df_g = pd.read_csv(generated_data_path)

    df_g = _drop_warmup(df_g)
    df_g = _drop_invalid_prices(df_g)
    df_r = _drop_invalid_prices(df_r)

    print("size of real data: ", len(df_r))
    print("size of generated data: ", len(df_g))
    df_r = Preprocessor(df_r).preprocess().dropna(subset=FEATURES + ['MID_PRICE'])
    df_g = Preprocessor(df_g).preprocess().dropna(subset=FEATURES + ['MID_PRICE'])

    # chronological splits, rows are never shuffled
    train_r, test_r = _chronological_split(df_r, REAL_TEST_RATE)
    train_r, val_r = _chronological_split(train_r, VAL_RATE)
    train_g, val_g = _chronological_split(df_g, VAL_RATE)

    ############ TEST "real" lstm on "real" test set ############
    print("\n Predictive Score Real data:")
    trainer_r = _make_trainer(train_r, val_r, test_r, device)
    trainer_r.train(epochs=MAX_EPOCHS)
    mae_r = trainer_r.test()

    ############ TEST "generated" lstm on "real" test set ############
    print("\n Predictive Score Generated data:")
    trainer_g = _make_trainer(train_g, val_g, test_r, device)
    trainer_g.train(epochs=MAX_EPOCHS)
    mae_g = trainer_g.test()
    '''
    #compute FTSD
    hidden_states_r = np.concatenate(trainer_r.hidden_states)
    hidden_states_g = np.concatenate(trainer_g.hidden_states)
    hidden_states_r = hidden_states_r[:hidden_states_g.shape[0]]
    print("FTSD: ", calculate_ftsd(hidden_states_r, hidden_states_g))
    dir_path = os.path.dirname(generated_data_path)
    out_path = os.path.join(dir_path, "prd.pdf")
    plot(compute_prd_from_embedding(hidden_states_g, hidden_states_r), out_path=out_path)
    ipr = IPR(num_samples=hidden_states_g.shape[0])
    ipr.compute_manifold_ref(hidden_states_r)
    precision, recall = ipr.precision_and_recall(hidden_states_g)
    print(f"Improved Precision {precision} and Recall {recall}")
    print("PRDC: ", compute_prdc(hidden_states_r, hidden_states_g, nearest_k=5))
    '''
    mae_zero = zero_change_mae(test_r)
    print(f"\n Zero-change MAE: {mae_zero}  relative real: {mae_r / mae_zero:.3f}  relative generated: {mae_g / mae_zero:.3f}")
    return {"real": mae_r, "generated": mae_g, "zero_change": mae_zero,
            "real_relative": mae_r / mae_zero, "generated_relative": mae_g / mae_zero}

if __name__ == '__main__':
    main()
