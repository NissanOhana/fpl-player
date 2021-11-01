from __future__ import annotations
# Genereal purpose
import sys
import os
from typing import Tuple, List
import datetime
import pickle
import glob
# Learning
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
# Data processing
import pandas as pd
import numpy as np
from scipy import stats
# Graphs
import seaborn as sns
from matplotlib import pyplot as plt
# Project specific
sys.path.append(os.getcwd())
from utils.fplVars import RANDOM_STATE
from datasets import FPLDatasetWrapper, get_flatten_dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############
# Custom RNNs #
###############
class Trainer:
    def __init__(self, model: FPLBaseDNN, loss_fn: function, optimizer: optim.optimizer, hps: dict):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.hps = hps
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):       
        self.model.train()              # Sets model to train mode
        yhat = self.model(x)            # Makes predictions
        loss = self.loss_fn(y, yhat)    # Computes loss
        loss.backward()                 # Computes gradients
        self.optimizer.step()           # Updates parameters
        self.optimizer.zero_grad()
        return loss.item()              # Returns the loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, batch_size: int = 32, n_epochs: int = 100, n_features: int = 1):
        model_path = os.path.join('TPnet', 'models', f'{self.model.model_name}-{datetime.datetime.now().strftime("%d_%m-%H_%M")}.pth')

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                # x_batch = x_batch.view([batch_size, -1, n_features])  # for RNN
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    # x_val = x_val.view([batch_size, -1, n_features])   # for RNN
                    x_val = x_val.to(DEVICE)
                    y_val = y_val.to(DEVICE)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if epoch % 5 == 0:
                print(f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}")

        torch.save(self.model.state_dict(), model_path)
        with open(model_path.replace('.pth', '_trainer.pkl'), 'wb') as f:
            pickle.dump(self, f)

    def evaluate(self, test_loader, batch_size=1, n_features=1) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                # x_test = x_test.view([batch_size, -1, n_features])    # for RNN
                x_test = x_test.to(DEVICE)
                y_test = y_test.to(DEVICE)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(DEVICE).detach().numpy())
                values.append(y_test.to(DEVICE).detach().numpy())

        return np.array(predictions), np.array(values)
    
    def plot_losses(self):
        figname = os.path.join('TPnet', 'models', f'{self.model.model_name}-{datetime.datetime.now().strftime("%d_%m-%H_%M")}')
        plt.plot(self.train_losses, label="Training")
        plt.plot(self.val_losses, label="Validation")
        plt.legend()
        plt.title(f"Losses for model {self.model.model_name}")
        plt.xlabel('Epoch')
        plt.ylabel(f'{self.loss_fn._get_name()}')
        plt.savefig(f"{figname}.png")
        plt.close()

    def format_predictions(self, predictions: np.ndarray, values: np.ndarray) -> pd.DataFrame:
        # FIXME not working when scale=False
        preds = np.concatenate(predictions, axis=0).ravel()
        vals = np.concatenate(values, axis=0).ravel()
        df_result = pd.DataFrame(data={"value": vals, "prediction": preds})
        # df_result = df_result.sort_index()
        df_result = self._inverse_transform(df_result, [["value", "prediction"]])
        return df_result
    
    def _inverse_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        if self.model.ds_wrapper.scaler:
            for col in columns:
                df[col] = self.model.ds_wrapper.scaler.inverse_transform(df[col])
        return df


class FPLBaseDNN(torch.nn.Module):
    def __init__(self, model_name: str, in_dim: int, out_dim: int, dropout: float, ds_wrapper: FPLDatasetWrapper) -> None:
        super(FPLBaseDNN, self).__init__()
        self.model_name : str               = model_name
        self.in_dim     : int               = in_dim
        self.out_dim    : int               = out_dim
        self.dropout    : float             = dropout
        self.ds_wrapper : FPLDatasetWrapper = ds_wrapper

        # NeuralNetwork architecture; each class should implement its own arch
        self.dnn = None   # TODO: batch_first=True
        self.fc  = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.layer_dims[0], x.size(0), self.hidden_dims[0]).requires_grad_()
        # Apply RNN layers
        out, h_last = self.dnn(x, h0)
        out = out[:, -1, :] # reshaping to (batch_size, seq_length, hidden_size)
        # Apply FC layer
        out = self.fc(out)

        return out  # (batch_size, out_size)
    
    def get_datasets(self, load_ds: bool, merged: bool, scaled: bool) -> None:
        ds_pickled = {
            (True, True): 'datasets_merged_scaled.pkl',
            (True, False): 'datasets_merged.pkl',
            (False, True): 'datasets_scaled.pkl',
            (False, False): 'datasets.pkl',
        }[(merged, scaled)]
        ds_pickled = 'datasets_merged.pkl' if merged else 'datasets.pkl'
        if load_ds:
            with open(os.path.join('TPnet', 'models', ds_pickled), 'rb') as f:
                datasets = pickle.load(f)
        else:
            with open(os.path.join('TPnet', 'models', ds_pickled), 'wb') as f:
                datasets = self.ds_wrapper.get_ds()
                pickle.dump(datasets, f)
        self.datasets = datasets


class FPLVanillaRNN(FPLBaseDNN):
    def __init__(self, in_dim: int, out_dim: int, ds_wrapper: FPLDatasetWrapper) -> None:
        super(FPLVanillaRNN, self).__init__('VanillaRNN', in_dim, out_dim, dropout=0.2, ds_wrapper=ds_wrapper)
        self.hidden_dims = [16]
        self.layer_dims = [128]
        self.dnn : torch.nn.RNN = torch.nn.RNN(self.in_dim, self.hidden_dims[0], self.layer_dims[0], dropout=self.dropout, batch_first=True)
        self.fc  : torch.nn.Linear = torch.nn.Linear(self.hidden_dims[-1], self.out_dim)


class FPLLSTM(FPLBaseDNN):
    def __init__(self, in_dim: int, hidden_dims: int, layer_dims: int, out_dim: int, dropout: float, ds_wrapper: FPLDatasetWrapper) -> None:
        # TODO make it work
        assert len(hidden_dims) == 1 and len(layer_dims) == 1
        super(FPLLSTM, self).__init__(in_dim, hidden_dims, layer_dims, out_dim, dropout, ds_wrapper)

        self.model_name = 'LSTM'
        self.dnn : torch.nn.Sequential(
            torch.nn.LSTM(in_dim, hidden_dims[0], layer_dims[0], batch_first=True, dropout=dropout),
            torch.nn.BatchNorm2d(layer_dims[0])
        )
        # self.dnn : torch.nn.RNN = torch.nn.RNN(in_dim, hidden_dims[0], layer_dims[0], dropout=dropout, batch_first=True)
        self.fc  : torch.nn.Linear = torch.nn.Linear(self.hidden_dims[-1], out_dim)

class FPLLinear(FPLBaseDNN):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, ds_wrapper: FPLDatasetWrapper) -> None:
        super(FPLLinear, self).__init__('LinearCNN', in_dim, out_dim, dropout, ds_wrapper)
        self.hidden_dims = [16]
        self.layer_dims = [128]
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(self.in_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(256, self.out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dnn(x)
        return out


def get_hyperparams(merged: bool, gws_to_predict: int):
    return {
        'merged': merged,
        'gws_to_predict': gws_to_predict,
        'epochs': 60,
        'model_params': {
            'in_dim': 86 if merged else 33,
            'out_dim': gws_to_predict,
            'dropout': 0.2
        },
        'opt_params': {
            'lr': 1e-7,
            'weight_decay': 5e-3
        },
        'dl_params': {
            'batch_size': 8,
            'shuffle': False,
            'drop_last': True   # TODO why?
        }
    }

def plot_kde(df: pd.DataFrame, model_name: str) -> None:
    figname = os.path.join('TPnet', 'models', f'{model_name}_kde-{datetime.datetime.now().strftime("%d_%m-%H_%M")}')
    std = 3
    df['delta'] = df['prediction'] - df['value']
    abs_z_scores = np.abs(stats.zscore(df['delta']))    # TODO DOH
    df = df[abs_z_scores <= std]
    rp = sns.displot(data=df, kind='kde', x='delta')
    rp.fig.subplots_adjust(top=0.98)
    rp.fig.suptitle(f'KDE of total score prediction\nModel: {model_name}; Diff up to {std} STD')
    # plt.legend()
    plt.xlabel('Predicted minus Real')
    plt.ylabel('Density')
    plt.savefig(f'{figname}.png')

def custom_cnn():
    merged = False
    gws_to_predict = 1
    load_ds = False
    load_last_model = False
    scaled = False
    model_type = 'Linear'
    
    hps = get_hyperparams(merged, gws_to_predict)
    if load_last_model:
        files = sorted(filter(os.path.isfile, glob.glob(os.path.join(os.getcwd(), 'TPnet', 'models', '*'))), key=os.path.getmtime)
        last_model = [f for f in files if f.endswith('_trainer.pkl')][-1]
        with open(last_model, 'rb') as f:
            trainer = pickle.load(f)
    else:
        ds_wrapper = FPLDatasetWrapper(gws_to_predict, merged, scaled)
        
        model_obj = {
            'VanillaRNN': FPLVanillaRNN,
            'LSTM': FPLLSTM,
            'Linear': FPLLinear
        }[model_type]
        
        model      = model_obj(**hps['model_params'], ds_wrapper=ds_wrapper)
        model.get_datasets(load_ds, hps['merged'], scaled)  # Get train, validate and test datasets)
        loss_fn   = torch.nn.MSELoss(reduction='mean')    # TODO manual loss function
        optimizer = optim.Adam(model.parameters(), **hps['opt_params'])
        trainer   = Trainer(model, loss_fn, optimizer, hps=hps)
        trainer.train(
            train_loader=DataLoader(model.datasets['train'], **hps['dl_params']),
            val_loader=DataLoader(model.datasets['validate'], **hps['dl_params']),
            batch_size=hps['dl_params']['batch_size'],
            n_epochs=hps['epochs'],
            n_features=hps['model_params']['in_dim']
        )
    trainer.plot_losses()

    predictions, values = trainer.evaluate(DataLoader(trainer.model.datasets['test']), batch_size=1, n_features=hps['model_params']['in_dim'])
    df_results = trainer.format_predictions(predictions, values)
    plot_kde(df_results, trainer.model.model_name)

#######
# MLP #
#######
LAYERS = {1: [32],
          2: [32, 64],
          3: [32, 64, 32],
          4: [32, 64, 128],
          5: [32, 64, 128, 64]
}
def sklearn_cnn(gws_to_predict: int = 1, shrink_zeros_ratio: float = 0):
    scaler = MinMaxScaler()
    ds_wrapper = FPLDatasetWrapper(gws_to_predict, merged=True, scaled=True, model_type='MLP')
    ds = load_datasets(merged=True, gws_to_predict=gws_to_predict, load_ds=True, ds_wrapper=ds_wrapper)
    X_train, y_train = get_flatten_dataset(ds['train'])
    if shrink_zeros_ratio:   # cut 0 scored events
            zeros = np.delete(np.arange(len(X_train)), np.nonzero(y_train))
            idx_zeros_to_delete = np.random.choice(zeros, size=(int(len(zeros)*shrink_zeros_ratio)), replace=False)
            idx_reduced_zeros = np.delete(np.arange(len(y_train)), idx_zeros_to_delete )
            X_train, y_train = X_train[idx_reduced_zeros], y_train[idx_reduced_zeros]
    X_validate, y_validate = get_flatten_dataset(ds['validate'])
    X_train = scaler.fit_transform(X_train)
    X_validate = scaler.transform(X_validate)

    cache_file = os.path.join('TPnet', 'models', 'DNNs', 'mlp_results.pkl')
    if not os.path.exists(cache_file):
        epochs = 300
        results = {}
        for merged in [True, False]:
            results[merged] = {}
            for layers in LAYERS:
                results[merged][layers] = {}
                for lr in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                    print(f'Working on layer code {layers}, LR: {lr}')
                    model = MLPRegressor(hidden_layer_sizes=LAYERS[layers] + [gws_to_predict],
                                        activation='relu',
                                        solver='adam',
                                        alpha=1e-2,
                                        batch_size=16,
                                        learning_rate_init=lr,
                                        max_iter=epochs,
                                        random_state=RANDOM_STATE)
                    trained_model = model.fit(X_train, y_train)
                    pred = trained_model.predict(X_validate)
                    score = trained_model.score(X_validate, y_validate)
                    results[merged][layers][lr] = {
                        'pred': pred,
                        'delta': pred - y_validate,
                        'loss': trained_model.loss_curve_
                    }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
    else:
        with open(cache_file, 'rb') as f:
            results = pickle.load(f)
    
    for only_scored in [True, False]:
        _plot_kde(results, y_real=y_validate, gws_to_predict=gws_to_predict, only_scored=only_scored)
    pass


def load_datasets(merged: bool, gws_to_predict: int, load_ds: bool, ds_wrapper) -> None:
    ds_pickled = os.path.join('TPnet', 'models', 'DNNs', f'datasets_merged_{gws_to_predict}.pkl' if merged else f'datasets_{gws_to_predict}.pkl')
    if load_ds and os.path.exists(ds_pickled):
        with open(ds_pickled, 'rb') as f:
            datasets = pickle.load(f)
    else:
        with open(ds_pickled, 'wb') as f:
            datasets = ds_wrapper.get_ds()
            pickle.dump(datasets, f)
    return datasets

def _plot_kde(results: dict, y_real, gws_to_predict: int = 1, only_scored: bool = True):
    players = 'OnlyScoredPlayers' if only_scored else 'AllPlayers'
    for merged in results:
        ds_name = 'WithPrevSeason' if merged else 'WithoutPrevSeason'
        for layer_code in results[merged]:
            df = pd.DataFrame(columns=['lr', 'delta', 'MAE'])
            for lr in results[merged][layer_code]:
                y_pred = results[merged][layer_code][lr]['pred']
                idx = [idx for idx,score in enumerate(y_real) if np.all(score)] if only_scored else range(len(y_real))
                mae = [mean_absolute_error(y_pred[idx][gw], y_real[idx][gw]) for gw in range(gws_to_predict)]
                subset = [{'LR': lr, 'delta': pred - real, 'MAE': mae} for (pred, real) in zip(y_pred[idx, :], y_real[idx, :])]
                df = df.append(subset, ignore_index=True)

            for i in range(gws_to_predict):
                tmp_df = df.copy(deep=True)
                if gws_to_predict > 1:
                    tmp_df['MAE'] = df.apply(lambda row: row['MAE'][i], axis=1)
                    tmp_df['delta'] = df.apply(lambda row: row['delta'][i], axis=1) + np.random.normal(1, 7)
                tmp_df['LR'] = tmp_df.apply(lambda row: f"LR:{row['LR']}", axis=1)
                rp = sns.displot(data=tmp_df, kind='kde', x='delta', hue='LR', palette='hsv')
                rp.fig.subplots_adjust(top=0.98)
                rp.fig.suptitle(f'KDE by LR values, layers={LAYERS[layer_code] + [gws_to_predict]}\nGWs: {i+1}/{gws_to_predict}, Dataset: {ds_name}\n{players}')
                plt.legend()
                plt.xlabel('Delta')
                plt.ylabel('Density')
                folder = os.path.join('TPnet', 'models', 'DNNs', ds_name, players)
                plt.savefig(os.path.join(folder, f'kde-layercode_{layer_code}-gws_{i+1}_outof_{gws_to_predict}.png'))

if __name__ == '__main__':
    print("This file should not be runned directly")
    # sklearn_cnn(gws_to_predict=3)
