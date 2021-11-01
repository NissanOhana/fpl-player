# General porpuse
import sys
import os
# Data processing
import numpy as np
import pandas as pd
# Learning
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import MinMaxScaler
# Project specific
sys.path.append(os.getcwd())
from utils.fplVars import DIDNT_PLAY_VALUE, ENCODING, DATASET_DICT, GWS_TO_PREDICT, GWS_PER_SEASON


class FPLDatasetWrapper:
    def __init__(self, gws_to_predict: int = 1, merged: bool = True, scaled: bool = True, model_type: str = 'RNN') -> None:
        self.gws_to_predict : int           = gws_to_predict
        self.merged         : bool          = merged
        self.scaler         : MinMaxScaler  = MinMaxScaler() if scaled else None
        self.model_type     : str           = model_type

    def get_ds(self):   # train/valid/test -> gw
        X_dict_scaled, y_dict_scaled, ds = {}, {}, {}
        for mode in ['train', 'validate', 'test']:
            X_dict, X_dict_scaled[mode] = {}, {}
            y_dict, y_dict_scaled[mode] = {}, {}
            ds[mode] = {}
            ds_name = f'{mode}_merged' if self.merged else mode
            orig_df = pd.read_csv(DATASET_DICT[ds_name], encoding=ENCODING)
            for gw in range(1, GWS_PER_SEASON+1):
                for step in ['fetch', 'scale_x', 'scale_y']: 
                    all_gws_titles = [w for w in orig_df.columns.to_list() if w.startswith('total_points')]
                    req_gws_titles = [f'total_points_{i}' for i in range(1, GWS_TO_PREDICT+1)]
                    if step == 'fetch':
                        # df = orig_df[orig_df.apply(lambda row: row['gw'] == gw, axis=1)]  
                        df = orig_df[orig_df.apply(lambda row: row['gw'] == gw and all(DIDNT_PLAY_VALUE != row[req_gws_titles]), axis=1)]
                        X_dict[gw], y_dict[gw] = df = df.drop(columns=all_gws_titles + ['gw']), df[req_gws_titles]
                    elif step == 'scale_x': # https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe
                        if self.scaler:
                            X_dict_scaled[mode][gw] = self.scaler.fit_transform(X_dict[gw]) if mode == 'train' else self.scaler.transform(X_dict[gw])
                        else:
                            ordered_cols = ['id'] + [col for col in X_dict[gw].columns.to_list() if col != 'id']
                            X_dict_scaled[mode][gw] = np.array(X_dict[gw][ordered_cols])
                    else: # step == 'scale_y':
                        if self.scaler:
                            y_dict_scaled[mode][gw] = self.scaler.fit_transform(y_dict[gw]) if mode == 'train' else self.scaler.transform(y_dict[gw])
                        else:
                            y_dict_scaled[mode][gw] = np.array(y_dict[gw])
            if self.model_type == 'RNN':
                ds[mode] = TensorDataset(torch.Tensor(X_dict_scaled[mode]), torch.Tensor(y_dict_scaled[mode]))
            else: # RegTree / MLP
                ds[mode]['X'] = X_dict_scaled[mode]
                ds[mode]['y'] = y_dict_scaled[mode]
        return ds

def get_flatten_dataset(ds: dict):
    X, y = ds['X'], ds['y']
    X_flatten, y_flatten = np.concatenate([X for X in X.values()]), np.concatenate([X for X in y.values()])
    return X_flatten, y_flatten

if __name__ == '__main__':
    print("This file should not be runned directly")
