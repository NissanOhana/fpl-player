# Genereal purpose
import os
import sys
import math
import pickle
from typing import List, Tuple
from collections import defaultdict
# Data processing
import pandas as pd
import numpy as np
# Learning
from sklearn.tree import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Graphs
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
# Project specific
sys.path.append(os.getcwd())
from utils.fplVars import ENCODING, GWS_TO_PREDICT, GWS_PER_SEASON, RANDOM_STATE
from datasets import FPLDatasetWrapper, get_flatten_dataset


#############
# Constants #
#############
DEBUG = False
DEBUG_FRAC = 0.1
SHUFFLE = False
PHASES = ['train', 'test']
np.random.seed(RANDOM_STATE)

##########################
# Tree Regressor Classes #
##########################

class FPLRegTree:
    def __init__(self, model, ds: FPLDatasetWrapper, model_params: dict = {}):
        self.model = model(**model_params)
        self.dataset = ds
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def get_dataset(self) -> dict:
        return self.dataset.get_ds()

    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class FPLSlidingTree:
    def __init__(self, gws_to_predict: int = 3, window_size: int = 20, 
                 Ns: Tuple[int] = (31, 31, 31), Ks: Tuple[int] = (23, 23, 23), leaf_size: Tuple[int] = (3, 3, 3)):
        self.gws_to_predict : int           = gws_to_predict
        self.window_size    : int           = window_size
        self.warmup         : int           = window_size
        self.ds             : List[Tuple]   = []
        self.Ns             : int           = Ns
        self.Ks             : int           = Ks
        self.cur_gw         : int           = 0
        self.models         : dict          = {i: FPLRandomForest(gws_to_predict, merged=False, N=Ns[i], use_boost=False, leaf_size = leaf_size[i]) for i in range(self.gws_to_predict)}
        self.hist           : dict          = {i: {} for i in range(self.gws_to_predict)}   # prediction for future +i round
    
    def update(self, X: np.array, y: np.array):
        self.cur_gw += 1
        # Refresh hist and update future predictions
        for i in range(self.gws_to_predict):
            self.hist[i].pop(self.cur_gw-self.window_size, None)
            if i == 0:  # Update with real scores
                self.hist[i][self.cur_gw] = {X[j][0]: (X[j], y[j]) for j in range(X.shape[0])}
            elif self.cur_gw-i > 0:
                self.hist[i][self.cur_gw] = {X[j][0]: (self.hist[0][self.cur_gw-i][X[j][0]][0], y[j]) 
                                             for j in range(X.shape[0]) 
                                             if (self.cur_gw-i) in self.hist[0] and  X[j][0] in self.hist[0][self.cur_gw-i]}
        
        # Recreate model for each round
        for hist_round in range(self.gws_to_predict):
            X_new, y_new = [], []
            for gw in self.hist[hist_round]:
                for id in self.hist[hist_round][gw]:
                    X_new.append(self.hist[hist_round][gw][id][0])
                    y_new.append(self.hist[hist_round][gw][id][1])
            X_new, y_new = np.array(X_new, copy=True), np.array(y_new, copy=True)
            if X_new.size > 0:
                self.models[hist_round].fit(X_new, y_new)
    
    def predict(self, X: np.array):
        return np.array([self.models[i].predict(X, K=self.Ks[i]) for i in range(self.gws_to_predict)]).T
    
    def get_test_ds(self):
        return self.models[0].datasets['test']['X'], self.models[0].datasets['test']['y']

    def do_warmup(self):
        # Get real scores for training y
        cache_file = os.path.join('TPnet', 'models', 'RegTrees', 'db', 'training_real_scores.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                real_scores = pickle.load(f)
        else:
            real_scores = defaultdict(dict)
            for season_num, season_name in enumerate(['2018-19', '2019-20']):
                for gw in range(1, GWS_PER_SEASON+1):
                    effective_gw = GWS_PER_SEASON*season_num + gw
                    df = pd.read_csv(os.path.join('db', season_name, 'gws', f'gw{gw}.csv'), encoding=ENCODING)
                    for (code, score) in zip(df['code'], df['total_points']):
                        if code not in real_scores:
                            real_scores[code] = {}
                        real_scores[code][effective_gw] = score
            with open(cache_file, 'wb') as f:
                pickle.dump(real_scores, f)
        
        self.cur_gw = 2*GWS_PER_SEASON-self.warmup-self.gws_to_predict-1    # for matching internal state
        for gw in range(GWS_PER_SEASON-self.warmup-self.gws_to_predict, GWS_PER_SEASON):
            effective_gw = GWS_PER_SEASON + gw
            df = pd.read_csv(os.path.join('db', '2019-20', 'gws', f'gw{gw}.csv'), encoding=ENCODING).dropna().drop_duplicates(subset=['code'])
            y_dict = {}
            for id in [id for id in df['code'] if not math.isnan(id)]:
                y_dict[id] = [real_scores[id][effective_gw], 
                            real_scores[id].get(gw+1, real_scores[id][effective_gw]), # get is just for not breaking compatability
                            real_scores[id].get(gw+2, real_scores[id][effective_gw])  #  it doesn't effect anything
                ]
            ids = np.array(list(y_dict.keys()))
            df = df[df['code'].isin(ids)]               
            cols_no_col = [col for col in df.columns.to_list() if col != 'code']
            df = df[['code'] + cols_no_col]
            X = np.array(df.drop(columns=['name', 'name_p', 'kickoff_time', 'total_points']))
            y = np.array(df['total_points']) 
            self.update(X, y)

    @classmethod
    def analyze(cls, gws_to_predict: int = 3, window_size: int = 10, Ns: Tuple[int] = (31, 21, 35), Ks: Tuple[int] = (23, 11, 31), 
                    leaf_size: Tuple[int] = (3, 9, 18), force_train: bool = False):
        cache_file = os.path.join('TPnet', 'models', 'RegTrees', f'FPLSlidingTree-windowsize_{window_size}-N_{Ns}-K_{Ks}-leafsize_{leaf_size}.pkl')
        if force_train or not os.path.exists(cache_file):
            # Get real scores for y
            scores_cache_files = os.path.join('TPnet', 'models', 'RegTrees', 'db', 'training_real_scores.pkl')
            if os.path.exists(scores_cache_files):
                with open(scores_cache_files, 'rb') as f:
                    real_scores = pickle.load(f)
            else:
                real_scores = defaultdict(dict)
                for season_num, season_name in enumerate(['2018-19', '2019-20']):
                    for gw in range(1, GWS_PER_SEASON+1):
                        effective_gw = GWS_PER_SEASON*season_num + gw
                        df = pd.read_csv(os.path.join('db', season_name, 'gws', f'gw{gw}.csv'), encoding=ENCODING)
                        for (code, score) in zip(df['code'], df['total_points']):
                            if code not in real_scores:
                                real_scores[code] = {}
                            real_scores[code][effective_gw] = score
                with open(scores_cache_files, 'wb') as f:
                    pickle.dump(real_scores, f)
            
            # Warm up and train
            model = FPLSlidingTree(gws_to_predict, window_size, Ns=Ns, Ks=Ks, leaf_size=leaf_size)
            results = {}
            for season_num, season_name in enumerate(['2018-19', '2019-20']):
                for gw in range(1, GWS_PER_SEASON):
                    effective_gw = GWS_PER_SEASON*season_num + gw
                    df = pd.read_csv(os.path.join('db', season_name, 'gws', f'gw{gw}.csv'), encoding=ENCODING).dropna().drop_duplicates(subset=['code'])
                    y_dict = {}
                    for id in [id for id in df['code'] if not math.isnan(id)]:
                        y_dict[id] = [real_scores[id][effective_gw], 
                                    real_scores[id].get(effective_gw+1, real_scores[id][effective_gw]), # get is just for not breaking compatability
                                    real_scores[id].get(effective_gw+2, real_scores[id][effective_gw])  #  it doesn't effect anything
                        ]
                    ids = np.array(list(y_dict.keys()))
                    y_real_future = np.array(list(y_dict.values()))
                    df = df[df['code'].isin(ids)]               
                    cols_no_col = [col for col in df.columns.to_list() if col != 'code']
                    df = df[['code'] + cols_no_col]
                    X = np.array(df.drop(columns=['name', 'name_p', 'kickoff_time']))
                    y = np.array(df['total_points'])           
                    pass
                    if effective_gw > model.warmup: # warmed up, can predict
                        y_pred = model.predict(X)
                        results[effective_gw] = {
                            'ids': ids,
                            'pred': y_pred,
                            'real': y_real_future,
                            'delta': y_pred - y_real_future,
                            'MAE': [mean_absolute_error(y_real_future[:,i], y_pred[:,i]) for i in range(gws_to_predict)]
                        }
                        print(f"GW: {effective_gw}\tMAE: {results[effective_gw]['MAE']}")
                    model.update(X, y)
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
        else:
            with open(cache_file, 'rb') as f:
                results = pickle.load(f)
        
        # Plot
        pct = 0.8
        max_abs_err_pct, MAEs, medians = [], [], []
        for effective_gw in results:
            tmp_max_abs_err_pct, tmp_MAEs,tmp_medians = [], [], []
            for i in range(results[effective_gw]['delta'].shape[1]):
                deltas = results[effective_gw]['delta'][:,i]
                tmp_max_abs_err_pct.append(np.quantile(np.abs(deltas), pct))
                tmp_MAEs.append(results[effective_gw]['MAE'][i])
                tmp_medians.append(np.median(deltas))
            max_abs_err_pct.append(tmp_max_abs_err_pct)
            MAEs.append(tmp_MAEs)
            medians.append(tmp_medians)
        
        max_abs_err_pct, MAEs, medians = np.array(max_abs_err_pct), np.array(MAEs), np.array(medians)
        plt.clf(); plt.cla
        _, axs = plt.subplots(3)
        plt.suptitle(f'Mean and Maximal errors by gameweek\n'
                     f'WindowsSize={window_size}, N={Ns}, K={Ks}, leafsize={leaf_size}', size=14, weight='bold')
        first_predicted_gw, last_predicted_gw = list(results.keys())[0], list(results.keys())[-1]
        colors = ['darkred', 'seagreen', 'royalblue', 'darkgoldenrod', 'peru']
        for i in range(gws_to_predict):
            last_index = last_predicted_gw - first_predicted_gw + 1 - i
            axs[0].plot(range(first_predicted_gw, first_predicted_gw+len(MAEs[:last_index,i])), MAEs[:last_index,i], color=colors[i], label=f'Next {i+1}')
            axs[1].plot(range(first_predicted_gw, first_predicted_gw+len(max_abs_err_pct[:last_index,i])), max_abs_err_pct[:last_index,i], color=colors[i], label=f'Next {i+1}')
            axs[2].plot(range(first_predicted_gw, first_predicted_gw+len(medians[:last_index,i])), medians[:last_index,i], color=colors[i], label=f'Next {i+1}')
        axs[0].set_title('Mean Absolute Error')
        axs[0].set_ylabel('Absolute Error')
        axs[0].legend(loc='upper left')
        axs[1].set_title(f'Maximal error of {pct*100}% players')
        axs[1].set_ylabel('Absolute Error')
        axs[2].set_title(f'Median absolute error')
        axs[2].set_ylabel('Absolute error')
        axs[2].set_xlabel('Effective gameweek')  
        figname = os.path.join('TPnet', 'models', 'RegTrees', f'FPLSlidingTree-windowsize_{window_size}-N_{Ns}-K_{Ks}-leafsize_{leaf_size}.png')
        plt.tight_layout()
        plt.savefig(figname)

        # Plot KDE
        scaler = MinMaxScaler()
        for scored_only in [True, False]:
            plt.cla(); plt.cla()
            deltas = []
            for effective_gw in results:
                pred = results[effective_gw]['pred'][np.array(~np.all(results[effective_gw]['real'] == 0, axis=1)), :] if scored_only else results[effective_gw]['pred']
                normalized_pred = scaler.fit_transform(pred)
                real = results[effective_gw]['real'][np.array(~np.all(results[effective_gw]['real'] == 0, axis=1)), :] if scored_only else results[effective_gw]['real']
                normalized_real = scaler.fit_transform(real)
                deltas.extend(normalized_pred-normalized_real)
                #deltas.extend(results[effective_gw]['delta'][np.array(~np.all(results[effective_gw]['real'] == 0, axis=1)), :] if scored_only else results[effective_gw]['delta'])
            deltas = np.array(deltas)
            deltas_df = pd.DataFrame(columns=[f'gw{i+1}' for i in range(3)], data=deltas)
            rp = sns.displot(data=deltas_df, kind='kde')
            rp.fig.subplots_adjust(top=0.98)
            rp.fig.suptitle(f'KDE plot per gameweek; WindowsSize={window_size},\n'
                            f'N={Ns}, K={Ks}, leafsize={leaf_size}\n'
                            f'{"Scored Players Only" if scored_only else "All Players"}', size=14, weight='bold')
            plt.legend()
            plt.xlabel('Delta')
            plt.ylabel('Density')
            figname = os.path.join('TPnet', 'models', 'RegTrees', f'FPLSlidingTree-windowsize_{window_size}-N_{Ns}-K_{Ks}-leafsize_{leaf_size}-scoredonly_{scored_only}-kde.png')
            plt.tight_layout()
            plt.savefig(figname)

class FPLRegTreeBase:
    def __init__(self, gws_to_predict: int = 1, merged: bool = True):
        self.ds_wrapper = FPLDatasetWrapper(gws_to_predict, merged, scaled=False, model_type='RegTree')
        self.merged = merged
        self.datasets = None
        self.scaler = MinMaxScaler()
    
    def load_datasets(self, gws_to_predict: int, load_ds: bool) -> None:
        ds_pickled = os.path.join('TPnet', 'models', 'RegTrees', f'datasets_merged_{gws_to_predict}.pkl' if self.merged else f'datasets_{gws_to_predict}.pkl')
        if load_ds and os.path.exists(ds_pickled):
            with open(ds_pickled, 'rb') as f:
                datasets = pickle.load(f)
        else:
            with open(ds_pickled, 'wb') as f:
                datasets = self.ds_wrapper.get_ds()
                pickle.dump(datasets, f)
        self.datasets = datasets

class FPLRandomForest(FPLRegTreeBase):
    def __init__(self, gws_to_predict: int = 1, merged: bool = True, N: int = 5, use_boost: bool = False, boost_ratio: int = 0.95, leaf_size: int = 3):
        super(FPLRandomForest, self).__init__(gws_to_predict, merged)
        self.N = N
        self.use_boost = use_boost
        self.boost_ratio = boost_ratio
        self.gws_to_predict = gws_to_predict
        self.load_datasets(gws_to_predict, load_ds=True)
        self.trees : List[FPLTreeNode] = [FPLTreeNode(leaf_size=leaf_size) for _ in range(self.N)]
    
    def fit(self, X: np.ndarray, y: np.ndarray, shuffle: bool = False) -> None:
        if self.use_boost:   # cut 0 scored events
            zeros = np.delete(np.arange(len(X)), np.nonzero(y))
            idx_zeros_to_delete = np.random.choice(zeros, size=(int(len(zeros)*self.boost_ratio)), replace=False)
            idx_reduced_zeros = np.delete(np.arange(len(y)), idx_zeros_to_delete )
            X, y = X[idx_reduced_zeros], y[idx_reduced_zeros]
        X_scaled = self.scaler.fit_transform(X)
        base_indexes = np.random.choice(range(X_scaled.shape[0]), size=X_scaled.shape[0], replace=False) if shuffle else range(X_scaled.shape[0])
        indexes = np.array_split(base_indexes, self.N)
        X_splitted = [X_scaled[idx] for idx in indexes]
        y_splitted = [y[idx] for idx in indexes]
        for i, (X_i, y_i) in enumerate(zip(X_splitted, y_splitted)):
            self.trees[i].fit(X_i, y_i)
    
    def predict(self, X: np.ndarray, K: int = 3) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        pred = []
        for line in X_scaled:
            line_pred = []
            deltas = np.array([np.linalg.norm(line - tree.centroid) for tree in self.trees])
            for tree in [self.trees[i] for i in deltas.argsort()[:K]]:
                boost = tree.scored_ratio if self.use_boost else 0
                line_pred.append(tree.predict(line) * (1+boost))
            score = np.array(line_pred).mean() if self.gws_to_predict == 1 else np.array(line_pred).squeeze().mean(axis=0)
            pred.append(score)
        return np.array(pred).squeeze()  

class FPLTreeNode:
    def __init__(self, leaf_size: int = 3):
        self.X            : np.ndarray            = None
        self.y            : np.ndarray            = None
        self.fitted       : bool                  = False
        self.centroid     : np.ndarray            = None
        self.scored_ratio : float                 = 0
        self.model        : DecisionTreeRegressor = DecisionTreeRegressor(criterion='mae', 
                                                                          splitter='best', 
                                                                          random_state=RANDOM_STATE,
                                                                          min_samples_leaf=leaf_size, 
                                                                          min_samples_split=2*leaf_size-1)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.centroid = np.mean(X, axis=0)
        self.X = X
        self.y = y
        self.scored_ratio = 0 if len(y) == 0 else len(np.where(y != 0)) / len(y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            self.model.fit(self.X, self.y)
            self.fitted = True
        return self.model.predict(X.reshape(1, -1))

class FPLUpdatingForest:
    def __init__(self, gws_to_predict: int = 1, merged: bool = False, 
                       N: int = 5, K: int = 3, use_boost: bool = False):
        self.gw = 1
        
        self.prev_seasons_K = K
        self.prev_seasons_regr = FPLRandomForest(gws_to_predict, merged=merged, N=40, use_boost=True, boost_ratio=0.95)
        self._fit_prev()
        
        self.X = np.array([])
        self.y = np.array([])
        self.cur_season_rgr = FPLRandomForest(gws_to_predict, merged=merged, N=20, use_boost=True, boost_ratio=0.9)
        self._warm_cur()
    
    def _fit_prev(self):
        X_train_flatten, y_train_flatten = get_flatten_dataset(self.prev_seasons_regr.datasets['train'])
        X_valid_flatten, y_valid_flatten = get_flatten_dataset(self.prev_seasons_regr.datasets['validate'])
        X_base, y_base = np.concatenate((X_train_flatten, X_valid_flatten)), np.concatenate((y_train_flatten, y_valid_flatten))
        self.prev_seasons_regr.fit(X_base, y_base, shuffle=True)
    
    def _warm_cur(self):
        pass
    
    def update(self, X: np.ndarray, y: np.ndarray):
        self.gw += 1
        self.X = np.concatenate((self.X, X)) if self.X.size > 0 else X
        self.y = np.concatenate((self.y, y)) if self.y.size > 0 else y
        self.cur_season_rgr.fit(self.X, self.y)
    
    def predict(self, X: np.ndarray):
        prev_seasons_score = self.prev_seasons_regr.predict(X, K=self.prev_seasons_K)
        if self.gw < GWS_TO_PREDICT:
            return prev_seasons_score
        cur_season_score = self.cur_season_rgr.predict(X)
        cur_season_score_weight = 0.3 + (self.gw / (2*GWS_PER_SEASON))
        weighted_cur_season_score = cur_season_score_weight * cur_season_score
        weighted_prev_seasons_score = (1-cur_season_score_weight) * prev_seasons_score
        return weighted_prev_seasons_score + weighted_cur_season_score

    def get_test_ds(self):
        return self.prev_seasons_regr.datasets['test']['X'], self.prev_seasons_regr.datasets['test']['y']

###############################
# Previous training processes #
###############################
def train_knn(merged: False, k_range: range, gws_to_predict = 1):
    real_y, predict_y = [], []
    rmse, deltas, deltas_nz = [], [], []
    dataset = FPLDatasetWrapper(gws_to_predict=gws_to_predict, merged=merged, scaled=True, model_type='RegTree')
    ds_name = "WithPrevSeason" if merged else "WithoutPrevSeason"
    for K in k_range:
        model_params = {'n_neighbors': K,
                        'weights': 'distance',
                        'algorithm': 'ball_tree',
                        'leaf_size': 30,
                        'p': 2,
                        'n_jobs': -1}
        regr = FPLRegTree(model=KNeighborsRegressor, model_params=model_params, ds=dataset)
        X_train, y_train = regr.get_dataset('train')
        regr.fit(X_train, y_train)
        X_test, y_test = regr.get_dataset('test')
        y_test_pred = regr.predict(X_test)

        err = math.sqrt(mean_squared_error(y_test, y_test_pred))
        print(f'RMSE for K={K}: {err}')
        rmse.append(err)
        real_y.append(y_test)
        predict_y.append(y_test_pred)
        deltas.append(y_test - y_test_pred)
        deltas_nz.append((y_test - y_test_pred)[np.where(y_test > 0)])
    plt.cla()
    plt.clf()
    plt.title(f'KNeighborsRegressor: RMSE by K value\nDataset: {ds_name}; GWs: {gws_to_predict}')
    plt.xlabel('K value')
    plt.ylabel('RMSE')
    plt.plot(k_range, rmse)
    plt.savefig(f'knn_{ds_name}_gws-{gws_to_predict}.png')

    plt.cla()
    plt.clf()
    
    df = pd.DataFrame(columns=['K', 'delta'])
    for i,K in enumerate(k_range):
        subset = [{'K': K, 'y': y, 'pred': pred, 'delta': delta} for delta, y, pred in zip(deltas[i], real_y[i], predict_y[i])]
        # subset = deltas[i]
        df = df.append(subset, ignore_index=True)
        
    for i in range(gws_to_predict):
        tmp_df = df.copy(deep=True)
        if gws_to_predict > 1:
            tmp_df['delta'] = df.apply(lambda row: row['delta'][i], axis=1)
        rp = sns.displot(data=tmp_df, kind='kde', x='delta', hue='K', palette='hsv')
        rp.fig.subplots_adjust(top=0.98)
        rp.fig.suptitle(f'KDE plot per K value\nDataset: {ds_name}; GWs: {gws_to_predict}')
        plt.legend()
        plt.xlabel('Delta')
        plt.ylabel('Density')
        plt.savefig(f'knn_kde_{ds_name}_gws-{i+1}_{gws_to_predict}.png')

    df.to_csv(f'stats_{ds_name}_gws-{gws_to_predict}.csv', index=False, encoding=ENCODING)

def list_repr_to_list(arr: List):
    import re
    import ast
    if isinstance(arr[0], float):
        return arr
    return [ast.literal_eval(re.sub(' +', ' ', row).replace('[ ', '[').replace(' ]', ']').replace(' ', ',')) for row in arr]

def recreate_mae(k_range: range):
    ds_rmse = {}
    for ds_name in ['WithoutPrevSeason', 'WithPrevSeason']:
        ds_rmse = {}
        for gws_to_predict in range(1, GWS_TO_PREDICT+1):
            reals, preds = defaultdict(list), defaultdict(list)
            df = pd.read_csv(f'stats_{ds_name}_gws-{gws_to_predict}.csv', encoding=ENCODING)
            df.apply(lambda row: reals[row['K']].append(row['y']), axis=1)
            df.apply(lambda row: preds[row['K']].append(row['pred']), axis=1)
            ds_rmse[gws_to_predict] = [mean_absolute_error(list_repr_to_list(reals[kr]), list_repr_to_list(preds[kp])) for (kr, kp) in zip (reals.keys(), preds.keys())]
        plt.cla()
        plt.clf()
        plt.title(f'MAE per K values and gameweeks to predict\nDataset {ds_name}')
        plt.xlabel('K value')
        plt.ylabel('Mean Averaged Error')
        for gws in range(1, GWS_TO_PREDICT+1):
            plt.plot(k_range, ds_rmse[gws], label=f'gws={gws}')
        plt.legend(loc='upper right')
        plt.savefig(f'knn_mae_{ds_name}.png')


def random_forest_runner(load_model: bool = False, use_boost: bool = False, gws_to_predict: int = 1):
    basedir = os.path.join('TPnet', 'models', 'RegTrees')
    validation = {}
    test = {}
    # Train & predict
    if load_model:
        with open(os.path.join(basedir, 'RandomForestDict.pkl'), 'rb') as f:
            tmp = pickle.load(f)
        validation, test = tmp['validation'], tmp['test']
    else:
        minN, maxN = 21, 40
        minK, stepK = 7, 3
        ratios = [0.8, 0.9] if use_boost else [1]
        for ratio in ratios:
            for merged in [True, False]:    
                validation[ratio][merged] = {}
                test[ratio][merged] = {}
                for N in range(minN, maxN+1, 3):
                    # Same model by N; speed up predictions
                    regr = FPLRandomForest(gws_to_predict=gws_to_predict, merged=merged, N=N, use_boost=use_boost, boost_ratio=ratio)
                    X_train, y_train = regr.datasets['train']                    
                    regr.fit(X_train, y_train, shuffle=True)
                    validation[ratio][merged][N] = {}
                    test[ratio][merged][N] = {}
                    for K in range(minK, N+1, stepK):
                        X_valid, y_valid = regr.datasets['validate']
                        if gws_to_predict == 1:
                            y_valid = y_valid.reshape(-1,)
                        y_pred_valid = regr.predict(X_valid, K=K)
                        validation[ratio][merged][N][K] = {
                            'MAE': mean_absolute_error(y_valid, y_pred_valid),
                            'pred': y_pred_valid,
                            'real': y_valid
                        }

                        X_test, y_test = regr.datasets['test']
                        if gws_to_predict == 1:
                            y_test = y_test.reshape(-1,)
                        y_pred_test = regr.predict(X_test, K=K)
                        test[ratio][merged][N][K] = {
                            'MAE': mean_absolute_error(y_test, y_pred_test),
                            'pred': y_pred_test,
                            'real': y_test
                        }
                        print(f'N: {N}, K: {K}')
                        print(f"Validation MAE: {validation[merged][N][K]['MAE']}\tTest MAE: {test[merged][N][K]['MAE']}")
                        print('='*90) 
            with open(os.path.join(basedir, f'RandomForestDict-N_{maxN}-boost_{use_boost}_{ratio*100}.pkl'), 'wb') as f:
                pickle.dump({'validation': validation, 'test': test}, f)

    # Plot
    for merged in [False, True]:
        Ns, Ks, MAEs = [], [], []
        for N in validation[merged]:
            for K in validation[merged][N]:
                Ns.append(N)
                Ks.append(K)
                MAEs.append(validation[merged][N][K]['MAE'])
        ds_name = 'WithPrevSeason' if merged else 'WithoutPrevSeason'
        plt.clf(); plt.cla()
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.set_title(f'Mean Average Error by K, N\n'
                     f'Dataset: {ds_name}', fontweight='bold', fontsize=14)
        surf = ax.plot_trisurf(Ks, Ns, MAEs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, orientation='horizontal')
        ax.set_xlabel('K', fontweight='bold', fontsize=10) 
        ax.set_ylabel('N', fontweight='bold', fontsize=10)
        ax.set_zlabel('MAE', fontweight='bold', fontsize=10)
        figname = os.path.join(basedir, f'FPLRandomForest_{ds_name}_N{maxN}_boost{100*ratio}.png' if use_boost else f'FPLRandomForest_{ds_name}_N{maxN}.png')
        plt.savefig(figname)
    pass            

def _plot_knn_kde(results: dict, gws_to_predict: int = 1, only_scored: bool = False, top_mae = None):
    for merged in results:
        ds_name = 'WithPrevSeason' if merged else 'WithoutPrevSeason'
        players = 'Only scored players' if only_scored else 'All players'
        for N in results[merged]:
            if N > 20:
                continue
            df = pd.DataFrame(columns=['K', 'delta', 'MAE'])
            for K in results[merged][N]:
                y, y_pred = results[merged][N][K]['real'], results[merged][N][K]['pred']
                idx = [idx for idx,score in enumerate(y) if np.all(score)] if only_scored else range(len(y))
                mae = [mean_absolute_error(y_pred[idx][gw], y[idx][gw]) for gw in range(gws_to_predict)]
                subset = [{'K': K, 'delta': pred - real, 'MAE': mae} for (pred, real) in zip(y_pred[idx], y[idx])]
                df = df.append(subset, ignore_index=True)

            for i in range(gws_to_predict):
                tmp_df = df.copy(deep=True)
                if top_mae:
                    tmp_df['MAE'] = df.apply(lambda row: row['MAE'][i], axis=1)
                    MAEs = sorted(tmp_df['MAE'].unique())
                    tmp_df = tmp_df.loc[tmp_df.apply(lambda row: row['MAE'] in MAEs[:top_mae], axis=1)]
                if gws_to_predict > 1:
                    tmp_df['delta'] = df.apply(lambda row: row['delta'][i], axis=1)
                tmp_df['K'] = tmp_df.apply(lambda row: f"K:{row['K']}", axis=1)
                rp = sns.displot(data=tmp_df, kind='kde', x='delta', hue='K', palette='hsv')
                rp.fig.subplots_adjust(top=0.98)
                rp.fig.suptitle(f'KDE by K values, N={N}, gws: {i+1}/{gws_to_predict}\nDataset: {ds_name}; {players}')
                plt.legend()
                plt.xlabel('Delta')
                plt.ylabel('Density')
                plt.savefig(os.path.join('TPnet', 'models', 'RegTrees', ds_name, f'kde-N{N}-merged_{merged}-onlyscored_{only_scored}-gws_{i+1}_outof_{gws_to_predict}.png'))

if __name__ == '__main__':
    print("This file should not be runned directly")
    # for window_size in [5, 10, 15, 20]:
    #     for (N,K) in [(23,11), (31, 23), (35, 31)]:
    #         for leaf_size in [3, 9, 18]:
    #             FPLSlidingTree.analyze(gws_to_predict=GWS_TO_PREDICT,
    #                                    window_size=window_size,
    #                                    Ns=[N]*GWS_TO_PREDICT,
    #                                    Ks=[K]*GWS_TO_PREDICT,
    #                                    leaf_size=[leaf_size]*GWS_TO_PREDICT
    #             )
