# General porpuse
import os
import sys
import re
# Data processing
import numpy as np
import pandas as pd
# Project specific
sys.path.append(os.getcwd())
from namesHandler import PlayerName
from fplVars import PLAYERS_DICT, ENCODING, PLAYERS_STATS_FILE, DIDNT_PLAY_VALUE, GWS_TO_PREDICT, DROPPED_COLS, DATASET_DICT, TRAIN_SEASONS, TEST_SEASONS

def create_ds(merge: bool = True):
    ds = pd.DataFrame()
    snum = 0
    for season in sorted(os.listdir('db')):
        if os.path.isfile(os.path.abspath(os.path.join('db', season))):
            continue
        snum += 1
        gws_path = os.path.join('db', season, 'gws')
        for gw in os.listdir(gws_path):
            df = pd.read_csv(os.path.join(gws_path, gw), encoding=ENCODING)
            df['season'] = snum
            df['gw'] = int(re.search('gw([0-9]+)\.csv', gw).group(1))
            df['id'] = df.apply(lambda row: PlayerName(row['name'], PLAYERS_DICT).get_id(), axis=1)
            ds = ds.append(df)
    ds['was_home'] = ds.apply(lambda row: 1 if row['was_home'] else 0, axis=1)
    scores = {(id, season, gw): score for (id, season, gw, score) in zip(ds['id'], ds['season'], ds['gw'], ds['total_points'])}
    for i in range(1, GWS_TO_PREDICT+1):
        ds[f'total_points_{i}'] = ds.apply(lambda row: scores.get((row['id'], row['season'], row['gw']+i), DIDNT_PLAY_VALUE), axis=1)
    if merge:
        season_stats = pd.read_csv(PLAYERS_STATS_FILE, encoding=ENCODING)
        ds = pd.merge(season_stats, ds, how='right', on=('id', 'season'))
    ds.dropna(inplace=True)
    ds['sortby'] = ds.apply(lambda row: (row['id'], row['season'], row['gw']), axis=1)
    ds = ds.sort_values(by='sortby').drop_duplicates()
    ds_all = ds.drop(columns=DROPPED_COLS + ['sortby'] - ['season'], axis=1, errors='ignore')
    ds_name = DATASET_DICT['all_merged'] if merge else DATASET_DICT['all']
    ds_all.to_csv(ds_name, index=False, encoding=ENCODING)
    ds_train = ds.loc[ds['season'].isin(TRAIN_SEASONS)].drop(DROPPED_COLS + ['sortby'], axis=1, errors='ignore') 
    ds_train_name = DATASET_DICT['train_merged'] if merge else DATASET_DICT['train']
    ds_train.to_csv(ds_train_name, index=False, encoding=ENCODING)
    ds_test = ds.loc[ds['season'].isin(TEST_SEASONS)].drop(DROPPED_COLS + ['sortby'], axis=1, errors='ignore') 
    ds_test_name = DATASET_DICT['test_merged'] if merge else DATASET_DICT['test']
    ds_test.to_csv(ds_test_name, index=False, encoding=ENCODING)


def train_validate_split(merge: bool = False, frac: float = 0.2) -> None:
    ds_train_name = DATASET_DICT['train_merged'] if merge else DATASET_DICT['train']
    df = pd.read_csv(ds_train_name, encoding=ENCODING)
    ids = np.unique(np.array(df['id']))
    validate_ids = np.random.choice(ids, int(frac*len(ids)), replace=False)
    df_validate = df.loc[df['id'].isin(validate_ids)]
    df_validate.to_csv(ds_train_name.replace('train', 'validate'), index=False, encoding=ENCODING)
    train_ids = np.setdiff1d(ids, validate_ids)
    df_train = df.loc[df['id'].isin(train_ids)]
    df_train.to_csv(ds_train_name, index=False, encoding=ENCODING)


if __name__ == '__main__':
    for merge in [True, False]:
        create_ds(merge)
        train_validate_split(merge)
