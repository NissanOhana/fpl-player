import pandas as pd
from random import random, seed
import numpy as np

seed(2)

def generate_hype(ict, tb):
    prob = np.random.uniform(0.7, 1)
    if ict >= 8 and tb >= 55000:
        return 1*prob
    elif tb <= -100000 and ict <= 1:
        return -1*prob
    else:
        return 0

def db_media_hype_param():
    for i in range(2, 39):
        gw_path = f'../db/2020-21/gws/gw{i}.csv'
        gw = pd.read_csv(gw_path, encoding="ISO-8859-1")
        gw.drop(
            labels=gw.columns.difference(['code', 'ict_index', 'transfers_balance']),
            axis='columns',
            inplace=True)
        gw['hype'] = gw.apply(lambda x: generate_hype(x['ict_index'], x['transfers_balance']),
                              axis='columns')
        gw.rename(columns={'code': 'player_code'},  inplace=True)

        gw.drop(
            labels=gw.columns.difference(['player_code', 'hype', ]),
            axis='columns',
            inplace=True)

        prediction_gw_path = f'../db/simulation/predictions/gw{i}.csv'
        pgw = pd.read_csv(prediction_gw_path, encoding="ISO-8859-1")
        gw_new = gw.join(pgw.set_index('player_code'), on='player_code')
        gw_new.to_csv(prediction_gw_path, index=False, encoding="ISO-8859-1")

def generate_minutes_prediction(value):
    prob = random()
    if value == 0:
        if prob > 0.01:
            return -1
    if value != 0:
        if prob > 0.025:
            return 1
    return 0


def db_realdata_creator():
    for i in range(1, 39):
        gw_path = f'../db/simulation/realdata/gw{i}.csv'
        gw = pd.read_csv(gw_path, encoding="ISO-8859-1")
        gw.drop(
            labels=gw.columns.difference(['id', 'total_points']),
            axis='columns',
            inplace=True)
        gw.rename(columns={'code': 'id'},  inplace=True)

        new_cols = ['id', 'total_points']
        gw_new = gw[new_cols]

        gw_new.to_csv(gw_path, index=False, encoding="ISO-8859-1")

if __name__ == '__main__':
    db_realdata_creator()




