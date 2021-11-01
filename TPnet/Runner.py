# General porpuse
import os
import sys
# Data processing
import numpy as np
import pandas as pd
from heapq import nlargest
# Graphs
from matplotlib import pyplot as plt
# Project specific
sys.path.append(os.getcwd())
from TPnet.RegTrees import FPLSlidingTree
from utils.fplVars import GWS_PER_SEASON, ENCODING, GWS_TO_PREDICT, PLAYERS_DICT
from utils.namesHandler import PlayerName


def predict_season():
    # Predicts our test season, which is 2020-21
    preds = {}
    output = pd.DataFrame()
    regr = FPLSlidingTree(gws_to_predict=GWS_TO_PREDICT, window_size=10, Ns=(31, 21, 35), Ks=(23, 11, 31), leaf_size=(3, 9, 18))
    regr.do_warmup()    # train on last window_size gws of last season
    for gw in range(GWS_PER_SEASON):
        # Read data and prepare to insert to model
        gw_df = pd.read_csv(os.path.join('db', '2020-21', 'gws', f'gw{gw}.csv'), encoding=ENCODING).dropna().drop_duplicates(subset=['code'])
        cols_no_col = [col for col in gw_df.columns.to_list() if col != 'code']
        gw_df = gw_df[['code'] + cols_no_col]
        X = np.array(gw_df.drop(columns=['name', 'name_p', 'kickoff_time', 'total_points']))
        y = np.array(gw_df['total_points'])
        # Predicte and update
        preds[gw] = regr.predict(X)
        regr.update(X, y)
        
        gw_dict = [{'gw': gw, 'id': id, 'ps1': ps1, 'ps2': ps2, 'ps3': ps3}
                    for (id, ps1, ps2, ps3) 
                    in zip(gw_df['code'],preds[gw][:,0],preds[gw][:,1],preds[gw][:,2])] 
        gw_df = pd.DataFrame(gw_dict)
        gw_df.to_csv(os.path.join('db', 'simulation', 'scores', f'gw{gw}.csv'), index=False, encoding=ENCODING)
        output = gw_df if output.empty else pd.concat([output, pd.DataFrame(gw_dict)])    
    output.to_csv(os.path.join('db', 'simulation', 'scores', 'all_gws.csv'), index=False, encoding=ENCODING)

def merge():
    # Merge scores & twitter predictions to one file, for feeding the algorithm
    scores_dir = os.path.join('db', 'simulation', 'scores')
    twitter_dir = os.path.join('db', 'simulation', 'predictions')
    out_dir = os.path.join('db', 'simulation', 'merged_predictions')
    for gw in range(1, GWS_PER_SEASON+1):
        print(f'Working on gw {gw}')
        df_scores = pd.read_csv(os.path.join(scores_dir, f'gw{gw-1}.csv'), encoding=ENCODING).drop_duplicates(subset='id')
        df_twitter = pd.read_csv(os.path.join(twitter_dir, f'gw{gw}.csv'), encoding=ENCODING).drop_duplicates(subset='id')
        df_twitter['hipe'] = df_twitter['hype']
        df_twitter.drop('hype', axis=1)
        df_merged = pd.concat([df_twitter.set_index('id'), df_scores.set_index('id')], axis=1, join='inner').reset_index()
        df_scores['id'].drop_duplicates()
        cols = ['id', 'pos', 'team', 'val', 'ps1', 'ps2', 'ps3', 'play', 'hipe']
        df_merged[cols].to_csv(os.path.join(out_dir, f'gw{gw}.csv'), index=False, encoding=ENCODING)

def plot_scores_histogram():  
    scores_by_season = {}
    seasons = ['2018-19', '2019-20', '2020-21']
    for season in seasons:
        folder = os.path.join('db', season, 'gws')
        scores = []
        for gw in range(1, GWS_PER_SEASON+1):
            df = pd.read_csv(os.path.join(folder, f'gw{gw}.csv'), encoding=ENCODING)
            scores.extend(list(df['total_points']))
        scores_by_season[season] = np.array(scores)
    plt.clf(); plt.cla()
    plt.title('Total points distribution by season', size=14, weight='bold')
    plt.xlabel('Total points')
    plt.ylabel('Records')
    bins = range(-5, 20)
    for season in seasons:
        plt.hist(scores_by_season[season], bins, label=season, histtype='step', stacked=False, fill=False)
    plt.legend(loc='upper right')
    plt.tight_layout()
    figname = os.path.join('db', 'real_scores_histogram.png')
    plt.savefig(figname)

def plot_chosen_players():
    players = ['Harry Kane', 'Mohamed Salah', 'Aaron Cresswell']
    players_id = {name: PlayerName(name, PLAYERS_DICT).get_id() for name in players}
    real_scores = {id: [] for id in players_id.values()} 
    pred_scores = {id: [] for id in players_id.values()} 
    preds_dir = os.path.join('db', 'simulation', 'scores')
    real_dir = os.path.join('db', 'simulation', 'realdata')
    for gw in range(GWS_PER_SEASON):
        df_real = pd.read_csv(os.path.join(real_dir, f'gw{gw+1}.csv'), encoding=ENCODING)
        df_pred = pd.read_csv(os.path.join(preds_dir, f'gw{gw}.csv'), encoding=ENCODING)
        for id in real_scores:
            real_scores[id].append(float(df_real[df_real['id']==id]['total_points'].mean() if id in list(df_real.id) else real_scores[id][-1]))         
            pred_scores[id].extend([[float(ps1), float(ps2), float(ps3)] for ps1,ps2,ps3 in df_pred[df_pred['id']==id][['ps1', 'ps2', 'ps3']].values]
                if id in list(df_pred.id) else [[pred_scores[id][-1], pred_scores[id][-1], pred_scores[id][-1]]])

    for gw in range(1, GWS_TO_PREDICT+1):
        plt.clf(); plt.cla()
        _, axs = plt.subplots(len(players))
        plt.suptitle(f'Real vs predicted scores of chosen players\n'
                     f'Gameweek {gw}/{GWS_TO_PREDICT}', size=14, weight='bold')
        for i,name in enumerate(players):
            axs[i].plot(range(1, GWS_PER_SEASON+1), real_scores[players_id[name]], color='seagreen', label='Real')
            axs[i].plot(range(1, GWS_PER_SEASON+1), np.array(pred_scores[players_id[name]])[:,gw-1], color='darkblue', label='Predicted')
            med = np.median(np.array(pred_scores[players_id[name]])[:,gw-1])
            avg = np.average(np.array(pred_scores[players_id[name]])[:,gw-1])
            axs[i].set_title(f'{name}; avg {round(avg,2)}, median {round(med,2)}')
            axs[i].set_ylabel('Points')
        axs[0].legend(loc='upper right')
        axs[-1].set_xlabel('Gameweek')
        plt.tight_layout()
        figname = os.path.join('db', f'players_comparison_gw{gw}.png')
        plt.savefig(figname)

def analyze_real_results():
    real_scores = {}
    for gw in range(1, GWS_PER_SEASON+1):
        df = pd.read_csv(os.path.join('db', 'simulation', 'realdata', f'gw{gw}.csv'), encoding=ENCODING)
        real_scores[gw] = {code: score for (code, score) in zip(df['id'], df['total_points'])}
    preds = {}
    for gw in range(GWS_PER_SEASON):
        df = pd.read_csv(os.path.join('db', 'simulation', 'scores', f'gw{gw}.csv'), encoding=ENCODING)
        preds[gw+1] = {code: {0: ps1, 1: ps2, 2: ps3} for (code,ps1,ps2,ps3) in zip(df['id'],df['ps1'],df['ps2'],df['ps3'])}
    deltas_0, deltas_1, deltas_2 = [], [], []
    for gw in range(1, GWS_PER_SEASON+1):
        for code in preds[gw]:
            if code in real_scores[gw]:
                deltas_0.append(abs(preds[gw][code][0] - real_scores[gw][code]))
            if gw+1 in real_scores and code in real_scores[gw+1]:
                deltas_1.append(abs(preds[gw][code][1] - real_scores[gw+1][code]))
            if gw+2 in real_scores and code in real_scores[gw+2]:
                deltas_2.append(abs(preds[gw][code][2] - real_scores[gw+2][code]))
    deltas_0 = np.array(deltas_0)
    deltas_1 = np.array(deltas_1)
    deltas_2 = np.array(deltas_2)
    print(f'Next 1 gw MAE: {np.mean(deltas_0)}')
    print(f'Next 2 gw MAE: {np.mean(deltas_1)}')
    print(f'Next 3 gw MAE: {np.mean(deltas_2)}')
    
    for gw in range(GWS_PER_SEASON):
        df = pd.read_csv(os.path.join('db', '2020-21', 'gws', f'gw{gw}.csv'), encoding=ENCODING)
        for (code, score) in zip(df['code'], df['total_points']):
            if code not in real_scores:
                real_scores[code] = {}
            real_scores[code][gw] = score
    
def get_best_scorers():
    N = 30
    # Get all predicted
    preds_df = pd.read_csv(os.path.join('db', 'simulation', 'scores', 'all_gws.csv'), encoding=ENCODING)
    preds_dict = {}
    for id in preds_df['id'].unique():
        preds_dict[id] = float(preds_df[preds_df['id'] == id].groupby(by=['id']).sum()['ps1'])
    pred_highest = nlargest(N, preds_dict, key=preds_dict.get)

    # Get all real scores
    real_df = pd.read_csv(os.path.join('db', 'simulation', 'realdata', 'gw1.csv'), encoding=ENCODING)
    real_dict = {}
    for gw in range(2, GWS_PER_SEASON+1):
        tmp_df = pd.read_csv(os.path.join('db', 'simulation', 'realdata', f'gw{gw}.csv'), encoding=ENCODING)
        real_df = pd.concat([real_df, tmp_df])
    for id in real_df['id'].unique():
        real_dict[id] = float(real_df[real_df['id'] == id].groupby(by=['id']).sum()['total_points'])
    real_highest = nlargest(len(real_dict), real_dict, key=real_dict.get)

    # Generate csv
    names_df = pd.read_csv(os.path.join('db', 'player_indexing.csv'), encoding=ENCODING)
    res = []
    for i,id in enumerate(pred_highest):
        res.append({'Name': names_df[names_df['id'] == id]['name'].item(),
                    'Predicted rank': i+1,
                    'Real rank': real_highest.index(id)+1,
                    'Predicted scores': preds_dict[id],
                    'Real scores': real_dict.get(id, -999)
                }
        )
    comp_df = pd.DataFrame(data=res)
    comp_df.to_csv(os.path.join('TPnet', 'models', 'RegTrees', f'final_top{N}.csv'), index=False, encoding=ENCODING)



if __name__ == '__main__':
    # plot_scores_histogram()
    predict_season()
    merge()    
    plot_chosen_players()
    get_best_scorers()
