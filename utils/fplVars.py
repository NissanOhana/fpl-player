import os
import pandas as pd

TEAMS_ID_MAP_FILE           = os.path.join('db', 'all_teams.csv')
PLAYERS_ID_MAP_FILE         = os.path.join('db', 'all_players.csv')
PLAYERS_STATS_FILE          = os.path.join('db', 'players_general_stats.csv')
MISSING_PLAYERS_STATS_FILE  = os.path.join('db', 'missing_players_general_stats.csv')
ENCODING                    = 'iso-8859-1'
NO_PHONETIC                 = 'NaN'
INVALID_ID                  = -1
PLAYERS_DF                  = pd.read_csv(PLAYERS_ID_MAP_FILE, encoding=ENCODING)
PLAYERS_DICT                = {k: v for k, v in zip(PLAYERS_DF['FullNamePhonetic1'], PLAYERS_DF['ID'])}
ALL_PLAYERS_PHONETIC        = list(PLAYERS_DF['FullName'])
DIDNT_PLAY_VALUE            = -999
DATASET_DICT                = {'train'           : os.path.join('db', 'train_dataset.csv'),
                               'train_merged'    : os.path.join('db', 'train_dataset_merged.csv'),
                               'test'            : os.path.join('db', 'test_dataset.csv'),
                               'test_merged'     : os.path.join('db', 'test_dataset_merged.csv'),
                               'validate'        : os.path.join('db', 'validate_dataset.csv'),
                               'validate_merged' : os.path.join('db', 'validate_dataset_merged.csv'),
                               'all'             : os.path.join('db', 'all_dataset.csv'),
                               'all_merged'      : os.path.join('db', 'all_dataset_merged.csv')
}
GWS_PER_SEASON              = 38
GWS_TO_PREDICT              = 3     # How many gws in advance we try to predict (> 0)
DROPPED_COLS                = ['name', 'name_p', 'kickoff_time', 'total_points', 'Name', 'round', 'season']
TRAIN_SEASONS               = [1, 2]
TEST_SEASONS                = [3]
RANDOM_STATE                = 42