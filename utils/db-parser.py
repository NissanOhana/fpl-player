import pandas as pd
from typing import Tuple
# from namesHandler import PlayerName


def db_build_players_indexing():
    player_raw_df = []
    for season in ['2018-19', '2019-20', '2020-21']:
        raw_players_table = pd.read_csv(f'../{season}/players_raw.csv', encoding="ISO-8859-1")
        raw_players_table.drop(
            labels=raw_players_table.columns.difference(['first_name', 'second_name', 'code']),
            axis='columns',
            inplace=True)
        raw_players_table['name'] = raw_players_table['first_name'] + \
                                           ' ' + \
                                           raw_players_table['second_name']
        raw_players_table.drop(labels=['first_name', 'second_name'], axis='columns', inplace=True)
        player_raw_df.append(raw_players_table.copy())

    player_indexing = pd.concat(player_raw_df).drop_duplicates()
    player_indexing.to_csv(path_or_buf='../player_indexing.csv', index=False)

def db_build_team_code_map(season: str):
    raw_team_df = pd.read_csv(f'../db/{season}/teams_raw.csv', encoding="ISO-8859-1")
    raw_team_df.drop(raw_team_df.columns.difference(['code', 'id']), inplace=True)
    raw_team_df.to_csv(path_or_buf=f'../db/{season}/team_code_map.csv')

def db_fix_2021season():
    for i in range(1, 39):
        gw_path = f'../2020-21/gw/gw{i}.csv'
        gw = pd.read_csv(gw_path, encoding="ISO-8859-1")
        try:
            gw.drop(labels=['position', 'team', 'xP'], axis='columns', inplace=True)  # Old labels.
            gw.to_csv(gw_path, index=False, encoding="ISO-8859-1")
        except Exception:
            print('can not edit csv file.')
    print('done fixing 20-21 season db')

def db_fix_prev_seasons(season: str):
    print(f'-------------- DB PARSER: Edit Season {season} DB --------------')
    if (season != '2018-19') and (season != '2019-20'):
        return

    gw2021 = pd.read_csv('../2020-21/gw/gw1.csv', encoding="ISO-8859-1")
    db_col = list(gw2021.columns)

    for i in range(1, 39):
        gw_path = f'../{season}/gw/gw{i}.csv'
        gw = pd.read_csv(gw_path, encoding="ISO-8859-1")
        try:
            gw.drop(labels=gw.columns.difference(db_col), axis='columns', inplace=True)
            gw.to_csv(gw_path, index=False)
        except Exception:
            print('can not edit csv file.')
        print(f'edit {season} gw{i} - Success!')

def db_diff_seasons_check():
    gw2018 = pd.read_csv('../2018-19/gw/gw1.csv', encoding="ISO-8859-1")
    gw2019 = pd.read_csv('../2019-20/gw/gw1.csv', encoding="ISO-8859-1")
    gw2020 = pd.read_csv('../2020-21/gw/gw1.csv', encoding="ISO-8859-1")

    if list(gw2018.columns) == list(gw2019.columns) == list(gw2020.columns):
        print('----- We are good to go! :) ------')


def extract_cost(num: int) -> Tuple[int, int]:
    return num // 10 * 10, num % 10


def db_seasons_add_inject_date(season: str):
    """
    Adding to 2020-2021 DB:
    * Formatting Player name
    * Player code (indexing)
    * Player cost
    * Player Position
    * Player Current team
    * Player Opponent team this GW
    :return: None.
    Write to season/gw/gw_number updated csv file
    """
    player_indexing = pd.read_csv('../player_indexing.csv', encoding="ISO-8859-1")
    player_indexing['name_p'] = player_indexing.apply(lambda x: PlayerName(x['name']).get_phonetic1(), axis='columns')
    raw_players_table = pd.read_csv(f'../{season}/players_raw.csv', encoding="ISO-8859-1")
    raw_players_table.drop(labels=raw_players_table.columns.difference(['code', 'team_code', 'now_cost', 'element_type']),
                           axis='columns',
                           inplace=True)

    # NOT NEEDED FOR NOW:
    # # Format cost to start of the season cost:
    # for index, row in raw_players_table.iterrows():
    #     round_price, unit_digit = extract_cost(row['now_cost'])
    #     row['now_cost'] = round_price + (0 if 0 <= unit_digit < 5 else 5)
    #     row['now_cost'] = row['now_cost'] if row['now_cost'] >= 40 else 40  # Minimum price

    # Add init price, player code and player current team code:
    players_extra_data_df = raw_players_table.join(player_indexing.set_index('code'), on='code')
    players_extra_data_df.drop(columns=['name'], inplace=True)
    team_code_mapping_df = pd.read_csv(f'../{season}/teams_code_map.csv', encoding="ISO-8859-1")
    team_code_mapping = dict(zip(team_code_mapping_df.id, team_code_mapping_df.code))

    for i in range(1, 39):
        gw_path = f'../{season}/gw/gw{i}.csv'
        gw = pd.read_csv(gw_path, encoding="ISO-8859-1")
        gw['name_p'] = gw.apply(lambda x: PlayerName(x['name']).get_phonetic1(), axis='columns')
        gw_new = gw.join(players_extra_data_df.set_index('name_p'), on='name_p')
        gw_new.rename(columns={'now_cost': 'init_price', 'value': 'current_price', 'element_type': 'position'},
                      inplace=True)
        # Convert team id to universal code:
        gw_new['opponent_team'] = gw_new.apply(lambda row_entry: team_code_mapping[row_entry['opponent_team']],
                                               axis='columns')
        gw_new.to_csv(f'../{season}/gws_new/gw{i}.csv', index=False)


def prod_play_column(value: int) -> int:
    return 1 if value == 1 else 0


def db_ready_for_simulations():
    """
    Format the simulations gw csv for the testing
    """
    raw_players_table = pd.read_csv(f'../db/2020-21/players_raw.csv', encoding="ISO-8859-1")
    raw_players_table.drop(labels=raw_players_table.columns.difference(['code', 'element_type']),
                           axis='columns',
                           inplace=True)
    raw_players_table.rename(columns={'code': 'player_code'},
                             inplace=True)

    team_df = pd.read_csv(f'../db/simulation/team_code_map_prod.csv', encoding="ISO-8859-1")

    for i in range(1, 39):
        gw_path = f'../db/simulation/predictions/gw{i}.csv'
        gw = pd.read_csv(gw_path, encoding="ISO-8859-1")
        gw_new = gw.join(raw_players_table.set_index('player_code'), on='player_code')
        gw_new['pos'] = gw_new.apply(lambda x: (x['element_type'] - 1), axis='columns')
        gw_new['play'] = gw_new.apply(lambda x: prod_play_column(x['availability_prediction']),
                                             axis='columns')
        gw_new = gw_new.join(team_df.set_index('team_code'), on='team_code')
        gw_new.rename(columns={'current_price': 'val', 'player_code': 'id'},
                      inplace=True)
        gw_new.drop(labels=['element_type', 'availability_prediction', 'team_code'], axis='columns', inplace=True)
        if (i == 1):
            gw_new['hype'] = 0

        new_cols = ['id', 'pos', 'team', 'val', 'play', 'hype']
        gw_new = gw_new[new_cols]
        # Convert team id to universal code:

        gw_new.to_csv(gw_path, index=False)


if __name__ == '__main__':
    print('-------------- DB PARSER --------------\n Choose Function from above')
    """
    First, fix DB:
    """
    # DONE -> db_fix_2021season()
    # DONE -> db_fix_prev_seasons('2018-19')
    # DONE -> db_fix_prev_seasons('2019-20')
    #db_diff_seasons_check()

    """
    Add data:
    """
    #db_seasons_add_inject_date('2020-21')
    #db_seasons_add_inject_date('2018-19')
    #db_seasons_add_inject_date('2019-20')

    """
    Ready for production -
    """
    db_ready_for_simulations()





