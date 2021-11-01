import pandas as pd
from datetime import datetime
from utils.namesHandler import PlayerName

days_prefix = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu',  'Fri', 'Sat']
injury_update_key = 'Update:'
ruled_out_key = 'No Return Date'

def clean_raw_tweets():
    lines = []
    with open('./injury_tweets_raw.csv', 'r') as fp:
        lines = fp.readlines()
    x = lines[-1]
    with open('injury_tweets_clean.csv', 'w') as fp:

        for i, line in enumerate(lines):
            if i == 0:
                fp.write(line)
            splited = line.split(' ')
            if len(splited) > 6:
                prefix, fpl_injuery_key = splited[0], splited[6]
                if (prefix in days_prefix) and fpl_injuery_key == injury_update_key:
                    fp.write(line)

def build_table():
    ruled_out_df = pd.DataFrame(columns=['gw', 'player'])
    df = pd.read_csv('./injury_tweets_clean.csv', encoding="ISO-8859-1")
    df_gw_dates = pd.read_csv('../db/gw_202021_dates.csv', encoding="ISO-8859-1")
    df.drop(labels=df.columns.difference(['created_at', 'full_text']),
            axis='columns',
            inplace=True)

    # Convert time to datetime:
    for i, row in df.iterrows():
        new_datetime = datetime.strftime(datetime.strptime(row['created_at'], '%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d %H:%M:%S')
        df.at[i, 'created_at'] = new_datetime
        print()
    df['created_at'] = pd.to_datetime(df.created_at)
    df.sort_values(by=['created_at'], inplace=True)

    for i, row in df.iterrows():
        keys = row['full_text'].split(' ')

        # If 100% ruled out, we want the data:
        if ruled_out_key not in row['full_text']:
            continue

        # Take name:
        player_name_raw = []
        for word in keys:
            if word == '#FPL' or word == 'Update:':
                continue
            if word == '-':
                break
            player_name_raw.append(word)

        player_name = ' '.join(player_name_raw)
        player_name = PlayerName(player_name).get_phonetic1()

        # Take GW:
        ruled_out_gw = 0
        for gw_num in range(len(df_gw_dates) - 1):
            if df_gw_dates.loc[i, 'date'] <= row['created_at'] <= df_gw_dates.loc[i+1, 'date']:
                ruled_out_gw = df_gw_dates.loc[i, 'gw']

        ruled_out_df.append(pd.DataFrame([ruled_out_gw, player_name]), ignore_index=True)

    print('building ruled out table - done :)')

if __name__ == '__main__':
    #clean_raw_tweets()
    build_table()
