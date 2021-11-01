import pandas as pd
from datetime import datetime
from textblob import TextBlob
from utils.namesHandler import PlayerName

days_prefix = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu',  'Fri', 'Sat']


def clean_raw_tweets(data_path):
    # Only tweets that start with date are valid for processing -
    with open(f'./{data_path}.csv', 'r') as fp:
        lines = fp.readlines()
    x = lines[-1]
    with open(f'{data_path}.csv', 'w') as fp:
        for i, line in enumerate(lines):
            if i == 0:
                fp.write(line)
            splited = line.split(' ')
            if len(splited) > 6:
                prefix = splited[0]
                if prefix in days_prefix:
                    fp.write(line)


def build_training_dataset():
    df_tweets = pd.read_csv('./all_media_tweets_until_2020.csv', encoding="ISO-8859-1")
    df_tweets.drop(labels=df_tweets.columns.difference(['created_at', 'full_text']),
                   axis='columns',
                   inplace=True)

    df_player_code = pd.read_csv('./../db/player_indexing.csv', encoding="ISO-8859-1")
    # Add phonetic to players code df:
    df_player_code['phonetic'] = PlayerName(df_player_code['name']).get_phonetic1()

    # Loader gw dates:
    df_gw_dates = pd.read_csv('../db/gw_202021_dates.csv', encoding="ISO-8859-1")

    # Convert time to datetime:

    for i, row in df_tweets.iterrows():
        new_datetime = datetime.strftime(datetime.strptime(row['created_at'], '%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d %H:%M:%S')
        df_tweets.at[i, 'created_at'] = new_datetime
        print()

    df_tweets['created_at'] = pd.to_datetime(df_tweets.created_at)
    df_tweets.sort_values(by=['created_at'], inplace=True)

    # Label tweet with player name -
    for i, row in df_tweets.iterrows():
        text = row['full_text'].split(' ')
        keys = text

        # We choose tweets in which only one player appears.
        players_in_tweets = 0
        possibly_player_name_p = ''
        # Take name:
        for word_idx in range(len(keys) - 1):
            possibly_player_name = ''.join(keys[word_idx]) + ' '.join(keys[word_idx + 1])
            possibly_player_name_p = PlayerName(possibly_player_name).get_phonetic1()
            if possibly_player_name_p not in df_player_code:
                players_in_tweets += 1

        if players_in_tweets != 1:
            continue  # Iterate to the next tweet.
        else:
            # Label the tweet ->
            row['polarity'] = TextBlob(text).polarity
            row['subjectivity'] = TextBlob(text).subjectivity
            row['label_player_name_phonetic'] = possibly_player_name_p



if __name__ == '__main__':
    # clean_raw_tweets('bbc')
    # clean_raw_tweets('sky_sports')
    # clean_raw_tweets('andy_fpl')
    # clean_raw_tweets('fpl_scout')
    # clean_raw_tweets('pl_o')

    # build_training_dataset()
    pass