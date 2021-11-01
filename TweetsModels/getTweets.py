import tweepy
import pandas as pd
from TweetsModels import twitterSecret as ts

def get_tweets(ds_name: str):
    twitter_username = {'injury_tweets': 'premierinjuries',
                        'media_sky_sport': 'SkySportsPL',
                        '': ''}
    auth = tweepy.OAuthHandler(ts.consumer_key, ts.consumer_key_secret)
    auth.set_access_token(ts.access_token, ts.access_token_secret)
    api = tweepy.API(auth)

    print('Loading...')
    tw_cursor = tweepy.Cursor(api.user_timeline, id=twitter_username[ds_name], tweet_mode='extended').items(3500)
    json_data = [tw._json for tw in tw_cursor]
    df = pd.json_normalize(json_data)
    df.to_csv(f'./{ds_name}_raw.csv', index=False)
    print('DONE! :)')

if __name__ == '__main__':
    get_tweets("injury_tweets")
