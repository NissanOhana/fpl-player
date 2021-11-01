# General porpuse
from typing import List
import os
import re
import sys
from time import sleep
import pickle
# Data processing
import pandas as pd
# Web scrapping
import requests
import json
import urllib
import bs4
from bs4 import BeautifulSoup as bs
import googlesearch
# Project specific
sys.path.append(os.getcwd())
from namesHandler import PlayerName
from fplVars import ENCODING, PLAYERS_STATS_FILE, PLAYERS_DICT, MISSING_PLAYERS_STATS_FILE

SEASON_CODE = {'19-20': '274',
               '18-19': '210',
               '17-18': '79'
}
SEASON_TO_FOLLOWING = {'19-20': 3, # stats of season 19-20 is relevant for 20-21 (#3)
                       '18-19': 2,
                       '17-18': 1
}
BASE_URL = 'https://www.premierleague.com'
DEFAULT_NAN_VALUE = 0
MAX_REQUESTS = 5

def get_players_urls(season: str) -> List[str]:
    with open(os.path.join('db', f'{season}.html'), 'r', encoding='UTF-8') as f:
        data = f.read()
        soup = bs(data, 'lxml')   
        players = soup.find_all('a', {'class': 'playerName'}, href=True)
        return [player['href'] for player in players]

def overview2stats(urls: List[str], season: str) -> List[str]:
    return [f'{url.replace("overview", "stats")}?co=1&se={SEASON_CODE[season]}'
            for url in urls]

def to_float(s: str) -> float:
    s = s.replace(',', '')
    if '%' in s:
        s = s.replace('%', '')
    return float(s)

def add_stats(topStats: bs4.element.ResultSet, stats) -> None:
    for stat in topStats:
        text = stat.find_all('span', {'class': 'stat'})[0].text
        splitted = re.sub('\s+', ' ', text).strip().split()
        key = '_'.join(splitted[:-1])
        val = splitted[-1]
        stats[key.lower()] = to_float(val)


def scrap_stats(players_stats_urls: List[str]) -> pd.DataFrame:
    df = pd.DataFrame()
    for url in players_stats_urls:
        req_counter = 0
        r = requests.get(url)
        while r.status_code != 200 and req_counter < MAX_REQUESTS:
            sleep(1)
            r = requests.get(url)
        if r.status_code != 200:
            print(f"Failed to get data from {url} after {MAX_REQUESTS} times")
            continue    

        soup = bs(r.text, 'lxml')
        player_name = soup.find_all('title')[0].text.split(' Statistics')[0]
        user_stats = {'Name': player_name,
                      'id': PlayerName(player_name, PLAYERS_DICT).get_id()}
        add_stats(soup.find_all('div', {'class': 'topStat'}), user_stats)
        add_stats(soup.find_all('div', {'class': 'normalStat'}), user_stats)
        df = df.append(user_stats, ignore_index=True)
    return df.fillna(DEFAULT_NAN_VALUE)


def season_str_to_following_int(season: str) -> int:
    return SEASON_TO_FOLLOWING[season]
def int_to_prev_season(season: int) -> str:
    return {v: k for k,v in SEASON_TO_FOLLOWING.items()}[season]

def generate_csv():
    df = pd.DataFrame()
    for season in SEASON_CODE:
        players_urls = get_players_urls(season)
        players_stats_urls = overview2stats(players_urls, season)
        stats = scrap_stats(players_stats_urls)
        stats['season'] = season_str_to_following_int(season)
        df = df.append(stats)
        df.to_csv(PLAYERS_STATS_FILE, index=False, encoding=ENCODING)

def refetch_missings(refetch: bool):
    df = pd.DataFrame()
    player_urls = {}
    if refetch:
        missing_players_file = os.path.join('db', 'missing_players.txt')
        with open(missing_players_file, 'r', encoding=ENCODING) as f:
            for player_name in f.readlines():
                player_name = player_name.strip()
                search_string = f'premier league stats {player_name}'
                plurl = list(googlesearch.search(search_string, num=4, stop=4, pause=0.5))
                url = next((url for url in plurl if 'premierleague.com' in url and 'stats' in url), None)
                if url:
                    player_urls[player_name] = url
                else:
                    print(f'Could not find stats for {player_name}')
            with open('missing_urls.txt', 'w') as f:
                print(json.dumps(player_urls, indent=4), file=f)
            with open('missing_urls.pkl', 'wb') as f:
                pickle.dump(player_urls, f)
    else:
        with open('missing_urls.pkl', 'rb') as f:
            player_urls = pickle.load(f)

    for season in SEASON_CODE:
        season_urls = [f'{url}?co=1&se={SEASON_CODE[season]}' for url in player_urls.values()]
        stats = scrap_stats(season_urls)
        stats['season'] = season_str_to_following_int(season)
        df = df.append(stats)
        df.to_csv(MISSING_PLAYERS_STATS_FILE, index=False, encoding=ENCODING)

def fix_missings():
    df = pd.read_csv(os.path.join('db', 'dataset.csv'), encoding=ENCODING)
    missings_df = df[df.isnull().any(axis=1)]
    missings_set = set(missings_df.apply(lambda row: (row['id'], row['season']), axis=1))
    players = {}
    for season in SEASON_CODE:
        for player_url in get_players_urls(season):
            name = re.search('players/[0-9]+/(.*)/overview', urllib.parse.unquote(player_url)).group(1)
            players[PlayerName(name, PLAYERS_DICT).get_id()] = player_url
    
    missings = {season: [] for season in SEASON_CODE}
    for miss_id, miss_season in list(missings_set):
        if players.get(int(miss_id), None):
            missing_url = overview2stats([players[int(miss_id)]], int_to_prev_season(miss_season))
            missings[int_to_prev_season(miss_season)].extend(missing_url)
    
    df = pd.DataFrame()
    for season, urls in missings.items():
        stats = scrap_stats(urls)
        stats['season'] = season_str_to_following_int(season)
        df = df.append(stats)
        # df.to_csv(MISSING_PLAYERS_STATS_FILE, index=False, encoding=ENCODING)

    df_orig = pd.read_csv(PLAYERS_STATS_FILE, encoding=ENCODING)
    combined = pd.concat([df_orig, df])
    combined.to_csv(PLAYERS_STATS_FILE, index=False, encoding=ENCODING)
    pass


if __name__ == '__main__':
    generate_csv()
    refetch_missings(refetch=True)
    # fix_missings()
    