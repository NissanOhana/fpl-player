from abc import abstractmethod
import metaphone
import unicodedata
from typing import Tuple
from enum import Enum
import re
import pandas as pd
import os
from .fplVars import PLAYERS_ID_MAP_FILE, ENCODING, NO_PHONETIC, INVALID_ID


class IdCertainty(Enum):
    WEAK    = 0
    NORMAL  = 1
    STRONG  = 2

class Name:
    name        : str
    phonetic1   : str
    phonetic2   : str    
    id          : int
    id_certainty: IdCertainty
    
    def __init__(self, name: str) -> None:
        self.name = Name.name_preprocess(name)
        self.phonetic1, self.phonetic2 = Name.get_metaphone(self.name)
        self.set_id()
    
    @classmethod
    def name_preprocess(cls, name: str) -> str:
        def _normalize(name: str) -> str:
            normalized = unicodedata.normalize('NFD', name)
            normalized = normalized.encode('ascii', 'ignore').decode('ascii')
            return normalized
        def _remove_special_chars(name: str) -> str:
            new_name = re.sub('[\.\-_0-9 ]+', ' ', name)
            new_name = re.sub(' +', ' ', new_name)
            return new_name[:-1] if new_name.endswith(' ') else new_name
        normalized = _normalize(name)
        return _remove_special_chars(normalized.lower())
    
    @classmethod
    def get_metaphone(cls, name: str) -> Tuple[str, str]:
        splitted_name = name.split(' ')
        sorted_name = ' '.join(sorted(splitted_name))
        return metaphone.doublemetaphone(sorted_name)

    @abstractmethod
    def set_id(self) -> None:
        ...

    def get_id(self) -> int:
        return self.id
    
    def get_name(self) -> str:
        return self.name

    def get_phonetic1(self) -> str:
        return self.phonetic1
    
    def get_phonetic2(self) -> str:
        return self.phonetic2

class PlayerName(Name):
    def __init__(self, name: str, players_id_map: dict = None) -> None:
        self.players_id_map: dict = players_id_map
        super().__init__(name)
    
    def set_id(self) -> None:
        if self.players_id_map:
            self.id = self.players_id_map.get(self.get_phonetic1(), INVALID_ID)

class TeamName(Name):
    def __init__(self, name: str, teams_id_map: dict) -> None:
        self.teams_id_map: dict = teams_id_map
        super().__init__(name)
    
    def set_id(self) -> None:
        if self.teams_id_map:
            self.id = self.teams_id_map[self.get_phonetic1()]


def create_all_players_csv():
    all_players = set()
    players = pd.read_csv(os.path.join('db', 'player_indexing.csv'), encoding=ENCODING)
    players['phonetic'] = players.apply(lambda row: PlayerName(row['name']).get_phonetic1(), axis=1)
    players_dict = {k: v for k, v in zip(players['phonetic'], players['id'])}
    for season in os.listdir('db'):
        if os.path.isfile(os.path.abspath(os.path.join('db', season))):
            continue
        for gw in os.listdir(os.path.join('db', season, 'gw')):
            df = pd.read_csv(os.path.join('db', season, 'gw', gw), encoding=ENCODING).drop('name_p', axis=1)
            for name in df['name']:
                pn = PlayerName(name)
                by_word_phonetic1, by_word_phonetic2 = [], []
                for word in pn.get_name().split(' '):
                    word_pn = PlayerName(word)
                    by_word_phonetic1.append(word_pn.get_phonetic1())
                    by_word_phonetic2.append(word_pn.get_phonetic2())
                all_players.add((pn.get_name(), 
                                 pn.get_phonetic1(), 
                                 pn.get_phonetic2() if pn.get_phonetic2() else NO_PHONETIC, 
                                 tuple(by_word_phonetic1), 
                                 tuple(by_word_phonetic2) if any(by_word_phonetic2) else NO_PHONETIC,
                                 players_dict.get(pn.get_phonetic1(), INVALID_ID)))
    df = pd.DataFrame(all_players, columns=['FullName', 'FullNamePhonetic1', 'FullNamePhonetic2', 'SeparatedNamePhonetic1', 'SeparatedNamePhonetic2', 'ID'])
    df.to_csv(PLAYERS_ID_MAP_FILE, index=False, encoding=ENCODING)

if __name__ == '__main__':
    create_all_players_csv()
