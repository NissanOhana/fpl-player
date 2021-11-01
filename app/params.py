from enum import Enum


class Columns(Enum):
    id = 0
    pos = 1
    team = 2
    val = 3
    ps1 = 4
    ps2 = 5
    ps3 = 6
    play = 7

teams = ['Arsenal',
         'Aston Villa',
         'Brighton',
         'Burnley',
         'Chelsea',
         'Crystal Palace',
         'Everton',
         'Fulham',
         'Leeds United',
         'Leicester',
         'Liverpool',
         'Manchester City',
         'Manchester United',
         'Newcastle United',
         'Sheffield United',
         'Southampton',
         'Tottenham Hotspur',
         'West Bromwich Albion',
         'West Ham United',
         'Wolverhampton']

squad_weight = [1.0, 0.4, 0.1]
subs_weight = [0.0, 0.4, 0.1]
fh_sq_weight = [1, 0, 0, 0, 0]
fh_b_weight = [0, 0, 0, 0, 0]
ex_sub = 6
sub = 5
random_teams = 15
super_captain = 13
super_bench = 12
wild_card = 30
free_hit = 30
hype = 1.0
budget = 100
total_cycles = 38
