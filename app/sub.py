
import params as pr
from app import Game as Game
import numpy as np
import pandas as pd
from params import Columns as col
import random



def getPlayers(cycle):
    df = pd.read_csv("../db/simulation/merged_predictions/gw{}.csv".format(cycle))
    df.ps1 += df.ps1*df.hipe*pr.hype
    df.val /= 10
    df.drop(['hipe'], axis = 1, inplace=True)
    return df


def fromTeams(team):
    from_teams=[]
    for i in range(20):
        if len([x for x in team if x[col.team.value]==i]) < 3 :
            from_teams.append(i)
    return from_teams

def evaluateTeam(team, fh = False):
    sq_weight = pr.fh_sq_weight if fh else pr.squad_weight
    b_weight = pr.fh_b_weight if fh else pr.subs_weight
    gk = sorted([i for i in team if i[col.pos.value] == 0] ,key=lambda x:x[col.ps1.value],reverse=True)[:1]
    deff = sorted([i for i in team if i[col.pos.value] == 1] ,key=lambda x:x[col.ps1.value],reverse=True)[:3]
    fwd = sorted([i for i in team if i[col.pos.value] == 2] ,key=lambda x:x[col.ps1.value],reverse=True)[:1]
    rest = sorted([i for i in team if i[col.id.value] not in [x[col.id.value] for x in (gk+deff+fwd)]],key=lambda x:x[col.ps1.value],reverse=True)
    other = sorted([x for x in rest if x[col.pos.value] != 0],key=lambda x:x[col.ps1.value],reverse=True)[:6]
    squad = other+gk+deff+fwd
    squad_calc = 0
    for i in range(len(squad)):
        for j in range(3):
            squad_calc += squad[i][col.ps1.value+j]*sq_weight[j]*squad[i][col.play.value]
    subs = [x for x in team if x[col.id.value] not in [y[col.id.value] for y in squad]]
    subs_calc = 0
    for i in range(len(subs)):
        for j in range(3):
            subs_calc += subs[i][col.ps1.value+j]*b_weight[j]
    return squad_calc+subs_calc

def bestSub(game):
    players = getPlayers(game.curent_cycle).to_numpy()
    team = game.getTeam().to_numpy()
    players = [p for p in players if p[col.id.value] not in [x[col.id.value] for x in team]]
    max_score = 0
    fired,hired = 0,0
    for i in range(len(team)):
        tmp_team = team[[g != i for g in range(len(team))]]
        budget = game.extra_budget + team[i][col.val.value]
        pos = team[i][col.pos.value]
        players_pool = np.array(players)[[p[col.pos.value] == pos for p in players]]
        players_pool = players_pool[[p[col.team.value] in fromTeams(tmp_team) for p in players_pool]]
        players_pool = players_pool[[p[col.val.value] <= budget for p in players_pool]]
        players_pool = players_pool[[p[col.id.value] not in [k[col.id.value] for k in team] for p in players_pool]]
        for p1 in range(len(players_pool)):
            new_team = np.concatenate((tmp_team, [players_pool[p1]]))
            score = evaluateTeam(new_team)
            if score > max_score:
                max_score = score
                hired = list([players_pool[p1][col.id.value]])
                fired = list([team[i][col.id.value]])
    return hired, fired, max_score

def best2Subs(game):
    players = getPlayers(game.curent_cycle).to_numpy()
    team = game.getTeam().to_numpy()
    players = [p for p in players if p[col.id.value] not in [x[col.id.value] for x in team]]
    max_score = 0
    fired,hired = 0,0
    for i in range(len(team)-1):
        for j in range(i+1,15):
            tmp_team = team[[g not in [i,j] for g in range(15)]]
            budget = game.extra_budget + team[i][col.val.value]+ team[j][col.val.value]
            pos = [team[i][col.pos.value],team[j][col.pos.value]]
            players_pool = np.array(players)[[p[col.pos.value] in pos for p in players]]
            for p1 in range(len(players_pool)-1):
                new_team = np.concatenate((tmp_team ,[players_pool[p1]]))
                for p2 in range(p1,len(players_pool)):
                    if not ((pos[0] == players_pool[p1][col.pos.value] and pos[1] == players_pool[p2][col.pos.value]) or \
                            (pos[1] == players_pool[p1][col.pos.value] and pos[0] == players_pool[p2][col.pos.value])):
                        continue
                    if players_pool[p1][col.val.value]+players_pool[p2][col.val.value] > budget:
                        continue
                    if players_pool[p2][col.team.value] not in fromTeams(new_team):
                        continue
                    new_team = np.concatenate((new_team ,[players_pool[p2]]))
                    score = evaluateTeam(new_team)
                    if score > max_score:
                        max_score = score
                        hired = [players_pool[p1][col.id.value],players_pool[p2][col.id.value]]
                        fired = [team[i][col.id.value],team[j][col.id.value]]
    return hired, fired, max_score
