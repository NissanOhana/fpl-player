import getopt
import sys
import os.path
import pandas as pd
import pickle
import random
import argparse
from multiprocessing import Process

import params as pr
from params import teams as Teams
import sub

positions = ["GK:","DEF:","MID:","FWD:"]


def commitSubs(team, buy, sell, cycle):
    trade = [[], []]
    players = sub.getPlayers(cycle)
    to_buy = players[players['id'].isin(buy)]
    to_sell = team[team['id'].isin(sell)]
    cost = to_buy['val'].sum() - to_sell['val'].sum()
    team = team.append(to_buy)
    team = team[~team['id'].isin(sell)]
    for i in range(len(buy)):
        trade[0].append([to_buy.iloc[i].id, to_buy.iloc[i].val])
        trade[1].append([to_sell.iloc[i].id, to_sell.iloc[i].val])
    return team, trade

def optimizeTeam(game):
    team = game.getTeam()
    pred = sub.evaluateTeam(team.to_numpy())
    buy, sell, opt_pred = sub.bestSub(game)
    while pred < opt_pred:
        pred = opt_pred
        team, trades = commitSubs(team, buy, sell, game.curent_cycle)
        game.setSquad(team)
        game.extra_budget -= sum([i[1] for i in trades[0]]) - sum([i[1] for i in trades[1]])
        buy, sell, opt_pred = sub.bestSub(game)
    return team
    buy, sell, opt_pred = sub.best2Subs(game)
    while pred < opt_pred:
        pred = opt_pred
        team, trades = commitSubs(team, buy, sell, game.curent_cycle)
        game.setSquad(team)
        game.extra_budget -= sum([i[1] for i in trades[0]]) - sum([i[1] for i in trades[1]])
        buy, sell, opt_pred = sub.best2Subs(game)
    return team

def genOptTeam(game,num):
    players = sub.getPlayers(game.curent_cycle)
    clubs = [k for k in range(20)]*3
    random.shuffle(clubs)
    team = pd.DataFrame(columns=['id', 'pos', 'team' , 'val' , 'ps1' , 'ps2' , 'ps3' , 'play' ])
    i = 0
    j = 0
    t = 0
    while i < 15:
        if i == 2:
            j = 1
        if i == 7:
            j = 2
        if i == 12:
            j = 3
        while i == len(team) and t<60:
            team = team.append(players.loc[(players['pos'] == j) & (players['team'] == clubs[t]) & (~players['id'].isin([p.id for i, p in team.iterrows()]))].head(1))
            t += 1
        i += 1

    while (team['val'].sum() > 100):
        out = random.randint(0, 14)
        players_pool = players[players['val'] < team.iloc[out].val]
        players_pool = players_pool[players_pool['pos'] == team.iloc[out].pos]
        players_pool = players_pool[players_pool['team'].isin(sub.fromTeams(team.to_numpy()))]
        players_pool = players_pool[~players_pool['id'].isin([p.id for i, p in team.iterrows()])]
        if players_pool.shape[0] == 0:
            continue
        team.iloc[out] = players_pool.iloc[0 if players_pool.shape[0] == 1 else random.randint(0, players_pool.shape[0] - 1)]

    tmp_game = Game('tmp_'+game.user+'_'+str(num), game.curent_cycle)
    tmp_game.setSquad(team)
    tmp_game.extra_budget = game.extra_budget - team['val'].sum()
    team = optimizeTeam(tmp_game)
    tmp_game.setSquad(team)
    storeGame(tmp_game, tmp=True)

class Game:
    def __init__(self, user='default_name', cycle=1):
        self.user = user
        self.start_cycle = cycle
        self.extra_sub = 0
        self.curent_cycle = cycle
        self.scores = [0]
        self.extra_budget = 100
        self.squad = pd.DataFrame({'id':[],'pos':[],'team':[],'val':[],'ps1':[],'ps2':[],'ps3':[],'play':[]})
        self.subs = pd.DataFrame({'id':[],'pos':[],'team':[],'val':[],'ps1':[],'ps2':[],'ps3':[],'play':[]})
        self.captain = 0
        self.last_trade = [[],[]]
        self.cheaps = []
        self.wild_card = True
        self.bench_boost = True
        self.free_hit = True
        self.triple_captain = True
        self.restore_free_hit = False

    def setSquad(self,team):
        gk = team[team['pos']==0].sort_values(by='ps1', ascending=False).head(1)
        deff = team[team['pos'] == 1].sort_values(by='ps1', ascending=False).head(3)
        fwd = team[team['pos' ]== 3].sort_values(by='ps1', ascending=False).head(1)
        rest = team[~team['id'].isin([p.id for i, p in gk.append(deff).append(fwd).iterrows()])]
        self.squad = rest[rest['pos'] != 0].sort_values(by='ps1', ascending=False).head(6).append(gk).append(deff).append(fwd)
        self.subs = team[~team['id'].isin(self.squad['id'])]
        self.captain = self.squad.sort_values(by='ps1', ascending=False)['id'].head(1).reset_index(drop = True)[0]



    def updateTeam(self):
        players = sub.getPlayers(self.curent_cycle)
        self.squad = players[players['id'].isin(self.squad['id'])]
        self.subs = players[players['id'].isin(self.subs['id'])]



    def getTeam(self):
            team = self.squad.append(self.subs)
            return team

    def teamValue(self):
        return self.getTeam()['val'].sum()

    def undoSubs(self): # buy sell
        if len(self.last_trade)==0:
            return
        team, last_trade = commitSubs(self.getTeam(), list([i[0] for i in self.last_trade[1]]), list([i[0] for i in self.last_trade[0]]), self.curent_cycle)
        self.setSquad(team)
        self.extra_budget += sum([i[1] for i in last_trade[1]]) - sum([i[1] for i in last_trade[0]])
        self.last_trade = [[],[]]

    def useCheap(self):
        self.cheaps = ''
        num_of_cheaps = self.wild_card+self.bench_boost+self.free_hit+self.triple_captain
        if num_of_cheaps == 0:
            return
        captain = self.squad['ps1'].max() if self.captain else 0
        bench_boost = self.subs['ps1'].sum() if self.bench_boost else 0
        free_hit_11 = best11(self.curent_cycle, self.getTeam()['val'].sum()+self.extra_budget) if self.free_hit and self.curent_cycle!=self.start_cycle else 0
        free_hit = 0
        if self.free_hit and self.curent_cycle!=self.start_cycle:
            free_hit = free_hit_11.squad['ps1'].sum() - self.squad['ps1'].sum()
        wc_game = Game('wc',cycle=self.curent_cycle)
        wc_game.extra_budget = self.getTeam()['val'].sum()+self.extra_budget
        wild_card = 0
        if self.wild_card and self.curent_cycle!=self.start_cycle :
            wc_game.initTeam()
            wild_card = sub.evaluateTeam(wc_game.getTeam().to_numpy()) - sub.evaluateTeam(self.getTeam().to_numpy())
        if self.curent_cycle == 19 and wild_card > 0:
            self.cheaps = 'wc'
            self.wild_card = False
            self.last_trade = [[],[]]
            self.setSquad(wc_game.getTeam())
            self.extra_budget = wc_game.extra_budget
            return
        if self.triple_captain and captain > pr.super_captain:
            self.cheaps = 'tc'
            self.triple_captain = False
            return
        if self.bench_boost and bench_boost > pr.super_bench:
            self.cheaps = 'bb'
            self.bench_boost = False
            return
        if self.wild_card and wild_card > pr.wild_card:
            self.cheaps = 'wc'
            self.wild_card = False
            self.last_trade =[[],[]]
            self.setSquad(wc_game.getTeam())
            self.extra_budget = wc_game.extra_budget
            return
        if self.free_hit and free_hit > pr.free_hit:
            self.cheaps = 'fh'
            self.free_hit = False
            self.undoSubs()
            self.restore_free_hit = True
            storeGame(self,fh = True)
            self.setSquad(free_hit_11.getTeam())
            self.extra_budget = free_hit_11.extra_budget
            return
        if num_of_cheaps == pr.total_cycles - self.curent_cycle + 1:
            if self.wild_card:
                self.cheaps = 'wc'
                self.wild_card = False
                self.last_trade = [[],[]]
                self.setSquad(wc_game.getTeam())
                self.extra_budget = wc_game.extra_budget
                return
            if self.bench_boost:
                self.cheaps = 'bb'
                self.bench_boost = False
                return
            if self.triple_captain:
                self.cheaps = 'tc'
                self.triple_captain = False
                return
            if self.free_hit:
                self.cheaps = 'fh'
                self.free_hit = False
                self.restore_free_hit = True
                self.undoSubs()
                storeGame(self, fh=True)
                self.setSquad(free_hit_11.getTeam())
                self.extra_budget = free_hit_11.extra_budget
                return

    def initTeam(self,fh = False): #todo
        p = []
        for i in range(pr.random_teams):
            p.append(Process(target=genOptTeam, args=(self,i,)))
            p[i].start()
        for i in range(pr.random_teams):
            p[i].join()
        game = loadGame('tmp_'+self.user+'_'+str(0),tmp=True)
        eval1 = sub.evaluateTeam(game.getTeam().to_numpy(), fh)
        for i in range(pr.random_teams-1):
            game2 = loadGame('tmp_' + self.user + '_' + str(i+1), tmp=True)
            eval2 = sub.evaluateTeam(game2.getTeam().to_numpy(),fh)
            if eval2 > eval1:
                game = game2
                eval1 = eval2
        self.setSquad(game.getTeam())
        self.extra_budget = game.extra_budget

    def runCycle(self):
        print(self.curent_cycle)
        self.cheaps =''
        self.updateTeam()
        sub_commit = 0
        if self.curent_cycle == 20:
            self.wild_card = True
        eval0subs = sub.evaluateTeam(self.getTeam().to_numpy())
        _buy,_sell,eval1sub = sub.bestSub(self)
        _buy2, _sell2, eval2sub = 0,0,0
        if self.extra_sub:
            _buy2, _sell2, eval2sub = sub.best2Subs(self)
        if eval2sub > (eval1sub + pr.ex_sub*(self.curent_cycle != pr.total_cycles)) :
            sub_commit = 2
        elif eval1sub > (eval0subs + pr.sub*(self.curent_cycle != pr.total_cycles)) or self.extra_sub:
            sub_commit = 1
        if sub_commit == 0:
            self.last_trade = [[],[]]
            self.extra_sub = True
            self.setSquad(self.getTeam())
        elif sub_commit == 1:
            team, self.last_trade = commitSubs(self.getTeam(),_buy,_sell,self.curent_cycle)
            self.setSquad(team)
            self.extra_budget -= sum([i[1] for i in self.last_trade[0]]) - sum([i[1] for i in self.last_trade[1]])
        else:
            team, self.last_trade = commitSubs(self.getTeam(),_buy2,_sell2,self.curent_cycle)
            self.setSquad(team)
            self.extra_budget -= sum([i[1] for i in self.last_trade[0]]) - sum([i[1] for i in self.last_trade[1]])
            self.extra_sub = False





def best11(cycle, budget):
    game = Game('best11', cycle)
    game.extra_budget = budget
    game.initTeam(fh = True)
    return game



def updateResult(game):
    res = pd.read_csv("../db/simulation/realdata/gw{}.csv".format(game.curent_cycle))
    score = res[res['id'].isin([p.id for i,p in game.squad.iterrows()])]['total_points'].sum()
    score += (res[res['id'] == game.captain].iloc[0].total_points *(1 + ('tc' == game.cheaps)))
    if 'bb' == game.cheaps:
        score += res[res['id'].isin([p.id for i,p in game.subs.iterrows()])]['total_points'].sum()
    game.scores.append(score)

def storeGame(game, tmp=False, fh=False):
    path = './tmp/'+game.user+'.game' if tmp else ('./games/'+'free_hit_restore_'+game.user+'.game' if fh else './games/'+game.user+'.game')
    f = open(path, 'wb')
    pickle.dump(game,f)
    f.close()

def loadGame(user, tmp=False):
    f = open('./tmp/'+user+'.game', 'rb') if tmp else open('./games/'+user+'.game', 'rb')
    game = pickle.load(f)
    f.close()
    return game

def printCycle(log, game):
    p_stats = pd.read_csv("./logs/" + game.user + '_players_stats.csv')
    g_stats = pd.read_csv("./logs/" + game.user + '_game_stats.csv')
    num_of_subs = len(game.last_trade[0])
    players = pd.read_csv("../db/player_indexing.csv")
    scores = pd.read_csv("../db/simulation/realdata/gw{}.csv".format(game.curent_cycle))
    g_stats = g_stats.append({'gw': game.curent_cycle, 'cheap': game.cheaps, 'captain': game.captain,
                    'pred': sum(game.squad['ps1']) + sum(game.subs['ps1'])*(game.cheaps == 'bb') + sum(game.squad[game.squad['id'] == game.captain]['ps1'])*(1 + (game.cheaps == 'tc')),
                    'score': game.scores[-1],
                    'buy1': '' if not num_of_subs else str(players[players['id'] == game.last_trade[0][0][0]].iloc[0]['name']),
                    'buy1_price':0 if not num_of_subs else game.last_trade[0][0][1],
                    'sold1': '' if not num_of_subs else str(players[players['id'] == game.last_trade[1][0][0]].iloc[0]['name']),
                    'sold1_price': 0 if not num_of_subs else game.last_trade[1][0][1],
                    'buy2': '' if num_of_subs!=2 else str(players[players['id'] == game.last_trade[0][1][0]].iloc[0]['name']),
                    'buy2_price': 0 if num_of_subs!=2 else  game.last_trade[0][1][1],
                    'sold2': '' if num_of_subs!=2 else str(players[players['id'] == game.last_trade[1][1][0]].iloc[0]['name']),
                    'sold2_price': 0 if num_of_subs!=2 else game.last_trade[1][1][1],
                    'team_value': game.teamValue(),
                    'extra_budget': game.extra_budget},ignore_index = True)
    g_stats.to_csv("./logs/" + game.user + "_game_stats.csv",index = False)

    if game.start_cycle == game.curent_cycle: #first time
        log.write("####################################################\n")
        log.write("user: {}, start cycle:{}\n".format(game.user, game.start_cycle))
        log.write("####################################################\n")
    log.write("cycle num: {}\n".format(game.curent_cycle))
    log.write("unused budget: {}\n".format(game.extra_budget))
    log.write("total team value: {}\n".format("%.2f" % game.teamValue()))
    log.write("last cycle score: {} ({} in total)\n".format(game.scores[-1],sum(game.scores)))
    if game.cheaps == 'tc':
        log.write("triple captain cheap was used!\n")
    elif game.cheaps == 'bb':
        log.write("bench boost cheap was used!\n")
    elif game.cheaps == 'wc':
        log.write("wild card cheap was used!\n")
        game.extra_sub = False
    elif game.cheaps == 'fh':
        log.write("free hit cheap was used!\n")
    if num_of_subs and game.start_cycle != game.curent_cycle:
        log.write("{} trade(s) was commited\n".format(num_of_subs))
        log.write("bought:\n")
        for i in range(num_of_subs):
            log.write("{} for {}\n".format(str(players[players['id'] ==game.last_trade[0][i][0]].iloc[0]['name']),game.last_trade[0][i][1]))
        log.write("sold:\n")
        for i in range(num_of_subs):
            log.write("{} for {}\n".format(str(players[players['id'] ==game.last_trade[1][i][0]].iloc[0]['name']),game.last_trade[1][i][1]))
    if game.extra_sub:
        log.write("no subs was commited, extra sub is saved for next cycle!\n")
    log.write("-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -\n")
    log.write("captain is {}\n".format(str(players[players['id'] ==game.captain].iloc[0]['name'])))
    for pos in range(4):
        log.write(positions[pos]+"\n")
        for j,player in game.squad[game.squad['pos'] == pos].iterrows():
            log.write("{}   team: {}    score: {}({})    value: {}\n".format(str(players[players['id'] ==player['id']].iloc[0]['name']),\
                                                                    Teams[int(player['team'])],\
                                                                    str(scores[scores['id']==player['id']].iloc[0]['total_points']),\
                                                                    str(player['ps1']),str(player['val'])))
            p_stats = p_stats.append({'id' : str(players[players['id'] == player['id']].iloc[0]['name']),
                                      'gw' : game.curent_cycle,
                                      'pos' : pos,
                                      'team' : Teams[int(player['team'])],
                                      'val' : player['val'],
                                      'pred' : player['ps1'],
                                      'score' : (scores[scores['id']==player['id']].iloc[0]['total_points']),
                                      'captain' : player['id'] == game.captain,
                                      'squad':1},
                                     ignore_index = True)
    log.write("subs:\n")
    for i, player in game.subs.iterrows():
        log.write("{}   team: {}    score: {}({})    value: {}\n".format(str(players[players['id'] ==player['id']].iloc[0]['name']), Teams[int(player['team'])], \
                                                                             str(scores[scores['id'] == player['id']].iloc[0].total_points), \
                                                                             str(player['ps1']), str(player['val'])))
        p_stats = p_stats.append({'id': str(players[players['id'] == player['id']].iloc[0]['name']),
                                  'gw': game.curent_cycle,
                                  'pos': player['pos'],
                                  'team': Teams[int(player['team'])],
                                  'val': player['val'],
                                  'pred': player['ps1'],
                                  'score': scores[scores['id'] == player['id']].iloc[0]['total_points'],
                                  'captain': False,
                                  'squad': 0},
                                 ignore_index=True)
    log.write("####################################################\n")
    p_stats.to_csv("./logs/" + game.user + "_players_stats.csv", index = False)

def printSum(user):
    game = loadGame(user)
    log = open("./logs/" + game.user + '.txt', 'a')
    log.write("####################################################\n")
    log.write("####################################################\n")
    log.write("SEASON SUMMERY:\n")
    log.write("total score is {}\n".format(sum(game.scores)))
    log.write("####################################################\n")
    log.write("####################################################\n")

    log.close()

def runCycle(user):
    game = loadGame(user)
    if game.restore_free_hit:
        pre_game = loadGame('free_hit_restore_'+user)
        game.setSquad(pre_game.getTeam())
        game.extra_budget = pre_game.extra_budget
        game.extra_sub = False
        game.restore_free_hit = False
    if game.curent_cycle > pr.total_cycles:
        return 0
    log = open("./logs/" + game.user + '.txt', 'a')
    game.runCycle()
    game.useCheap()
    updateResult(game)
    printCycle(log, game)
    log.close()
    game.curent_cycle+=1
    storeGame(game)
    return 1

def initPlayer(user, cycle):
    game = Game(user, cycle)
    log = open("./logs/" + game.user + ".txt", "w")
    p_df = pd.DataFrame(columns=['id','gw' ,'pos', 'team', 'val', 'pred', 'score', 'captain', 'squad'])
    g_df = pd.DataFrame(columns=['gw', 'cheap', 'captain', 'pred', 'score', 'buy1', 'buy1_price', 'sold1', 'sold1_price', 'buy2', 'buy2_price', 'sold2', 'sold2_price' , 'team_value', 'extra_budget'])
    p_df.to_csv("./logs/" + game.user + "_players_stats.csv", index = False)
    g_df.to_csv("./logs/" + game.user + "_game_stats.csv", index = False)
    game.initTeam()
    game.useCheap()
    updateResult(game)
    printCycle(log, game)
    log.close()
    game.curent_cycle+=1
    storeGame(game)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('user', help="player user name, must be unique", type=str ,metavar='user')
    parser.add_argument('-c','--cycle', help="start cycle you want to start in. must be between 1 and {}, defaulte is 1".format(pr.total_cycles), type=int, metavar='cycle', default=1)
    parser.add_argument('-r','--reset', help="if you want to reset an existing user", action='store_true')
    parser.add_argument('-s','--simulate', help="if you want to run a simulate the rest of the season", action='store_true')
    return parser

if __name__ == "__main__":
    parser = parse()
    args = parser.parse_args()
    if args.reset:
        initPlayer(args.user,args.cycle)
    if args.simulate:
        if not os.path.isfile('./games/'+args.user+'.game'):
            initPlayer(args.user, args.cycle)
        while runCycle(args.user):
            pass
        printSum(args.user)

    else:
        if os.path.isfile('./games/'+args.user+'.game'):
            if not runCycle(args.user):
                printSum(args.user)
        else:
            initPlayer(args.user, args.cycle)
