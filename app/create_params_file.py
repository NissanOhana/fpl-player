import argparse


squad_weight = "1,0.4,0.1"
subs_weight = "0,0.4,0.1"
fh_sq_weight = [1,0,0,0,0]
fh_b_weight = [0,0,0,0,0]
ex_sub = 6
sub = 5

captan_weight = 1
random_teams = 15

super_captain = 13
super_bench = 12
wild_card = 30
free_hit = 30

hype = 0.5

budget = 100
total_cycles = 38


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--bb', help="bench boost parameter, defaulte is {}".format(super_bench), type=int, metavar='super_bench', default=super_bench)
    parser.add_argument('-c','--tc', help="triple captain parameter, defaulte is {}".format(super_captain), type=int, metavar='super_captain', default=super_captain)
    parser.add_argument('-w','--wc', help="wild card parameter, defaulte is {}".format(wild_card), type=int, metavar='wild_card', default=wild_card)
    parser.add_argument('-f','--fh', help="free hit parameter, defaulte is {}".format(free_hit), type=int, metavar='free_hit', default=free_hit)
    parser.add_argument('-p','--hype', help="hype weight parameter, default is {}".format(hype), type=float, metavar='hype', default = hype)
    parser.add_argument('-m','--money', help="start budget parameter, default is {}".format(budget), type=float, metavar='budget', default = budget)
    parser.add_argument('-j','--threads', help="number of random teams to init, default is {}".format(random_teams), type=int, metavar='random_teams', default = random_teams)
    parser.add_argument('-s','--sub', help="one sub parameter, default is {}".format(sub), type=int, metavar='sub', default = sub)
    parser.add_argument('-d','--double_sub', help="two subs parameter, default is {}".format(ex_sub), type=int, metavar='ex_sub', default = ex_sub)
    parser.add_argument('-x','--squad_weight', help="prediction weights for squad <gw1,gw2,gw3> , default is {}".format(squad_weight), metavar='squad_weight', type=str, default = squad_weight)
    parser.add_argument('-y','--subs_weight', help="prediction weights for bench <gw1,gw2,gw3> , default is {}".format(subs_weight), metavar='subs_weight', type=str, default = subs_weight)
    return parser

imp = "from enum import Enum\n\n\n"

col = "class Columns(Enum):\n\
    id = 0\n\
    pos = 1\n\
    team = 2\n\
    val = 3\n\
    ps1 = 4\n\
    ps2 = 5\n\
    ps3 = 6\n\
    play = 7\n\n"

teams = "teams = ['Arsenal',\n\
         'Aston Villa',\n\
         'Brighton',\n\
         'Burnley',\n\
         'Chelsea',\n\
         'Crystal Palace',\n\
         'Everton',\n\
         'Fulham',\n\
         'Leeds United',\n\
         'Leicester',\n\
         'Liverpool',\n\
         'Manchester City',\n\
         'Manchester United',\n\
         'Newcastle United',\n\
         'Sheffield United',\n\
         'Southampton',\n\
         'Tottenham Hotspur',\n\
         'West Bromwich Albion',\n\
         'West Ham United',\n\
         'Wolverhampton']\n\n"

if __name__ == "__main__":
    parser = parse()
    args = parser.parse_args()
    f = open('params.py', 'w')
    f.write(imp)
    f.write(col)
    f.write(teams)
    f.write('squad_weight = {}\n'.format([float(i) for i in args.squad_weight.split(',')]))
    f.write('subs_weight = {}\n'.format([float(i) for i in args.subs_weight.split(',')]))
    f.write('fh_sq_weight = [1, 0, 0, 0, 0]\n')
    f.write('fh_b_weight = [0, 0, 0, 0, 0]\n')
    f.write('ex_sub = {}\n'.format(args.double_sub))
    f.write('sub = {}\n'.format(args.sub))
    f.write('random_teams = {}\n'.format(args.threads))
    f.write('super_captain = {}\n'.format(args.tc))
    f.write('super_bench = {}\n'.format(args.bb))
    f.write('wild_card = {}\n'.format(args.wc))
    f.write('free_hit = {}\n'.format(args.fh))
    f.write('hype = {}\n'.format(args.hype))
    f.write('budget = {}\n'.format(args.money))
    f.write('total_cycles = 38\n')
    f.close()