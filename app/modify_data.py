


import pandas as pd




for i in range(37):
    df1 = pd.read_csv("../db/simulation/merged_predictions/gw{}.csv".format(i+1))
    df2 = pd.read_csv("../db/simulation/merged_predictions/gw{}.csv".format(i+2))
    df1 = df1[~df1['id'].isin(df2['id'])]
    df1['ps1'] = 0
    df1['ps2'] = 0
    df1['ps3'] = 0
    df1['play'] = 0
    df1['hipe'] = 0
    df2 = df2.append(df1)
    df2.to_csv("../db/simulation/merged_predictions/gw{}.csv".format(i+2), index=False)

for i in range(37):
    df1 = pd.read_csv("../db/simulation/realdata/gw{}.csv".format(i+1))
    df2 = pd.read_csv("../db/simulation/realdata/gw{}.csv".format(i+2))
    df1 = df1[~df1['id'].isin(df2['id'])]
    df1['total_points'] = 0

    df2 = df2.append(df1)
    df2.to_csv("../db/simulation/realdata/gw{}.csv".format(i+2), index=False)