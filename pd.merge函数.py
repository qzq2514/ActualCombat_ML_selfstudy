import pandas as pd
import numpy as np

left = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2','k3'],
                     'key2': ['K0', 'K1', 'K0', 'K1','k3'],
                     'A': ['A0', 'A1', 'A2', 'A3','A4'],
                     'B': ['B0', 'B1', 'B2', 'B3','A4']})

right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2','k4'],
                       'key2': ['K0', 'K0', 'K0', 'K0','k4'],
                       'C': ['C0', 'C1', 'C2', 'C3','C5'],
                       'D': ['D0', 'D1', 'D2', 'D3','C4']})

print(left)
print("==========")
print(right)
print("----------")

#默认how="inner"    #改变how="left","right","outer"观察变化
result = pd.merge(left, right, on=['key1'],how="left")   #
print(result)