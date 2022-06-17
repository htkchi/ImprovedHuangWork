import numpy as np
from mip import Model, xsum, maximize, BINARY
import pandas as pd

#Parameter:
df = pd.read_csv('set_cover.csv', header=None)

#Range:
numAlternative = len(df)
numRank = len(df)

#Model:
m_rank = Model("DCRA")

#Decision Variable:
x = [[m_rank.add_var(var_type=BINARY) for i in range(numAlternative)] for j in range(numRank)]

#Objective Function:
m_rank.objective = maximize(xsum(df[i][j] * x[i][j] for i in range(numAlternative) for j in range(numRank)))

#Constraints:
for i in range(numAlternative):
    m_rank += xsum(x[i][j] for j in range(numRank)) == 1

for j in range(numRank):
    m_rank += xsum(x[i][j] for i in range(numAlternative)) == 1

#Optimizing:
m_rank.optimize()

#Print Ouput:
print("DCRA Rank:")
for i in range(numAlternative):
    for j in range(numAlternative):
        if x[i][j].x >= 0.99:
            print("Alternative {} is rank at {}".format(i,j))