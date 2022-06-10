from mip import Model, xsum, maximize, BINARY
import pandas as pd

#Parameter:
df = pd.read_csv('set_cover.csv', header=None)

#Range:
numAlternative = 65

#Model:
m = Model("DCRA")

#Decision Variable:
x = [[m.add_var(var_type=BINARY) for i in range(numAlternative)] for j in range(numAlternative)]

#Objective Function:
m.objective = maximize(xsum(df[i][j] * x[i][j] for i in range(numAlternative) for j in range(numAlternative)))

#Constraints:
for i in range(numAlternative):
    m += xsum(x[i][j] for j in range(numAlternative)) == 1

for j in range(numAlternative):
    m += xsum(x[i][j] for i in range(numAlternative)) == 1

#Optimizing:
m.optimize()

#Print Ouput:
print("DCRA Rank:")
for i in range(numAlternative):
    for j in range(numAlternative):
        if x[i][j].x >= 0.99:
            print("Alternative {} is rank at {}".format(i,j))