import networkx as nx
import pandas as pd
import random
import numpy as np

_instance_number = 20
_feature_number = 9
def getFixedDag():
    return nx.to_numpy_array(nx.DiGraph([(0,1),(0,2),(0,3),(1,4),(1,5),(4,6),(4,7),(3,7),(5,8)]))


    
def rand():
    return random.getrandbits(1)


def randomLinesWithAssertions(y):
    b = rand()
    c = rand()
    d = rand()
    e = rand() if b == 1 else 0
    f = rand() if b == 1 else 0
    g = rand() if e*d == 1 else 0
    h = rand() if e == 1 and d == 1 else 0
    i = rand() if f == 1 else 0
    return (1,b,c,d,e,f,g,h,i)


def getFixedData(instance_number = _instance_number):
    df = pd.DataFrame(columns=[i for i in range(0,_feature_number)])
    y = np.random.randint(0, 2, instance_number)
    for row in range(0,instance_number):
        df.loc[len(df)] = randomLinesWithAssertions(y)
    return df.to_numpy(), y
    df["y"] = np.random.randint(0, 2, df.shape[0])
    x = df.to_numpy()
    return x
