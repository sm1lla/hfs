import sys
sys.path.append('/home/kathrin/hfs/algo/')


from go import open_dag


g = open_dag("./hfs/data/go_digraph")
r = [n for n,d in g.in_degree() if d==0] 
collapse = [n for n,d in g.in_degree() if d==1]
collapse2 =  [n for n,d in g.out_degree() if d==1]


print(f"{set(collapse) & set(collapse2)}")

