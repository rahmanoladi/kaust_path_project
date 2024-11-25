import dgl
import torch
from dgl import GCNNorm
transform = GCNNorm()
g = dgl.graph(([0, 1, 2], [0, 0, 1]))
print(g.adj())


import dgl
g = dgl.graph(([0,0,1,1,2,2,3,  1,2,3,4,3,4,4], [1,2,3,4,3,4,4,  0,0,1,1,2,2,3]))
#g.edata['eids'] = g.edges()

print(g.edge_ids([0,0,1,1,2,2,3,  1,2,3,4,3,4,4], [1,2,3,4,3,4,4,  0,0,1,1,2,2,3] ))
edges = g.edges()
print("EDGES: ", edges)
print(g.edge_ids(   edges[0], edges[1] ) )

g = transform(g)
print(g.adj())

print("weights: ", g.edata['w'])

#g = g.add
deg = g.in_degrees()

deg_root = torch.sqrt(1.0/deg)
print("deg_root: ", deg_root)
