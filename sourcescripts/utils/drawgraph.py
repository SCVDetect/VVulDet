 
import matplotlib.pyplot as plt
from dgl import load_graphs
import networkx as nx  




def draw_dgl_graph(dgl_graph):
    """Convert DGL graph to NetworkX graph"""
    nx_graph = dgl_graph.to_networkx()
    pos = nx.spring_layout(nx_graph) 
    nx.draw(nx_graph, pos, with_labels=True, 
            node_color='skyblue', node_size=400, 
            edge_color='black', linewidths=1, 
            font_size=8)
 
    plt.show()
#draw_dgl_graph(g)
