import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

def visualize_graph(data, color_map=None, max_nodes=100):
    G = to_networkx(data, to_undirected=True)
    
    if G.number_of_nodes() > max_nodes:
        nodes = list(G.nodes())[:max_nodes]
        G = G.subgraph(nodes)
        
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    
    if color_map is not None:
        colors = [color_map[node] for node in G.nodes()]
        nx.draw(G, pos, node_color=colors, node_size=50, cmap=plt.cm.Set1)
    else:
        nx.draw(G, pos, node_size=50)
        
    plt.savefig("graph_visualization.png")
    plt.close()
