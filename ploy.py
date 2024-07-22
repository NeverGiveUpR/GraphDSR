import networkx as nx
import matplotlib.pyplot as plt
import torch as th

# function to visulize a graph through adjacent matrix
def print_graph(node_names, adj, name):
    print("name:", name)
    print("node_names:", node_names)
    graph = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] == 1:
                graph.append((i+1, j+1))
            elif adj[i][j] == 2:
                graph.append((i+1, j+1))
                graph.append((i+1, j+1))
    node_labels = node_names
    print("graph:", graph)
    print()
    # G = nx.MultiGraph([(1, 2), (1, 2), (1, 2), (3, 1), (3, 2)])
    G = nx.MultiDiGraph(graph)
    pos = nx.random_layout(G)
    # node_labels = {1: 'Node 1', 2: 'Node 2', 3: 'Node 3'}  # Node labels

    plt.figure(figsize=(8, 6))  # Adjust the figure size
    nx.draw_networkx_nodes(G, pos, node_color='cornflowerblue', node_size=700, alpha=1)
    ax = plt.gca()

    for e in G.edges:
        # Compute the mid-point between two nodes for arrow positioning
        mid_point = ((pos[e[0]][0] + pos[e[1]][0]) / 2, (pos[e[0]][1] + pos[e[1]][1]) / 2)
        ax.annotate("",
                    xy=pos[e[0]], xycoords='data',
                    xytext=pos[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="-|>", color="gray",
                                    shrinkA=20, shrinkB=20,
                                    patchA=None, patchB=None,
                                    connectionstyle=f"arc3,rad={0.3*e[2]}",
                                    ),
                    )
    
    # Add node labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black', verticalalignment="center")

    plt.axis('off')
    plt.savefig('./graph/'+name+'.png', bbox_inches='tight')
    plt.close()

# node_names = {1: 'N1', 2: 'N2', 3: 'N3'}
# adjacency_matrix = th.tensor([
#     [0, 1, 2],
#     [0, 0, 1],
#     [0, 0, 0]
# ])
# print_graph(node_names, adjacency_matrix, 'example_hu')
