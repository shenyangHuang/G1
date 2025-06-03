import networkx as nx
import ast


def construct_graph(nodes, edges, direction, task_name):
    assert direction in ['directed', 'undirected']
    if edges.startswith('('):
        node_tuples = ast.literal_eval(nodes)
        nodes_1, nodes_2 = node_tuples
        
        edge_tuples = ast.literal_eval(edges)
        edges_1, edges_2 = edge_tuples
        
        G1 = nx.Graph() if direction == 'undirected' else nx.DiGraph()
        G1.add_nodes_from(nodes_1)
        G1.add_edges_from(edges_1)
        
        G2 = nx.Graph() if direction == 'undirected' else nx.DiGraph()
        G2.add_nodes_from(nodes_2)
        G2.add_edges_from(edges_2)
        return G1, G2

    else:
        nodes = ast.literal_eval(nodes)
        nodes = list(range(nodes[0], nodes[1] + 1))
        edges = ast.literal_eval(edges)
        
        G = nx.Graph() if direction == 'undirected' else nx.DiGraph()
        G.add_nodes_from(nodes)
        if task_name in ['weighted_shortest_path', "weighted_minimum_spanning_tree"]:
            for u, v, w in edges:
                G.add_edge(u, v, weight=w)
        else:
            G.add_edges_from(edges)
        return G, None


if __name__ == '__main__':
    nodes = "(1, 24)"
    edges = "[(1, 14), (1, 15), (14, 13), (14, 21), (15, 7), (15, 13), (15, 21), (2, 17), (2, 18), (2, 8), (17, 18), (18, 3), (18, 8), (8, 16), (3, 4), (5, 22), (22, 19), (6, 20), (6, 10), (20, 23), (10, 9), (7, 21), (21, 13), (11, 12), (12, 24), (24, 19)]"
    G1, G2 = construct_graph(nodes, edges, 'undirected')
    print(G1)
    print(G2)
    
    nodes = "([0, 1, 2, 3, 4, 5], [101, 102, 103, 104, 100, 105])"
    edges = "([(0, 4), (0, 5), (0, 3), (0, 1), (1, 2), (1, 3), (1, 5), (3, 5), (4, 5)], [(101, 100), (101, 105), (101, 104), (101, 102), (102, 103), (102, 104), (102, 105), (104, 105), (100, 105)])"
    G1, G2 = construct_graph(nodes, edges, 'undirected')
    print(G1.nodes())
    print(G2.nodes())
    
    d = str({1: 101, 2: 102})
    d = ast.literal_eval(d)
    print(d)
    print(type(d))
    