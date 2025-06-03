import networkx as nx
from collections import deque
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD


def is_a_list_tuple(obj):
    return obj and type(obj) in [list, tuple]


def is_valid_node_set(node_set):
    if is_a_list_tuple(node_set):
        for ele in node_set:
            if type(ele) != int:
                return False
        return True
    return False


def is_valid_edge_set(edge_set):
    if is_a_list_tuple(edge_set):
        for ele in edge_set:
            if not is_a_list_tuple(ele) or len(ele) != 2 or type(ele[0]) != int or type(ele[1]) != int :
                return False
        return True
    return False


def is_valid_dict(dict_set):
    if dict_set and type(dict_set) == dict:
        for key, value in dict_set.items():
            if (not key or type(key) != int) or (not value or type(value) != int):
                return False
        return True
    return False
            


def is_minimum_edge_covering(graph, edge_set, min_cover_size=None):
    if list(nx.isolates(graph)):
        return False
    
    if not is_valid_edge_set(edge_set):
        return False
    
    edge_set = set(tuple(sorted((u, v))) for u, v in edge_set)
    covered_vertices = set()
    for u, v in edge_set:
        covered_vertices.add(u)
        covered_vertices.add(v)
    
    if covered_vertices != set(graph.nodes()):
        return False

    if min_cover_size is None:
        min_cover = nx.min_edge_cover(graph)
        min_cover_size = len(min_cover)
    
    return len(edge_set) == min_cover_size


def is_valid_bfs_sequence(graph, edge_set, start_node=None):
    if not edge_set:
        return len(list(graph.neighbors(start_node))) == 0

    if not is_valid_edge_set(edge_set):
        return False

    graph = graph.copy()
    is_directed = isinstance(graph, nx.DiGraph)
    if start_node not in graph.nodes():
        return False

    discovered_nodes = {start_node}
    discovered_edges = set()
    queue = deque([start_node])
    edge_index = 0
    while queue and edge_index < len(edge_set):
        current_node = queue.popleft()
        
        unvisited_neighbors = []
        for neighbor in graph.neighbors(current_node):
            edge = (current_node, neighbor)
            reverse_edge = (neighbor, current_node)
            
            edge_discovered = edge in discovered_edges
            reverse_edge_discovered = reverse_edge in discovered_edges if not is_directed else False
            
            if neighbor not in discovered_nodes and not (edge_discovered or reverse_edge_discovered):
                unvisited_neighbors.append(neighbor)
        
        possible_next_edges = [(current_node, neighbor) for neighbor in unvisited_neighbors]
        while possible_next_edges and edge_index < len(edge_set):
            current_edge = edge_set[edge_index]
            edge_index += 1
            
            if not is_directed and current_edge[::-1] in graph.edges():
                if current_edge not in graph.edges():
                    current_edge = current_edge[::-1]
            
            if current_edge not in graph.edges():
                return False
                
            u, v = current_edge
            if u != current_node and v != current_node:
                return False
            
            if v == current_node:
                u, v = v, u 
                if is_directed and (v, u) not in graph.edges():
                    return False  
            
            if v in discovered_nodes:
                return False
                
            discovered_edges.add(tuple(current_edge))
            if not is_directed:
                discovered_edges.add((v, u)) 
            discovered_nodes.add(v)
            queue.append(v)
            
            if (u, v) in possible_next_edges:
                possible_next_edges.remove((u, v))
            elif not is_directed and (v, u) in possible_next_edges:
                possible_next_edges.remove((v, u))
            else:
                return False
    return edge_index == len(edge_set)


def is_valid_dfs_sequence(graph, edge_set, start_node=None):
    if not edge_set:
        return len(list(graph.neighbors(start_node))) == 0

    if not is_valid_edge_set(edge_set):
        return False

    graph = graph.copy()
    is_directed = isinstance(graph, nx.DiGraph)
    
    if start_node not in graph.nodes():
        return False
    
    discovered_nodes = {start_node}
    discovered_edges = set()
    stack = [start_node]
    edge_index = 0
    backtracks = []
    while stack and edge_index < len(edge_set):
        current_node = stack[-1]
        unvisited_neighbors = []
        for neighbor in graph.neighbors(current_node):
            edge = (current_node, neighbor)
            reverse_edge = (neighbor, current_node)
            
            edge_discovered = edge in discovered_edges
            reverse_edge_discovered = reverse_edge in discovered_edges if not is_directed else False
            
            if neighbor not in discovered_nodes and not (edge_discovered or reverse_edge_discovered):
                unvisited_neighbors.append(neighbor)
        
        if not unvisited_neighbors:
            stack.pop() 
            backtracks.append(current_node)
            continue
        
        current_edge = edge_set[edge_index]
        edge_index += 1
        if not is_directed and current_edge[::-1] in graph.edges():
            if current_edge not in graph.edges():
                current_edge = current_edge[::-1]
        
        if current_edge not in graph.edges():
            return False
            
        u, v = current_edge
        if u != current_node and v != current_node:
            return False
        
        if v == current_node:
            u, v = v, u  
            if is_directed and (v, u) not in graph.edges():
                return False 
        
        if v in discovered_nodes:
            return False
        
        discovered_edges.add(tuple(current_edge))
        if not is_directed:
            discovered_edges.add((v, u))
        discovered_nodes.add(v)
        stack.append(v)
    
    return edge_index == len(edge_set)


def is_minimum_vertex_covering(graph, node_set, min_cover_size=None):
    if not is_valid_node_set(node_set):
        return False

    node_set = set(node_set)
    for u, v in graph.edges():
        if u not in node_set and v not in node_set:
            return False
        
    if min_cover_size is None:
        min_cover_size = min_vertex_cover(graph)
        
    return len(node_set) == len(min_cover_size)


def is_maximal_independent_set(graph, node_set):
    if not is_valid_node_set(node_set):
        return False
    
    node_set = set(node_set)
    for u in node_set:
        for v in node_set:
            if u != v and graph.has_edge(u, v):
                return False
    
    remaining_nodes = set(graph.nodes()) - node_set
    for node in remaining_nodes:
        has_neighbor_in_set = False
        for u in node_set:
            if graph.has_edge(node, u):
                has_neighbor_in_set = True
                break
        if not has_neighbor_in_set:
            return False
    return True


def is_valid_dominating_set(graph, node_set):
    if not is_valid_node_set(node_set):
        return False
    return nx.is_dominating_set(graph, node_set)


def is_vertex_cover(graph, node_set):
    node_set = set(node_set)
    for u, v in graph.edges():
        if u not in node_set and v not in node_set:
            return False
    return True


def min_vertex_cover(graph):
    if len(graph.nodes()) <= 20:
        return min_vertex_cover_branch_and_bound(graph)
    else:
        return min_vertex_cover_ilp(graph)
    

def min_vertex_cover_branch_and_bound(graph):
    nodes = list(graph.nodes())
    best_cover = set(nodes)
    
    def branch_and_bound(graph, current_cover, remaining_nodes):
        nonlocal best_cover
        if len(current_cover) >= len(best_cover):
            return
        
        if not remaining_nodes:
            if is_vertex_cover(graph, current_cover):
                best_cover = current_cover.copy()
            return
        
        node = remaining_nodes[0]
        new_remaining = remaining_nodes[1:]
        branch_and_bound(graph, current_cover | {node}, new_remaining)
        
        if graph.is_directed():
            edges_to_cover = list(graph.out_edges(node)) + list(graph.in_edges(node))
        else:
            edges_to_cover = list(graph.edges(node))
        
        nodes_to_add = set()
        for u, v in edges_to_cover:
            if u != node and v != node:
                continue
            
            other_node = v if u == node else u
            if other_node not in current_cover:
                nodes_to_add.add(other_node)
    
        new_cover = current_cover | nodes_to_add
        remaining_filtered = [n for n in new_remaining if n not in nodes_to_add]
        branch_and_bound(graph, new_cover, remaining_filtered)
    
    branch_and_bound(graph, set(), nodes)
    return best_cover


def min_vertex_cover_ilp(graph):
    model = LpProblem(name="min_vertex_cover", sense=LpMinimize)
    x = {node: LpVariable(name=f"x_{node}", cat='Binary') for node in graph.nodes()}
    model += lpSum(x.values())
    for u, v in graph.edges():
        model += x[u] + x[v] >= 1
    model.solve(PULP_CBC_CMD(msg=False))
    
    min_cover = {node for node in graph.nodes() if x[node].value() > 0.5}
    return min_cover


def is_minimum_spanning_tree(graph, edge_set, minial_weight=None):
    if not is_valid_edge_set(edge_set):
        return False
    
    num_nodes = graph.number_of_nodes()
    tree = nx.Graph()
    tree.add_nodes_from(graph.nodes())  
    tree.add_edges_from(edge_set)
    
    if not nx.is_connected(tree):
        return False
    
    if len(edge_set) != num_nodes - 1:
        return False
    
    if not nx.is_tree(tree):
        return False
    return True


def is_valid_shortest_path(G, path, source, target, shortest_length=None):
    if not is_valid_node_set(path):
        return False
    
    if not path or path[0] != source or path[-1] != target:
        return False
    
    for i in range(len(path) - 1):
        if not G.has_edge(path[i], path[i + 1]):
            return False
    
    path_length = sum(G[path[i]][path[i + 1]].get('weight', 1) for i in range(len(path) - 1))
    if shortest_length is None:
        shortest_length = nx.shortest_path_length(G, source=source, target=target, weight='weight')
    return path_length == shortest_length


def is_maximal_weight_matching(graph, edge_set, maximal_weight=None):
    if not is_valid_edge_set(edge_set):
        return False
    
    vertices_used = set()
    matching_weight = 0
    matching = nx.Graph()
    for edge in edge_set:
        u, v = edge
        if not graph.has_edge(u, v):
            return False
        
        if u in vertices_used or v in vertices_used:
            return False
        
        vertices_used.add(u)
        vertices_used.add(v)
    
        matching.add_edge(u, v, weight=graph[u][v].get('weight', 1))
        matching_weight += graph[u][v].get('weight', 1)
    
    if maximal_weight is None:
        optimal_matching = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=False)
        maximal_weight = sum(graph[u][v].get('weight', 1) for u, v in optimal_matching)
    
    return matching_weight == maximal_weight


def is_valid_topological_sort(graph, node_set):
    if not is_valid_node_set(node_set):
        return False
    
    if set(node_set) != set(graph.nodes()):
        return False
    
    position = {node: idx for idx, node in enumerate(node_set)}
    for u, v in graph.edges():
        if position[u] >= position[v]:
            return False
        
    return True


def is_valid_hamiltonian_path(graph, node_set):
    if not is_valid_node_set(node_set):
        return False

    if set(node_set) != set(graph.nodes()):
        return False
    
    if len(node_set) != len(set(node_set)):
        counts = {}
        duplicates = []
        for node in node_set:
            counts[node] = counts.get(node, 0) + 1
            if counts[node] > 1:
                duplicates.append(node)
        return False
    
    for i in range(len(node_set) - 1):
        if not graph.has_edge(node_set[i], node_set[i+1]):
            return False
    return True


def is_bipartite_maximum_matching(graph, edge_set, max_matching=None):
    if not is_valid_edge_set(edge_set):
        return False

    if not nx.is_bipartite(graph):
        raise ValueError("The input graph must be bipartite")

    for edge in edge_set:
        u, v = edge
        if not graph.has_edge(u, v):
            return False
    
    vertices = set()
    for u, v in edge_set:
        if u in vertices or v in vertices:
            return False
        vertices.add(u)
        vertices.add(v)
    
    top_nodes, bottom_nodes = nx.bipartite.sets(graph)
    if max_matching is None:
        max_matching = nx.bipartite.maximum_matching(graph, top_nodes)
    max_matching_size = len(max_matching) // 2
    return len(edge_set) == max_matching_size


def is_valid_isomorphism(G1, G2, mapping):
    if not is_valid_dict(mapping):
        return False
    
    if set(G1.nodes()) != set(mapping.keys()):
        return False
    
    if not all(node in G2.nodes() for node in mapping.values()):
        return False
    
    if len(set(mapping.values())) != len(mapping):
        return False
    
    for u in G1.nodes():
        for v in G1.nodes():
            edge_in_g1 = G1.has_edge(u, v)
            edge_in_g2 = G2.has_edge(mapping[u], mapping[v])
            if edge_in_g1 != edge_in_g2:
                return False
    return True


def is_minimum_spanning_tree_weighted(graph, edge_list: list) -> bool:
    if not is_valid_edge_set(edge_list):
        return False

    T = nx.Graph()
    T.add_nodes_from(graph.nodes)
    T.add_edges_from(edge_list)

    if not nx.is_connected(T) or T.number_of_edges() != graph.number_of_nodes() - 1:
        return False

    mst = nx.minimum_spanning_tree(graph, weight='weight')
    mst_weight = mst.size(weight='weight')

    edge_weight = 0
    for u, v in T.edges:
        if graph.get_edge_data(u, v) is None:
            return False
        edge_weight += graph.get_edge_data(u, v).get('weight', 1)
    return abs(edge_weight - mst_weight) < 1e-8



if __name__ == "__main__":
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add edges (implicitly adds nodes as well)
    G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (3, 5)])
    
    # A valid topological sort
    valid_sequence = [1, 2, 3, 5, 4]
    
    # An invalid topological sort (4 comes before 2)
    invalid_sequence = [1, 4, 3, 2, 5]
    
    # Check and print results
    print(f"Is {valid_sequence} a valid topological sort? {is_valid_topological_sort(G, valid_sequence)}")
    print(f"Is {invalid_sequence} a valid topological sort? {is_valid_topological_sort(G, invalid_sequence)}")