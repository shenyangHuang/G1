import math, itertools, re
import networkx as nx
from networkx.algorithms import approximation
from collections import defaultdict, deque
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD


def is_vertex_cover(graph, node_set):
    # Convert node_set to a set if it's a list
    node_set = set(node_set)
    
    # Check if every edge has at least one endpoint in node_set
    for u, v in graph.edges():
        if u not in node_set and v not in node_set:
            return False
    return True

def min_vertex_cover(graph):
    if len(graph.nodes()) <= 20:
        # For small graphs, use branch and bound
        return min_vertex_cover_branch_and_bound(graph)
    else:
        # For larger graphs, use ILP
        return min_vertex_cover_ilp(graph)

def min_vertex_cover_branch_and_bound(graph):
    nodes = list(graph.nodes())
    best_cover = set(nodes)  # Initialize with all nodes (worst case)
    
    def branch_and_bound(graph, current_cover, remaining_nodes):
        nonlocal best_cover
        
        # If current cover is already worse than best, prune
        if len(current_cover) >= len(best_cover):
            return
        
        # If no nodes left to consider
        if not remaining_nodes:
            # Check if it's a valid cover
            if is_vertex_cover(graph, current_cover):
                best_cover = current_cover.copy()
            return
        
        # Take next node to branch on
        node = remaining_nodes[0]
        new_remaining = remaining_nodes[1:]
        
        # Branch 1: Include the node in the cover
        branch_and_bound(graph, current_cover | {node}, new_remaining)
        
        # Branch 2: Exclude the node from the cover
        # For this branch, we need to ensure all edges where this node is an endpoint are covered
        
        # graphet all edges where node is an endpoint
        if graph.is_directed():
            # For directed graphs
            edges_to_cover = list(graph.out_edges(node)) + list(graph.in_edges(node))
        else:
            # For undirected graphs
            edges_to_cover = list(graph.edges(node))
        
        # Find nodes that must be added to cover these edges
        nodes_to_add = set()
        for u, v in edges_to_cover:
            if u != node and v != node:
                continue  # This edge doesn't involve our node
            
            other_node = v if u == node else u
            if other_node not in current_cover:
                nodes_to_add.add(other_node)
        
        # If we can exclude node, continue with branch 2
        new_cover = current_cover | nodes_to_add
        remaining_filtered = [n for n in new_remaining if n not in nodes_to_add]
        branch_and_bound(graph, new_cover, remaining_filtered)
    
    # Start the branch and bound process
    branch_and_bound(graph, set(), nodes)
    return best_cover

def min_vertex_cover_ilp(graph):
    # Create the ILP problem
    model = LpProblem(name="min_vertex_cover", sense=LpMinimize)
    
    # Create binary variables for each node
    # x[i] = 1 if node i is in the vertex cover, 0 otherwise
    x = {node: LpVariable(name=f"x_{node}", cat='Binary') for node in graph.nodes()}
    
    # Objective: minimize the number of nodes in the cover
    model += lpSum(x.values())
    
    # Constraints: for each edge (u,v), at least one of u or v must be in the cover
    for u, v in graph.edges():
        model += x[u] + x[v] >= 1
    
    # Solve the model
    model.solve(PULP_CBC_CMD(msg=False))
    
    # Extract the solution
    min_cover = {node for node in graph.nodes() if x[node].value() > 0.5}
    return min_cover


def solve_tsp_exact(G):
    n = G.number_of_nodes()

    # Generate all possible permutations of nodes
    nodes = list(G.nodes())
    all_permutations = permutations(nodes)
    
    min_cost = float('inf')
    best_path = None
    
    for path in all_permutations:
        cost = 0
        # Calculate path cost
        for i in range(n - 1):
            cost += G[path[i]][path[i+1]]['weight']
        # Add cost of returning to the starting node
        cost += G[path[-1]][path[0]]['weight']
        
        if cost < min_cost:
            min_cost = cost
            best_path = list(path) + [path[0]]  # Complete the cycle
    
    return best_path, min_cost


def find_isomorphism_mapping(G1, G2):
    if not nx.is_isomorphic(G1, G2):
        return None
    
    nodes_G1 = list(G1.nodes())
    nodes_G2 = list(G2.nodes())
    if len(nodes_G1) != len(nodes_G2):
        return None
    
    GM = nx.isomorphism.GraphMatcher(G1, G2)
    if GM.is_isomorphic():
        return GM.mapping
    return None

def is_a_list_tuple(obj):
    return obj and type(obj) in [list, tuple]

def is_valid_node_set(node_set):
    if is_a_list_tuple(node_set):
        for ele in node_set:
            if ele is None or type(ele) != int:
                return False
        return True
    else:
        return False

def is_valid_edge_set(edge_set):
    if is_a_list_tuple(edge_set):
        for ele in edge_set:
            if not is_a_list_tuple(ele) or len(ele) != 2 or type(ele[0]) != int or type(ele[1]) != int :
                return False
        return True
    else:
        return False



def is_valid_dominating_set(graph, node_set):
    if not is_valid_node_set(node_set):
        return False
    return nx.is_dominating_set(graph, node_set)
    
def is_maximal_independent_set(graph, node_set):
    if not is_valid_node_set(node_set):
        return False
    
    node_set = set(node_set)

    # For each pair of nodes in the set, they should not be adjacent
    for u in node_set:
        for v in node_set:
            if u != v and graph.has_edge(u, v):
                return False
    
    # For each node outside the set, it must be adjacent to at least one node in the set
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

def is_minimum_vertex_cover(graph, node_set, min_cover_size=None):
    if not is_valid_node_set(node_set):
        return False

    node_set = set(node_set)
    
    # Step 1: Check if node_set is a valid vertex cover
    for u, v in graph.edges():
        if u not in node_set and v not in node_set:
            return False
    
    # Step 2: Check if the size of node_set equals the minimum cover size
    if min_cover_size is None:
        min_cover_size = min_vertex_cover(graph)
        
    return len(node_set) == len(min_cover_size)

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

def is_valid_topological_sort(G, node_sequence):
    if not is_valid_node_set(node_sequence):
        return False

    # Check if all nodes in the sequence are in the graph
    if set(node_sequence) != set(G.nodes()):
        return False
    
    # Create a dictionary to store the position of each node in the sequence
    position = {node: idx for idx, node in enumerate(node_sequence)}
    
    # Check if for every edge (u, v), u comes before v in the sequence
    for u, v in G.edges():
        if position[u] >= position[v]:
            return False
    
    return True

def is_valid_hamiltonian_path(G, path):
    if not is_valid_node_set(path):
        return False

    # Check if the path contains all nodes in the graph
    if set(path) != set(G.nodes()):
        return False
    
    # Check if the path contains any duplicate nodes
    if len(path) != len(set(path)):
        counts = {}
        duplicates = []
        
        for node in path:
            counts[node] = counts.get(node, 0) + 1
            if counts[node] > 1:
                duplicates.append(node)
                
        return False
    
    # Check if each consecutive pair of nodes forms an edge in the graph
    for i in range(len(path) - 1):
        if not G.has_edge(path[i], path[i+1]):
            return False
    
    return True

def is_valid_bfs_sequence(graph, edge_set, start_node=None):

    if not edge_set:
        return len(list(graph.neighbors(start_node))) == 0

    if not is_valid_edge_set(edge_set):
        return False

    graph = graph.copy()
    is_directed = isinstance(graph, nx.DiGraph)
    if start_node not in graph.nodes():
        return False

    # Keep track of discovered nodes and edges
    discovered_nodes = {start_node}
    discovered_edges = set()
    
    queue = deque([start_node])
    edge_index = 0
    
    while queue and edge_index < len(edge_set):
        current_node = queue.popleft()
        
        # graphet all unvisited neighbors of the current node
        unvisited_neighbors = []
        for neighbor in graph.neighbors(current_node):
            edge = (current_node, neighbor)
            reverse_edge = (neighbor, current_node)
            
            # Check if this edge has been discovered
            edge_discovered = edge in discovered_edges
            reverse_edge_discovered = reverse_edge in discovered_edges if not is_directed else False
                
            # Check if this neighbor has been discovered
            if neighbor not in discovered_nodes and not (edge_discovered or reverse_edge_discovered):
                unvisited_neighbors.append(neighbor)
        
        # Gather possible next edges from the current node
        possible_next_edges = [(current_node, neighbor) for neighbor in unvisited_neighbors]
        
        # Process the next layer of edges
        while possible_next_edges and edge_index < len(edge_set):
            current_edge = edge_set[edge_index]
            edge_index += 1
            
            # For undirected graphs, normalize the edge orientation
            if not is_directed and current_edge[::-1] in graph.edges():
                if current_edge not in graph.edges():
                    current_edge = current_edge[::-1]
            
            # The edge must exist in the graph
            if current_edge not in graph.edges():
                return False
                
            u, v = current_edge
            
            # One of u or v must be the current node from the BFS queue
            if u != current_node and v != current_node:
                return False
                
            # Ensure the source node is the one we're currently processing
            if v == current_node:
                u, v = v, u  # swap to make u the current node
                if is_directed and (v, u) not in graph.edges():
                    return False  # Can't traverse against edge direction in directed graph
            
            # The destination node must be unvisited
            if v in discovered_nodes:
                return False
                
            # Mark the edge and destination node as discovered
            discovered_edges.add(current_edge)
            if not is_directed:
                discovered_edges.add((v, u))  # Add reverse edge for undirected graphs
            discovered_nodes.add(v)
            
            # Add the newly discovered node to the queue
            queue.append(v)
            
            # Remove the edge from possible next edges
            if (u, v) in possible_next_edges:
                possible_next_edges.remove((u, v))
            elif not is_directed and (v, u) in possible_next_edges:
                possible_next_edges.remove((v, u))
            else:
                return False  # We couldn't find the current edge in our possible edges
    
    # If we processed all edges in the sequence, it's valid
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
    
    # Keep track of discovered nodes and edges
    discovered_nodes = {start_node}
    discovered_edges = set()
    
    stack = [start_node]
    edge_index = 0
    backtracks = []
    
    while stack and edge_index < len(edge_set):
        current_node = stack[-1]  # Peek at the top of the stack
        
        # graphet all unvisited neighbors of the current node
        unvisited_neighbors = []
        for neighbor in graph.neighbors(current_node):
            edge = (current_node, neighbor)
            reverse_edge = (neighbor, current_node)
            
            # Check if this edge has been discovered
            edge_discovered = edge in discovered_edges
            reverse_edge_discovered = reverse_edge in discovered_edges if not is_directed else False
                
            # Check if this neighbor has been discovered
            if neighbor not in discovered_nodes and not (edge_discovered or reverse_edge_discovered):
                unvisited_neighbors.append(neighbor)
        
        # If no unvisited neighbors, we need to backtrack
        if not unvisited_neighbors:
            stack.pop()  # Backtrack
            backtracks.append(current_node)
            continue
        
        # graphet current edge from sequence
        current_edge = edge_set[edge_index]
        edge_index += 1
        
        # For undirected graphs, normalize the edge orientation
        if not is_directed and current_edge[::-1] in graph.edges():
            if current_edge not in graph.edges():
                current_edge = current_edge[::-1]
        
        # The edge must exist in the graph
        if current_edge not in graph.edges():
            return False
            
        u, v = current_edge
        
        # One of u or v must be the current node from the DFS stack
        if u != current_node and v != current_node:
            return False
            
        # Ensure the source node is the one we're currently processing
        if v == current_node:
            u, v = v, u  # swap to make u the current node
            if is_directed and (v, u) not in graph.edges():
                return False  # Can't traverse against edge direction in directed graph
        
        # The destination node must be unvisited
        if v in discovered_nodes:
            return False
            
        # Mark the edge and destination node as discovered
        discovered_edges.add(current_edge)
        if not is_directed:
            discovered_edges.add((v, u))  # Add reverse edge for undirected graphs
        discovered_nodes.add(v)
        
        # Push the newly discovered node to the stack
        stack.append(v)
    
    # If we processed all edges in the sequence, it's valid
    return edge_index == len(edge_set)

def is_minimum_edge_covering(graph, edge_set, min_cover_size=None):
    if list(nx.isolates(graph)):
        # If there are isolated nodes, no edge cover exists
        return False
    
    if not is_valid_edge_set(edge_set):
        return False
    
    # Convert edge_set to a set if it's a list
    edge_set = set(tuple(sorted((u, v))) for u, v in edge_set)
    
    # Step 1: Check if edge_set is an edge cover
    # Every vertex must be incident to at least one edge in edge_set
    covered_vertices = set()
    for u, v in edge_set:
        covered_vertices.add(u)
        covered_vertices.add(v)
    
    if covered_vertices != set(graph.nodes()):
        return False
    
    # Step 2: Check if edge_set is minimum by comparing with the provided size
    # or calculating it using NetworkX if not provided
    if min_cover_size is None:
        min_cover = nx.min_edge_cover(graph)
        min_cover_size = len(min_cover)
    
    return len(edge_set) == min_cover_size

def is_minimum_spanning_tree(graph, edge_list, minial_weight=None):
    if not is_valid_edge_set(edge_list):
        return False
    
    num_nodes = graph.number_of_nodes()
    
    tree = nx.Graph()
    tree.add_nodes_from(graph.nodes())  
    tree.add_edges_from(edge_list)
    
    # Check if the tree is connected
    if not nx.is_connected(tree):
        print('not connected')
        return False
    
    # Check if the number of edges is exactly (num_nodes - 1)
    if len(edge_list) != num_nodes - 1:
        print('not nodes-1')
        return False
    
    # Check if there are no cycles (a tree has no cycles)
    if not nx.is_tree(tree):
        print('not a tree')
        return False
    
    # For unweighted graphs, any spanning tree with (n-1) edges is a minimum spanning tree
    # For weighted graphs, we would need to compare with the actual MST
    return True

def is_minimum_spanning_tree_weighted(graph, edge_list: list,) -> bool:
    if not is_valid_edge_set(edge_list):
        return False

    T = nx.Graph()
    T.add_nodes_from(graph.nodes)
    T.add_edges_from(edge_list)

    if not nx.is_connected(T) or T.number_of_edges() != graph.number_of_nodes() - 1:
        return False

    mst = nx.minimum_spanning_tree(graph, weight='weight')
    mst_weight = mst.size(weight='weight')
    edge_weight = sum(
        graph.get_edge_data(u, v).get('weight', 1) for u, v in T.edges
    )
    return abs(edge_weight - mst_weight) < 1e-3

def is_maximal_weight_matching(G, edge_sequence, maximal_weight=None):
    if not is_valid_edge_set(edge_sequence):
        return False
    
    # Check if the edge sequence forms a valid matching
    vertices_used = set()
    matching_weight = 0
    
    matching = nx.Graph()
    for edge in edge_sequence:
        u, v = edge
        
        # Check if the edge exists in the original graph
        if not G.has_edge(u, v):
            return False
        
        # Check if either vertex is already used in the matching
        if u in vertices_used or v in vertices_used:
            return False
        
        vertices_used.add(u)
        vertices_used.add(v)
    
        matching.add_edge(u, v, weight=G[u][v].get('weight', 1))
        matching_weight += G[u][v].get('weight', 1)
    
    # Check if the matching is maximal (has maximum possible weight)
    if maximal_weight is None:
        optimal_matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False)
        maximal_weight = sum(G[u][v].get('weight', 1) for u, v in optimal_matching)
    
    return matching_weight == maximal_weight

def is_bipartite_maximum_matching(G, edge_list, max_matching=None):
    # Check if G is bipartite
    if not nx.is_bipartite(G):
        raise ValueError("The input graph must be bipartite")
    
    if not is_valid_edge_set(edge_list):
        return False

    # Convert edge_list to set for faster lookup
    edge_set = set(tuple(sorted(edge)) for edge in edge_list)
    
    # Check if edges in edge_list exist in G
    for edge in edge_list:
        u, v = edge
        if not G.has_edge(u, v):
            return False
    
    # Check if edge_list forms a valid matching (no common vertices)
    vertices = set()
    for u, v in edge_list:
        if u in vertices or v in vertices:
            return False
        vertices.add(u)
        vertices.add(v)
    
    # Get the maximum matching size using NetworkX's algorithm
    top_nodes, bottom_nodes = nx.bipartite.sets(G)
    if max_matching is None:
        max_matching = nx.bipartite.maximum_matching(G, top_nodes)
    max_matching_size = len(max_matching) // 2  # Divide by 2 because each edge is counted twice
    
    # If our matching has the same size as the maximum matching, it is maximum
    return len(edge_list) == max_matching_size


def is_valid_isomorphism(G1, G2, mapping):
    if set(G1.nodes()) != set(mapping.keys()):
        return False
    
    if not all(node in G2.nodes() for node in mapping.values()):
        return False
    
    if len(set(mapping.values())) != len(mapping):
        return False
    
    # Check that adjacency relationships are preserved
    for u in G1.nodes():
        for v in G1.nodes():
            edge_in_g1 = G1.has_edge(u, v)
            edge_in_g2 = G2.has_edge(mapping[u], mapping[v])
            if edge_in_g1 != edge_in_g2:
                return False
    return True





def extract_graph(text, info):
    import ast
    is_directed = (info['direction'].lower() == "directed")
    nodes_range = ast.literal_eval(info['nodes'])
    nodes = list(range(nodes_range[0], nodes_range[1]))
    edges = ast.literal_eval(info['edges'])

    if is_directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()
    graph.add_nodes_from(nodes)

    if len(edges[0]) == 2:
        graph.add_edges_from(edges)
    else:
        graph.add_weighted_edges_from(edges)

    return graph

def extract_two_graphs(text, info):
    import ast
    is_directed = (info['direction'].lower() == "directed")
    nodes_range_1, nodes_range_2 = ast.literal_eval(info['nodes'])
    nodes_1 = list(range(nodes_range_1[0], nodes_range_1[1]))
    nodes_2 = list(range(nodes_range_2[0], nodes_range_2[1]))
    edges_1, edges_2 = ast.literal_eval(info['edges'])

    if is_directed:
        graph_1 = nx.DiGraph()
        graph_2 = nx.DiGraph()
    else:
        graph_1 = nx.Graph()
        graph_2 = nx.Graph()
    
    graph_1.add_nodes_from(nodes_1)
    graph_1.add_edges_from(edges_1)

    graph_2.add_nodes_from(nodes_2)
    graph_2.add_edges_from(edges_2)

    return graph_1, graph_2

def extract_graph_mapping(answer):
    try:
        answer = ast.literal_eval(answer)
        if type(answer) != dict:
            return None
            
        for key, value in answer.items():
            if type(key) != int or type(value) != int:
                return None
        return answer
        
    except:
        return None

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string



def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    string = string.replace(":", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    string = string.replace('\\text', '')

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    # Remove slash 
    string = string.replace('\\', '')

    return string

def extract_solution(solution_str):
    from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
    possible_ans = last_boxed_only_string(solution_str)

    if possible_ans:
        return strip_string(remove_boxed(possible_ans))
    else:
        return None

    
def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 1.0  # 如果两个集合都是空的，通常定义相似度为1
    return len(intersection) / len(union)


def set_score(golden_set, generated_set):
    if len(golden_set):
        if (type(golden_set[0]) == int and is_valid_node_set(generated_set)) or (type(golden_set[0]) == tuple and is_valid_edge_set(generated_set)):
            return jaccard_similarity(golden_set, generated_set)
        else:
            return 0
    else:
        return (len(generated_set) == 0)



def extract_letters(input_string):
    return ''.join(re.findall(r'[a-zA-Z]', input_string))

def extract_floats(input_string):
    s = input_string.strip()

    try:
        value = float(s)
        return value
    except ValueError:
        pass

    frac_match = re.match(r'frac\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}', s)
    if frac_match and len(frac_match.groups()) >= 2:
        numerator = float(frac_match.group(1))
        denominator = float(frac_match.group(2))
        if denominator != 0:
            return numerator / denominator
        else:
            return None  # 分母为零非法

    div_match = re.match(r'(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$', s)
    if div_match and len(div_match.groups()) >= 2:
        numerator = float(div_match.group(1))
        denominator = float(div_match.group(2))
        if denominator != 0:
            return numerator / denominator
        else:
            return None  # 分母为零非法

    return None

def convert_int_to_set(string):
    if string and string.isdigit():
        return f'[{string}]'
    return string




import contextlib
import math
import re
import signal
from math import isclose
from typing import Union

from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


def is_digit(s):
    try:
        if "{,}" in str(s):
            num = float(str(s).replace("{,}", ""))
            return True, num

        num = float(str(s).replace(",", ""))
        return True, num
    except ValueError:
        return False, None


def normalize(answer, pi) -> str:
    # checking if answer is $<number> and removing $ in that case to compare
    if isinstance(answer, str) and bool(re.match(r"\$\d+(\.\d+)?", answer)):
        return answer[1:]

    # checking if answer is <number>% or <number>\\% and removing %
    if isinstance(answer, str) and (
        bool(re.match(r"^\d+(\.\d+)?%$", answer)) or bool(re.match(r"^\d+(\.\d+)?\\%$", answer))
    ):
        return answer.replace("\\%", "").replace("%", "")

    # handle base
    answer = handle_base(answer)

    # handle pi
    answer = handle_pi(answer, pi)

    return answer


def handle_base(x) -> str:
    if isinstance(x, str) and "_" in x:
        # Due to base
        x = x.split("_")[0]
        x = float(x)
        return int(x)
    return x


def handle_pi(string, pi):
    if isinstance(string, str) and "\pi" in string:
        # Find the first occurrence of "\pi"
        idx = string.find("\pi")

        # Iterate over the string and find all occurrences of "\pi" with a valid previous character
        while idx != -1:
            if idx > 0 and string[idx - 1].isdigit():
                # Replace "\pi" with "*math.pi" if the previous character is a digit
                string = string[:idx] + f"*{pi}" + string[idx + 3 :]
            else:
                # Replace "\pi" with "1*math.pi" if the previous character is not a digit
                string = string[:idx] + f"1*{pi}" + string[idx + 3 :]

            # Find the next occurrence of "\pi"
            idx = string.find("\pi", idx + 1)

        # Evaluate the expression using eval() function
        with contextlib.suppress(Exception):
            string = eval(string)

    return string


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    timeout: float = 10.0,
    pi: float = math.pi,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """

    prediction = normalize(prediction, pi)
    reference = normalize(reference, pi)

    if isinstance(prediction, str) and len(prediction) > 1000:  # handling weird corner-cases
        prediction = prediction[:1000]

    # 0. string comparison
    if isinstance(prediction, str) and isinstance(reference, str):
        if prediction.strip().lower() == reference.strip().lower():
            return True
        if prediction.replace(" ", "") == reference.replace(" ", ""):
            return True

    try:  # 1. numerical equal
        if is_digit(prediction)[0] and is_digit(reference)[0]:
            prediction = is_digit(prediction)[1]
            reference = is_digit(reference)[1]
            # number questions
            gt_result = [reference / 100, reference, reference * 100] if include_percentage else [reference]
            for item in gt_result:
                try:
                    if isclose(item, prediction, rel_tol=tolerance):
                        return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## deal with [], (), {}
    prediction = format_intervals(prediction)

    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        prediction
        and reference
        and prediction[0] in "(["
        and prediction[-1] in ")]"
        and prediction[0] == reference[0]
        and prediction[-1] == reference[-1]
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            [
                math_equal(pred_pt, ref_pt, include_percentage, tolerance)
                for pred_pt, ref_pt in zip(pred_parts, ref_parts)
            ]
        ):
            return True

    if "," in prediction and "," in reference:
        pred_parts = [item.strip() for item in prediction.split(",")]
        ref_parts = [item.strip() for item in reference.split(",")]

        if len(pred_parts) == len(ref_parts):
            return bool(
                all(
                    [
                        math_equal(pred_parts[i], ref_parts[i], include_percentage, tolerance)
                        for i in range(len(pred_parts))
                    ]
                )
            )

    # if we have point == tuple of values
    if prediction.startswith("Point") and reference[0] == "(" and reference[-1] == ")":
        pred_parts = prediction[prediction.find("(") + 1 : -1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            [
                math_equal(pred_pt, ref_pt, include_percentage, tolerance)
                for pred_pt, ref_pt in zip(pred_parts, ref_parts)
            ]
        ):
            return True

    # if reference is a matrix
    if "\begin{pmatrix}" in reference and prediction.startswith("Matrix"):
        try:
            pred_matrix = parse_expr(prediction)
            ref_matrix_items = reference.split()[1:-1:2]
            if len(pred_matrix) == len(ref_matrix_items) and all(
                [
                    math_equal(pred, ref, include_percentage, tolerance)
                    for ref, pred in zip(ref_matrix_items, pred_matrix)
                ]
            ):
                return True
        except Exception:
            pass
    elif "\begin{pmatrix}" in reference and prediction.startswith("[") and prediction.endswith("]"):
        if isinstance(eval(prediction), list):
            try:
                pred_matrix = eval(prediction)
                # ref_matrix_items = reference.split()[1:-1:2]
                ref_matrix_items = (
                    reference.lstrip("\\begin{pmatrix}")
                    .lstrip("\begin{pmatrix}")
                    .rstrip("\\end{pmatrix}")
                    .rstrip("\end{pmatrix}")
                )
                ref_matrix_items = ref_matrix_items.split("\\")
                ref_matrix_items = [row.split("&") if "&" in row else row for row in ref_matrix_items]
                if len(pred_matrix) == len(ref_matrix_items) and all(
                    [
                        math_equal(pred, ref, include_percentage, tolerance)
                        for ref, pred in zip(ref_matrix_items, pred_matrix)
                    ]
                ):
                    return True
            except Exception:
                pass

    return symbolic_equal(prediction, reference, tolerance, timeout)


def symbolic_equal(a, b, tolerance, timeout=10.0):
    def _parse(s):
        for f in [parse_expr, parse_latex]:
            try:
                with time_limit(timeout):
                    return f(s)
            except Exception:
                pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        with time_limit(timeout):
            if simplify(a - b) == 0:
                return True
    except Exception:
        pass

    try:
        with time_limit(timeout):
            if isclose(N(a), N(b), rel_tol=tolerance):
                return True
    except Exception:
        pass
    return False


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def format_intervals(prediction):
    patterns = {
        "Interval(": r"^Interval\((.*)\)$",
        "Interval.Ropen(": r"^Interval\.Ropen\((.*)\)$",
        "Interval.Lopen(": r"^Interval\.Lopen\((.*)\)$",
        "Interval.open(": r"^Interval\.open\((.*)\)$",
    }

    for key, pattern in patterns.items():
        match = re.match(pattern, prediction)
        if match:
            inner_content = match.group(1)

            if key == "Interval(":  # Intarval(a, b) == [a, b]
                return f"[{inner_content}]"
            elif key == "Interval.Ropen(":  # Intarval.Ropen(a, b) == [a, b)
                return f"[{inner_content})"
            elif key == "Interval.Lopen(":  # Intarval.Lopen(a, b) == (a, b]
                return f"({inner_content}]"
            elif key == "Interval.open(":  # Intarval.open(a, b) == (a, b)
                return f"({inner_content})"

    return prediction


if __name__ == '__main__':
    query = """
    The task is to determine the betweenness centrality of a node in the graph.

    Betweenness centrality of a node u is the sum of the fraction of all-pairs shortest paths that pass through u.
    Here is an undirected graph. In the graph, (u, v) means that node u and node v are connected. The graph contains nodes: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, and 30. The graph contains edges: (1, 25), (1, 2), (18, 25), (3, 4), (3, 5), (4, 5), (4, 7), (4, 8), (5, 7), (5, 8), (6, 7), (7, 8), (6, 8), (9, 10), (9, 11), (9, 13), (10, 11), (10, 12), (13, 14), (13, 15), (13, 16), (14, 26), (14, 27), (14, 15), (15, 24), (15, 26), (15, 16), (16, 24), (26, 27), (17, 23), (17, 18), (18, 23), (19, 20), (19, 21), (20, 21), (21, 22), (22, 30), (28, 30), (29, 30), (28, 29).

    Question: What is the betweenness centrality of node 28 in the graph?

    You need to format your answer as a float number.
    """

    graph = extract_graph(query)

    print(graph)