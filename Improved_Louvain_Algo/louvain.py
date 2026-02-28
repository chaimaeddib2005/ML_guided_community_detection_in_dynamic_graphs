"""
Improved Fast Louvain Algorithm for Community Detection
Based on: Zhang et al. (2021) "An Improved Louvain Algorithm for Community Detection"
DOI: 10.1155/2021/1485592

Key improvements over standard Louvain:
1. Dynamic iteration - only iterates over nodes that changed in previous iteration
2. Tree structure splitting - identifies and separates local tree structures using leaf pruning
3. Early stopping based on modularity gain threshold
"""

import networkx as nx
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple


class ImprovedFastLouvain:
    def __init__(self, graph: nx.Graph, resolution: float = 1.0, 
                 min_modularity_gain: float = 1e-7,
                 min_community_size: int = 2):
        """
        Initialize the Improved Fast Louvain algorithm.
        
        Args:
            graph: NetworkX graph
            resolution: Resolution parameter for modularity (default 1.0)
                      - Lower values (0.5-0.9) favor fewer, larger communities
                      - Higher values (1.1-2.0) favor more, smaller communities
            min_modularity_gain: Minimum modularity gain to continue iteration
            min_community_size: Minimum size for a community (merge smaller ones)
        """
        self.original_graph = graph.copy()
        self.graph = graph.copy()
        self.resolution = resolution
        self.min_modularity_gain = min_modularity_gain
        self.min_community_size = min_community_size
        self.m = graph.size(weight='weight')
        if self.m == 0:
            self.m = 1
            
        self.node_degrees = {node: self.graph.degree(node, weight='weight') 
                            for node in self.graph.nodes()}
        
    def detect_tree_structures(self, graph: nx.Graph) -> Tuple[List[Set[int]], nx.Graph]:
        """
        Identify local tree structures using iterative leaf node removal.
        
        The paper describes three types of tree structures:
        1. Single leaf nodes connected to a non-tree structure
        2. Chain structures (paths)
        3. Tree branches
        
        Algorithm: Iteratively remove leaf nodes (degree 1) until no more leaves exist.
        """
        tree_structures = []
        working_graph = graph.copy()
        removed_nodes = set()
        
        iteration = 0
        max_iterations = 100
        
        while iteration < max_iterations:
            iteration += 1
            
            leaf_nodes = [n for n in working_graph.nodes() 
                         if working_graph.degree(n) == 1]
            
            if not leaf_nodes:
                break
            
            leaf_groups = []
            processed_leaves = set()
            
            for leaf in leaf_nodes:
                if leaf in processed_leaves:
                    continue
                
                neighbors = list(working_graph.neighbors(leaf))
                if not neighbors:
                    continue
                
                neighbor = neighbors[0]
                
                if working_graph.degree(neighbor) == 2:
                    chain = {leaf, neighbor}
                    current = neighbor
                    
                    while working_graph.degree(current) == 2:
                        next_nodes = [n for n in working_graph.neighbors(current) 
                                     if n not in chain]
                        if not next_nodes:
                            break
                        current = next_nodes[0]
                        if current in leaf_nodes or working_graph.degree(current) == 1:
                            chain.add(current)
                            break
                        chain.add(current)
                    
                    leaf_groups.append(chain)
                    processed_leaves.update(chain)
                else:
                    leaf_groups.append({leaf})
                    processed_leaves.add(leaf)
            
            for group in leaf_groups:
                if len(group) >= 2:
                    tree_structures.append(group)
                removed_nodes.update(group)
            
            working_graph.remove_nodes_from(leaf_nodes)
            
            if working_graph.number_of_nodes() == 0:
                break
        
        reduced_graph = graph.copy()
        all_tree_nodes = set()
        for tree in tree_structures:
            all_tree_nodes.update(tree)
        reduced_graph.remove_nodes_from(all_tree_nodes)
        
        return tree_structures, reduced_graph
    
    def assign_tree_communities(self, trees: List[Set[int]], 
                               communities: Dict[int, int]) -> Dict[int, int]:
        """
        Assign tree structure nodes to communities based on their connections.
        
        For each tree structure, assign all nodes to the community of their 
        connecting node in the main graph.
        """
        tree_assignments = {}
        
        for tree in trees:
            connecting_community = None
            
            for tree_node in tree:
                for neighbor in self.original_graph.neighbors(tree_node):
                    if neighbor not in tree and neighbor in communities:
                        connecting_community = communities[neighbor]
                        break
                if connecting_community is not None:
                    break
            
            if connecting_community is None:
                connecting_community = min(tree)
            
            for node in tree:
                tree_assignments[node] = connecting_community
        
        return tree_assignments
    
    def compute_modularity(self, communities: Dict[int, int], graph: nx.Graph = None) -> float:
        """Compute modularity of the partition on the specified graph."""
        if graph is None:
            graph = self.original_graph
            
        Q = 0.0
        community_set = set(communities.values())
        
        m = graph.size(weight='weight')
        if m == 0:
            return 0.0
        
        for comm in community_set:
            nodes_in_comm = [n for n, c in communities.items() if c == comm]
            
            l_c = 0
            for u in nodes_in_comm:
                if u not in graph:
                    continue
                for v in graph.neighbors(u):
                    if v in nodes_in_comm and communities.get(v) == comm:
                        l_c += graph[u][v].get('weight', 1)
            l_c = l_c / 2
            
            d_c = sum(graph.degree(n, weight='weight') 
                     for n in nodes_in_comm if n in graph)
            
            Q += l_c / m - self.resolution * (d_c / (2 * m)) ** 2
        
        return Q
    
    def phase1_dynamic(self, graph: nx.Graph) -> Tuple[Dict[int, int], bool]:
        """
        Phase 1: Dynamic node movement with improved iteration.
        Only processes nodes that changed or have neighbors that changed.
        
        This is the key improvement: instead of iterating over all nodes,
        only iterate over nodes that were affected in the previous iteration.
        """
        communities = {node: node for node in graph.nodes()}
        
        m = graph.size(weight='weight')
        if m == 0:
            return communities, False
        
        community_degrees = {node: graph.degree(node, weight='weight') 
                            for node in graph.nodes()}
        
        active_nodes = set(graph.nodes())
        global_improvement = False
        iteration = 0
        
        while active_nodes and iteration < 100:
            iteration += 1
            local_improvement = False
            next_active = set()
            
            nodes_to_process = list(active_nodes)
            np.random.shuffle(nodes_to_process)
            
            for node in nodes_to_process:
                current_comm = communities[node]
                node_degree = graph.degree(node, weight='weight')
                
                neighbor_comm_weights = defaultdict(float)
                for neighbor in graph.neighbors(node):
                    comm = communities[neighbor]
                    weight = graph[node][neighbor].get('weight', 1)
                    neighbor_comm_weights[comm] += weight
                
                k_i_in_old = neighbor_comm_weights.get(current_comm, 0)
                sigma_tot_old = community_degrees[current_comm]
                
                best_comm = current_comm
                best_delta_Q = 0.0
                
                for comm, k_i_in_new in neighbor_comm_weights.items():
                    if comm == current_comm:
                        continue
                    
                    sigma_tot_new = community_degrees[comm]
                    
                    delta_Q_add = (k_i_in_new - self.resolution * sigma_tot_new * node_degree / m) / m
                    delta_Q_remove = -(k_i_in_old - self.resolution * (sigma_tot_old - node_degree) * node_degree / m) / m
                    
                    total_delta_Q = delta_Q_add + delta_Q_remove
                    
                    if total_delta_Q > best_delta_Q:
                        best_delta_Q = total_delta_Q
                        best_comm = comm
                
                if best_comm != current_comm and best_delta_Q > self.min_modularity_gain:
                    community_degrees[current_comm] -= node_degree
                    community_degrees[best_comm] += node_degree
                    
                    communities[node] = best_comm
                    local_improvement = True
                    global_improvement = True
                    
                    next_active.add(node)
                    for neighbor in graph.neighbors(node):
                        next_active.add(neighbor)
            
            active_nodes = next_active
            
            if not local_improvement:
                break
        
        return communities, global_improvement
    
    def phase2_aggregate(self, graph: nx.Graph, 
                        communities: Dict[int, int]) -> nx.Graph:
        """
        Phase 2: Create new graph where nodes are communities.
        """
        new_graph = nx.Graph()
        
        comm_nodes = defaultdict(list)
        for node, comm in communities.items():
            comm_nodes[comm].append(node)
        
        for comm in comm_nodes.keys():
            new_graph.add_node(comm)
        
        comm_edges = defaultdict(float)
        for node in graph.nodes():
            node_comm = communities[node]
            for neighbor in graph.neighbors(node):
                neighbor_comm = communities[neighbor]
                weight = graph[node][neighbor].get('weight', 1)
                
                if node_comm < neighbor_comm:
                    edge = (node_comm, neighbor_comm)
                    comm_edges[edge] += weight
                elif node_comm == neighbor_comm:
                    edge = (node_comm, node_comm)
                    comm_edges[edge] += weight / 2
        
        for (u, v), weight in comm_edges.items():
            if u == v:
                new_graph.add_edge(u, v, weight=weight)
            else:
                if new_graph.has_edge(u, v):
                    new_graph[u][v]['weight'] += weight
                else:
                    new_graph.add_edge(u, v, weight=weight)
        
        return new_graph
    
    def merge_small_communities(self, communities: Dict[int, int]) -> Dict[int, int]:
        """
        Merge communities that are too small into neighboring communities.
        This helps avoid over-fragmentation.
        """
        comm_sizes = defaultdict(int)
        for node, comm in communities.items():
            comm_sizes[comm] += 1
        
        small_comms = {comm for comm, size in comm_sizes.items() 
                      if size < self.min_community_size}
        
        if not small_comms:
            return communities
        
        merged = communities.copy()
        
        for small_comm in small_comms:
            nodes_in_small = [n for n, c in merged.items() if c == small_comm]
            
            neighbor_comms = defaultdict(float)
            for node in nodes_in_small:
                for neighbor in self.original_graph.neighbors(node):
                    if merged[neighbor] != small_comm:
                        weight = self.original_graph[node][neighbor].get('weight', 1)
                        neighbor_comms[merged[neighbor]] += weight
            
            if neighbor_comms:
                best_neighbor = max(neighbor_comms.items(), key=lambda x: x[1])[0]
                for node in nodes_in_small:
                    merged[node] = best_neighbor
        
        return merged
    
    def optimize_communities(self, communities: Dict[int, int]) -> Dict[int, int]:
        """
        Optimization step: Try to improve modularity with local adjustments.
        """
        optimized = communities.copy()
        improved = True
        iterations = 0
        
        while improved and iterations < 10:
            iterations += 1
            improved = False
            
            for node in self.original_graph.nodes():
                current_comm = optimized[node]
                
                neighbor_comms = set()
                for neighbor in self.original_graph.neighbors(node):
                    neighbor_comms.add(optimized[neighbor])
                
                best_comm = current_comm
                best_mod = self.compute_modularity(optimized)
                
                for test_comm in neighbor_comms:
                    if test_comm == current_comm:
                        continue
                    
                    test_partition = optimized.copy()
                    test_partition[node] = test_comm
                    test_mod = self.compute_modularity(test_partition)
                    
                    if test_mod > best_mod + self.min_modularity_gain:
                        best_mod = test_mod
                        best_comm = test_comm
                        improved = True
                
                if best_comm != current_comm:
                    optimized[node] = best_comm
        
        return optimized
    
    def detect_communities(self) -> Dict[int, int]:
        """
        Main algorithm: Improved Fast Louvain community detection.
        
        Steps:
        1. Split tree structures from the network
        2. Apply dynamic iteration on reduced network
        3. Add tree structures back and optimize
        """
        print("="*60)
        print("Step 1: Detecting and splitting tree structures...")
        
        tree_structures, reduced_graph = self.detect_tree_structures(self.original_graph)
        
        print(f"Found {len(tree_structures)} tree structures")
        total_tree_nodes = sum(len(tree) for tree in tree_structures)
        print(f"Total nodes in trees: {total_tree_nodes}")
        print(f"Reduced graph: {reduced_graph.number_of_nodes()} nodes, {reduced_graph.number_of_edges()} edges")
        
        print("\n" + "="*60)
        print("Step 2: Applying dynamic iteration on reduced network...")
        
        if reduced_graph.number_of_nodes() == 0:
            print("Graph consists entirely of tree structures")
            final_communities = {}
            for i, tree in enumerate(tree_structures):
                for node in tree:
                    final_communities[node] = i
            return final_communities
        
        current_graph = reduced_graph
        dendrogram = []
        
        self.node_degrees = {node: current_graph.degree(node, weight='weight') 
                            for node in current_graph.nodes()}
        
        prev_modularity = -1
        max_iterations = 100
        
        for iteration in range(max_iterations):
            communities, changed = self.phase1_dynamic(current_graph)
            
            if not changed:
                print(f"Stopped at iteration {iteration}: no changes")
                break
            
            current_level_mod = self.compute_modularity(communities, current_graph)
            
            if iteration == 0:
                original_communities = communities
            else:
                original_communities = {}
                for orig_node in reduced_graph.nodes():
                    curr_comm = orig_node
                    for level in dendrogram:
                        curr_comm = level.get(curr_comm, curr_comm)
                    curr_comm = communities.get(curr_comm, curr_comm)
                    original_communities[orig_node] = curr_comm
            
            original_mod = self.compute_modularity(original_communities, reduced_graph)
            
            print(f"Iteration {iteration}: Level Mod = {current_level_mod:.4f}, Original Mod = {original_mod:.4f}, Communities = {len(set(communities.values()))}")
            
            if abs(original_mod - prev_modularity) < self.min_modularity_gain:
                print(f"Stopped at iteration {iteration}: modularity change < threshold")
                break
            
            prev_modularity = original_mod
            dendrogram.append(communities.copy())
            
            new_graph = self.phase2_aggregate(current_graph, communities)
            
            if new_graph.number_of_nodes() == current_graph.number_of_nodes():
                print(f"Stopped at iteration {iteration}: no aggregation")
                break
            
            current_graph = new_graph
        
        final_communities = {node: node for node in reduced_graph.nodes()}
        
        for level_communities in dendrogram:
            new_final = {}
            for node, comm in final_communities.items():
                new_final[node] = level_communities.get(comm, comm)
            final_communities = new_final
        
        print("\n" + "="*60)
        print("Step 3: Adding tree structures back and optimizing...")
        
        tree_assignments = self.assign_tree_communities(tree_structures, final_communities)
        final_communities.update(tree_assignments)
        
        print("Optimizing final partition...")
        final_communities = self.optimize_communities(final_communities)
        
        if self.min_community_size > 1:
            print(f"Merging communities smaller than {self.min_community_size} nodes...")
            final_communities = self.merge_small_communities(final_communities)
        
        unique_comms = sorted(set(final_communities.values()))
        comm_mapping = {old: new for new, old in enumerate(unique_comms)}
        final_communities = {node: comm_mapping[comm] 
                            for node, comm in final_communities.items()}
        
        return final_communities


