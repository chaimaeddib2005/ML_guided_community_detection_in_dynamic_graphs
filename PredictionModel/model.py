"""
Machine Learning Model for Predicting Community Changes in Dynamic Graphs

This module trains a model to predict which nodes will change communities
after graph modifications (edge additions/deletions).

Approach:
1. Generate training data from graph modifications
2. Extract node features before modification
3. Label nodes that changed communities after modification
4. Train a classifier to predict which nodes will move
"""

import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from typing import Dict, List, Tuple, Set
import pickle


class DynamicCommunityPredictor:
    def __init__(self, louvain_detector):
        """
        Initialize the dynamic community predictor.
        
        Args:
            louvain_detector: Instance of ImprovedFastLouvain class
        """
        self.louvain = louvain_detector
        self.model = None
        self.feature_names = []
        
    def extract_node_features(self, graph: nx.Graph, node: int, 
                             communities: Dict[int, int]) -> np.ndarray:
        """
        Extract features for a node that will be used to predict if it moves.
        
        Features include:
        - Degree centrality
        - Betweenness centrality
        - Clustering coefficient
        - Community cohesion (internal vs external edges)
        - Neighbor community diversity
        - PageRank
        - Core number
        """
        features = []
        
        if node not in graph:
            return np.zeros(15)
        
        n = graph.number_of_nodes()
        node_comm = communities.get(node, -1)
        
        degree = graph.degree(node, weight='weight')
        features.append(degree)
        features.append(degree / n if n > 0 else 0)
        
        try:
            bc = nx.betweenness_centrality(graph, weight='weight').get(node, 0)
        except:
            bc = 0
        features.append(bc)
        
        try:
            cc = nx.clustering(graph, node, weight='weight')
        except:
            cc = 0
        features.append(cc)
        
        neighbors = list(graph.neighbors(node))
        if len(neighbors) > 0:
            internal_edges = sum(1 for n in neighbors if communities.get(n) == node_comm)
            external_edges = len(neighbors) - internal_edges
            features.append(internal_edges)
            features.append(external_edges)
            features.append(internal_edges / len(neighbors))
            
            neighbor_comms = [communities.get(n, -1) for n in neighbors]
            unique_comms = len(set(neighbor_comms))
            features.append(unique_comms)
            features.append(unique_comms / len(neighbors))
        else:
            features.extend([0, 0, 0, 0, 0])
        
        try:
            pr = nx.pagerank(graph, weight='weight').get(node, 0)
        except:
            pr = 0
        features.append(pr)
        
        try:
            core = nx.core_number(graph).get(node, 0)
        except:
            core = 0
        features.append(core)
        
        comm_nodes = [n for n, c in communities.items() if c == node_comm]
        if len(comm_nodes) > 1:
            comm_size = len(comm_nodes)
            features.append(comm_size)
            features.append(comm_size / n if n > 0 else 0)
        else:
            features.append(1)
            features.append(1 / n if n > 0 else 0)
        
        try:
            eigenvector = nx.eigenvector_centrality(graph, weight='weight', max_iter=100).get(node, 0)
        except:
            eigenvector = 0
        features.append(eigenvector)
        
        neighbor_degrees = [graph.degree(n, weight='weight') for n in neighbors]
        features.append(np.mean(neighbor_degrees) if neighbor_degrees else 0)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        return [
            'degree',
            'degree_normalized',
            'betweenness_centrality',
            'clustering_coefficient',
            'internal_edges',
            'external_edges',
            'internal_ratio',
            'neighbor_communities_count',
            'neighbor_communities_diversity',
            'pagerank',
            'core_number',
            'community_size',
            'community_size_normalized',
            'eigenvector_centrality',
            'avg_neighbor_degree'
        ]
    
    def generate_training_data(self, base_graph: nx.Graph, 
                              num_modifications: int = 100,
                              modification_type: str = 'both') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data by simulating graph modifications.
        
        Args:
            base_graph: Original graph
            num_modifications: Number of graph modifications to simulate
            modification_type: 'add', 'remove', or 'both'
            
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Labels (1 if node changed community, 0 otherwise)
        """
        X_list = []
        y_list = []
        
        print(f"Generating {num_modifications} training samples...")
        
        for i in range(num_modifications):
            if i % 10 == 0:
                print(f"Progress: {i}/{num_modifications}")
            
            G = base_graph.copy()
            
            self.louvain.original_graph = G
            self.louvain.graph = G
            self.louvain.m = G.size(weight='weight')
            communities_before = self.louvain.detect_communities()
            
            if modification_type == 'add' or (modification_type == 'both' and np.random.rand() > 0.5):
                nodes = list(G.nodes())
                if len(nodes) >= 2:
                    u, v = np.random.choice(nodes, 2, replace=False)
                    if not G.has_edge(u, v):
                        weight = np.random.uniform(1, 5)
                        G.add_edge(u, v, weight=weight)
                        affected_nodes = {u, v}
            else:
                edges = list(G.edges())
                if len(edges) > 0:
                    u, v = edges[np.random.randint(len(edges))]
                    G.remove_edge(u, v)
                    affected_nodes = {u, v}
                else:
                    continue
            
            for node in G.nodes():
                features = self.extract_node_features(base_graph, node, communities_before)
                X_list.append(features)
            
            self.louvain.original_graph = G
            self.louvain.graph = G
            self.louvain.m = G.size(weight='weight')
            communities_after = self.louvain.detect_communities()
            
            for node in G.nodes():
                changed = 1 if communities_before.get(node, -1) != communities_after.get(node, -1) else 0
                y_list.append(changed)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"\nGenerated {len(X)} samples")
        print(f"Positive samples (nodes that changed): {np.sum(y)} ({100*np.mean(y):.2f}%)")
        print(f"Negative samples (nodes that stayed): {len(y) - np.sum(y)} ({100*(1-np.mean(y)):.2f}%)")
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, model_type: str = 'random_forest'):
        """
        Train the prediction model.
        
        Args:
            X: Feature matrix
            y: Labels
            model_type: 'random_forest' or 'gradient_boosting'
        """
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Training {model_type}...")
        self.model.fit(X_train, y_train)
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        print("\nTRAINING SET PERFORMANCE:")
        print(classification_report(y_train, y_pred_train))
        
        print("\nTEST SET PERFORMANCE:")
        print(classification_report(y_test, y_pred_test))
        
        try:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"\nROC-AUC Score: {roc_auc:.4f}")
        except:
            pass
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_names = self.get_feature_names()
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("\nFEATURE IMPORTANCES:")
            for i in range(min(10, len(indices))):
                idx = indices[i]
                print(f"  {i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
        
        return self.model
    
    def predict_affected_nodes(self, graph_before: nx.Graph, 
                              graph_after: nx.Graph,
                              communities_before: Dict[int, int],
                              threshold: float = 0.5) -> Set[int]:
        """
        Predict which nodes will change communities after graph modification.
        
        Args:
            graph_before: Graph before modification
            graph_after: Graph after modification
            communities_before: Community assignments before modification
            threshold: Probability threshold for prediction
            
        Returns:
            Set of node IDs predicted to change communities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        predicted_nodes = set()
        
        for node in graph_before.nodes():
            features = self.extract_node_features(graph_before, node, communities_before)
            features = features.reshape(1, -1)
            
            proba = self.model.predict_proba(features)[0, 1]
            
            if proba >= threshold:
                predicted_nodes.add(node)
        
        return predicted_nodes
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {filepath}")


