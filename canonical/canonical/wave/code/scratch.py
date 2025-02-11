import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def analyze_graph_structure(file_path):
    """
    Analyze the structure of the graph from edge list format.
    
    Args:
        file_path (str): Path to the edge list file
    """
    # Read edges
    edges = pd.read_csv(file_path, sep=' ', header=None, names=['source', 'target'])
    
    # Create networkx graph
    G = nx.from_pandas_edgelist(edges, 'source', 'target')
    
    # Basic statistics
    print("Graph Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Degree analysis
    degrees = [d for n, d in G.degree()]
    degree_counts = Counter(degrees)
    
    print("\nDegree Distribution:")
    for degree, count in sorted(degree_counts.items()):
        print(f"Degree {degree}: {count} nodes")
    
    # Find hubs (nodes with highest degree)
    hub_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 Hub Nodes:")
    for node, degree in hub_nodes:
        print(f"Node {node}: {degree} connections")
    
    # Connected components analysis
    components = list(nx.connected_components(G))
    print(f"\nNumber of connected components: {len(components)}")
    for i, comp in enumerate(components, 1):
        print(f"Component {i} size: {len(comp)} nodes")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()))
    
    # Draw nodes with size proportional to degree
    node_sizes = [G.degree(node) * 50 for node in G.nodes()]
    nx.draw(G, pos,
           node_size=node_sizes,
           node_color='lightblue',
           with_labels=True,
           font_size=8,
           font_weight='bold')
    
    plt.title("Graph Visualization (Node size proportional to degree)")
    plt.show()
    
    return G, components

def analyze_component_structure(G, components):
    """
    Analyze the structure of each component in detail.
    
    Args:
        G (networkx.Graph): The full graph
        components (list): List of connected components
    """
    for i, component in enumerate(components, 1):
        subgraph = G.subgraph(component)
        print(f"\nComponent {i} Analysis:")
        print(f"Nodes: {subgraph.number_of_nodes()}")
        print(f"Edges: {subgraph.number_of_edges()}")
        
        # Calculate clustering coefficient for the component
        clustering_coeff = nx.average_clustering(subgraph)
        print(f"Average clustering coefficient: {clustering_coeff:.4f}")
        
        # Calculate diameter if the component is connected
        if nx.is_connected(subgraph):
            diameter = nx.diameter(subgraph)
            print(f"Diameter: {diameter}")
        
        # Identify central nodes in the component
        degree_centrality = nx.degree_centrality(subgraph)
        most_central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[0]
        print(f"Most central node: {most_central[0]} (centrality: {most_central[1]:.4f})")
        
        # Visualize the component
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(subgraph)
        nx.draw(subgraph, pos,
               node_color='lightblue',
               node_size=300,
               with_labels=True,
               font_size=8,
               font_weight='bold')
        plt.title(f"Component {i} Structure")
        plt.show()
# Example usage
if __name__ == "__main__":
    file_path = "/Users/xingzixu/clustering/canonical/wave/data/facebook_combined.txt"  # Replace with your file path
    G, components = analyze_graph_structure(file_path)
    analyze_component_structure(G, components)

