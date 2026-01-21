import os
import pandas as pd
import networkx as nx
from pathlib import Path

def build_graph():
    # Get project root (same logic as scraper)
    BASE_DIR = Path(__file__).resolve().parents[2]
    
    # Input from scraper output location
    DATA_DIR = BASE_DIR / "main" / "data"
    WALK_FILE = DATA_DIR / "recommendation_walk.csv"
    
    # Output location
    OUTPUT_DIR = BASE_DIR / "output"
    SBERT_FILE = OUTPUT_DIR / "sbert_dataset.csv"
    GRAPH_FILE = OUTPUT_DIR / "recommendation_graph.gexf"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Looking for walk file at: {WALK_FILE}")
    
    if not WALK_FILE.exists():
        raise FileNotFoundError(
            f"recommendation_walk.csv not found at {WALK_FILE}\n"
            f"Make sure scraper.py completed successfully."
        )
    
    print("Loading recommendation walk CSV...")
    walk_df = pd.read_csv(WALK_FILE).sort_values("step")
    
    print(f"📂 Looking for SBERT file at: {SBERT_FILE}")
    
    if not SBERT_FILE.exists():
        raise FileNotFoundError(
            f"sbert_dataset.csv not found at {SBERT_FILE}\n"
            f"Make sure analyzer.py completed successfully."
        )
    
    print("Loading SBERT dataset...")
    sbert_df = pd.read_csv(SBERT_FILE)
    
    # Map URL → cluster
    cluster_map = dict(zip(sbert_df["url"], sbert_df["cluster"]))
    
    G = nx.DiGraph()
    
    print("Adding nodes with cluster labels...")
    for _, row in walk_df.iterrows():
        url = row["url"]
        if not G.has_node(url):
            G.add_node(
                url,
                title=row["title"],
                step=int(row["step"]),
                cluster=int(cluster_map.get(url, -1))  # -1 if missing
            )
    
    print("Adding edges...")
    for i in range(len(walk_df) - 1):
        G.add_edge(
            walk_df.iloc[i]["url"],
            walk_df.iloc[i + 1]["url"],
            step=int(walk_df.iloc[i]["step"])
        )
    
    nx.write_gexf(G, GRAPH_FILE)
    
    print("\n✅ Graph created successfully!")
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")
    print(f"Clusters on nodes: {set(nx.get_node_attributes(G, 'cluster').values())}")
    print(f"📁 Saved to: {GRAPH_FILE}")
    
    return G

if __name__ == "__main__":
    build_graph()