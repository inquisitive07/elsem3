import networkx as nx
import random
import numpy as np

# -----------------------------
# RANDOM WALK UNTIL DANGER
# -----------------------------
def random_walk_until_danger(G, start, danger_nodes, max_steps=100):
    current = start
    for step in range(max_steps):
        if current in danger_nodes:
            return step

        neighbors = list(G.successors(current))
        if not neighbors:
            return None

        current = random.choice(neighbors)

    return None


# -----------------------------
# MEAN HITTING TIME (MHT)
# -----------------------------
def compute_mht(G, danger_nodes, trials=500):
    if not danger_nodes:
        return float("inf")

    safe_nodes = [n for n in G.nodes() if n not in danger_nodes]

    if not safe_nodes:
        return 0.0

    steps = []

    for _ in range(trials):
        start = random.choice(safe_nodes)
        result = random_walk_until_danger(G, start, danger_nodes)

        if result is not None:
            steps.append(result)

    if not steps:
        return float("inf")

    return np.mean(steps)


# -----------------------------
# PARTITION GRAPH BY CLUSTER
# -----------------------------
def partition_graph_by_cluster(G):
    cluster_map = nx.get_node_attributes(G, "cluster")

    if not cluster_map:
        raise ValueError("Graph nodes do not contain 'cluster' attribute")

    partition_A = {n for n, c in cluster_map.items() if c % 2 == 0}
    partition_B = set(G.nodes()) - partition_A

    return partition_A, partition_B


# -----------------------------
# RANDOM WALK CONTROVERSY (RWC)
# -----------------------------
def compute_rwc(G, A, B, trials=500, walk_length=20):
    cross = 0
    stay = 0

    if not A or not B:
        return 0.0

    for _ in range(trials):
        current = random.choice(list(A))

        for _ in range(walk_length):
            neighbors = list(G.successors(current))
            if not neighbors:
                break
            current = random.choice(neighbors)

        if current in B:
            cross += 1
        else:
            stay += 1

    if cross + stay == 0:
        return 0.0

    return cross / (cross + stay)


# -----------------------------
# ECHO CHAMBER INDEX (ECI)
# -----------------------------
def compute_eci(mht, rwc, max_mht=50):
    if mht == float("inf"):
        mht_risk = 0.0
    else:
        mht_risk = 1 - min(mht / max_mht, 1.0)

    return 0.5 * mht_risk + 0.5 * rwc


# -----------------------------
# ANALYSIS DRIVER
# -----------------------------
def analyze_echo_chamber(G, danger_nodes, verbose=True):
    mht = compute_mht(G, danger_nodes)

    A, B = partition_graph_by_cluster(G)
    rwc = compute_rwc(G, A, B)

    eci = compute_eci(mht, rwc)

    if verbose:
        print("\n" + "=" * 50)
        print("ECHO CHAMBER ANALYSIS")
        print("=" * 50)
        print(f"Total nodes: {G.number_of_nodes()}")
        print(f"Total edges: {G.number_of_edges()}")
        print(f"Danger nodes: {len(danger_nodes)}")

        print(f"\nMean Hitting Time (MHT): {mht:.2f}")
        print(f"Random Walk Controversy (RWC): {rwc:.2f}")
        print(f"Echo Chamber Index (ECI): {eci:.2f}")
        print(f"Risk Level: {get_risk_level(eci)}")
        print("=" * 50 + "\n")

    return {
        "mht": mht,
        "rwc": rwc,
        "eci": eci,
        "risk_level": get_risk_level(eci)
    }


def get_risk_level(eci):
    if eci > 0.6:
        return "HIGH"
    elif eci > 0.3:
        return "MEDIUM"
    return "LOW"


# -----------------------------
# EXECUTION ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    # Demo graph so the script actually runs
    G = nx.DiGraph()

    # Nodes with cluster labels
    for i in range(12):
        G.add_node(i, cluster=i % 3)

    # Directed edges
    edges = [
        (0,1),(1,2),(2,3),(3,4),(4,5),
        (5,6),(6,7),(7,8),(8,9),(9,10),
        (10,11),(11,0),(3,7),(5,2)
    ]
    G.add_edges_from(edges)

    # Example danger nodes
    danger_nodes = {2, 5, 7}

    analyze_echo_chamber(G, danger_nodes)
