import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
import sys
from pathlib import Path
import plotly.graph_objects as go

# -------------------- PATH SETUP --------------------
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

MAIN_DIR = ROOT_DIR / "main"
OUTPUT_DIR = ROOT_DIR / "output"
DATA_DIR = MAIN_DIR / "data"

# Import backend functions
from main.collection.scraper import scrape_recommendations
from main.collection.buildgraph import build_graph
from analyzer import EchoChamberAnalyzer
from main.Validator.validator import run_validation
from eci_analyzer import compute_mht, compute_rwc, compute_eci, partition_graph_by_cluster


def normalize_pos(pos, padding=0.1):
    """Normalize graph positions to [0,1] range with padding"""
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    norm_pos = {}
    for k, (x, y) in pos.items():
        nx = (x - min_x) / (max_x - min_x + 1e-9)
        ny = (y - min_y) / (max_y - min_y + 1e-9)

        nx = nx * (1 - 2 * padding) + padding
        ny = ny * (1 - 2 * padding) + padding

        norm_pos[k] = (nx, ny)

    return norm_pos


def run_sbert_analysis():
    """Run SBERT clustering analysis"""
    analyzer = EchoChamberAnalyzer()
    
    input_file = DATA_DIR / "recommendation_walk.csv"
    
    if not input_file.exists():
        raise FileNotFoundError(
            f"recommendation_walk.csv not found at {input_file}"
        )

    # Load data
    analyzer.load_data(input_file)
    
    # SBERT pipeline
    analyzer.create_embeddings()
    analyzer.cluster_videos(n_clusters=5)
    
    # Generate outputs
    analyzer.visualize_clusters(OUTPUT_DIR / "clusters_2d.png")
    analyzer.generate_sbert_dataset(OUTPUT_DIR / "sbert_dataset.csv")
    
    # Calculate echo score
    user_indices = list(range(len(analyzer.df)))
    score = analyzer.calculate_user_echo_score(user_indices)
    
    return score


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Algorithmic Amplification Auditor",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
img {
    max-height: 600px;
    object-fit: contain;
}
.metric-box {
    background-color: #161b22;
    padding: 28px 20px;
    border-radius: 18px;
    text-align: center;
    color: white;
    border: 1.5px solid #30363d;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    height: 210px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.metric-box h3 {
    font-size: 15px;
    font-weight: 600;
    color: #c9d1d9;
    margin-bottom: 12px;
}

.metric-box h1 {
    font-size: 54px;
    font-weight: 700;
    margin: 6px 0;
    line-height: 1.1;
}

.metric-box p {
    font-size: 14px;
    color: #8b949e;
    margin-top: 6px;
}

.metric-box:hover {
    transform: translateY(-4px);
    transition: all 0.2s ease;
    border-color: #58a6ff;
}

.title {
    color: #58a6ff;
    font-size: 40px;
    font-weight: bold;
    text-align: center;
}
.subtitle {
    color: #8b949e;
    text-align: center;
}
.tooltip-container {
    position: relative;
    display: inline-block;
    cursor: help;
}
.tooltip-text {
    visibility: hidden;
    width: 280px;
    background-color: #e6e6fa;
    color: #000;
    text-align: left;
    padding: 15px 18px;
    border-radius: 8px;
    font-size: 14px;
    line-height: 1.4;
    position: absolute;
    z-index: 1000;
    bottom: 120%;
    left: 50%;
    transform: translateX(-50%);
    box-shadow: 0 8px 20px rgba(0,0,0,0.6);
    border: 1px solid #30363d;
}

.metric-box:hover .tooltip-text {
    visibility: visible;
}

div[data-testid="stDataFrame"] {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border-radius: 10px;
    padding: 15px;
    border: 1px solid #30363d;
}

div[data-testid="stDataFrame"] thead tr th {
    background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%) !important;
    color: #58a6ff !important;
    font-weight: bold !important;
    border-bottom: 2px solid #58a6ff !important;
}

div[data-testid="stDataFrame"] tbody tr:nth-child(odd) {
    background-color: rgba(88, 166, 255, 0.05) !important;
}

div[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background-color: rgba(139, 148, 158, 0.03) !important;
}

div[data-testid="stDataFrame"] tbody tr:hover {
    background-color: rgba(88, 166, 255, 0.1) !important;
    transform: translateX(2px);
    transition: all 0.2s ease;
}
</style>
""", unsafe_allow_html=True)


# -------------------- TITLE --------------------
st.markdown("""
<div style="display:flex; flex-direction:column; align-items:center;">
    <div class="title">Algorithmic Amplification Auditor</div>
    <div class="subtitle">Live Audit: YouTube Recommendation Graph</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# -------------------- INPUT --------------------
st.subheader("🔗 Enter YouTube Video URL")
youtube_url = st.text_input("Paste a YouTube link to audit")

run = st.button("Run Algorithmic Audit")

tab1, tab2, tab3 = st.tabs(["🏠 Home", "🕸 Pathways & Clusters", "📊 Data Logs"])

# -------------------- MAIN LOGIC --------------------

if run and youtube_url:
    status_container = st.container()
    
    try:
        with status_container:
            # Phase 1: Scraping
            with st.spinner("Phase 1: Scraping YouTube recommendations..."):
                scrape_recommendations(youtube_url, max_steps=50)
                st.success("✓ Scraping complete!")

            # Phase 2: SBERT Analysis
            with st.spinner("Phase 2: Running SBERT clustering analysis..."):
                echo_score = run_sbert_analysis()
                st.success("✓ SBERT analysis complete!")

            # Phase 3: Graph Building
            with st.spinner("Phase 3: Building recommendation graph..."):
                G = build_graph()
                st.success("✓ Graph built with cluster labels!")

            # Phase 4: Validation
            with st.spinner("Phase 4: Validating clusters..."):
                validated_df = run_validation()
                st.success("✓ Validation complete!")
        
            # Phase 5: Risk Analysis
            with st.spinner("Phase 5: Algorithmic Risk Analysis (ECI)..."):
                # Load necessary data
                G_loaded = nx.read_gexf(OUTPUT_DIR / "recommendation_graph.gexf")
                sbert_df = pd.read_csv(OUTPUT_DIR / "sbert_dataset.csv")
                risk_df = pd.read_csv(OUTPUT_DIR / "validated_clusters.csv")

                # Identify danger nodes
                danger_clusters = risk_df[
                    risk_df["label"] == "Radicalization Node"
                ]["cluster_id"].tolist()

                danger_nodes = sbert_df[
                    sbert_df["cluster"].isin(danger_clusters)
                ]["url"].tolist()

                if not danger_nodes:
                    mht = float("inf")
                    rwc = 0.0
                    eci = 0.0
                    risk_level = "LOW"
                    
                    st.info(
                        f"✓ No radicalization clusters detected. "
                        f"Analyzed {len(sbert_df)} videos across {sbert_df['cluster'].nunique()} clusters. "
                        f"Algorithmic amplification risk is LOW."
                    )
                else:
                    mht = compute_mht(G_loaded, danger_nodes, trials=500)
                    A, B = partition_graph_by_cluster(G_loaded)
                    rwc = compute_rwc(G_loaded, A, B, trials=500)
                    eci = compute_eci(mht, rwc)

                    if eci > 0.6:
                        risk_level = "HIGH"
                    elif eci > 0.3:
                        risk_level = "MEDIUM"
                    else:
                        risk_level = "LOW"
                    
                    st.warning(
                        f"⚠️ Detected {len(danger_nodes)} videos in {len(danger_clusters)} "
                        f"radicalization clusters. Risk level: {risk_level}"
                    )

                st.success("✓ Algorithmic risk analysis complete!")

        # Clear status messages
        status_container.empty()

        # -------------------- PREPARE VISUALIZATION DATA --------------------
        
        # Calculate consensus
        consensus = validated_df["consensus"].mean() * 100

        # Load recommendation walk data
        walk_df = pd.read_csv(DATA_DIR / "recommendation_walk.csv")

        # Prepare graph visualization
        title_dict = dict(zip(sbert_df['url'], sbert_df['title']))
        pos_raw = nx.spring_layout(G_loaded, seed=42, k=0.4, iterations=100)
        pos = normalize_pos(pos_raw)
        
        # Color nodes by cluster
        unique_clusters = sorted(sbert_df['cluster'].unique())
        num_clusters = len(unique_clusters)

        base_colors = go.Figure().layout.colorway
        if base_colors is None or len(base_colors) < num_clusters:
            base_colors = [
                f"hsl({int(360*i/num_clusters)},70%,60%)"
                for i in range(num_clusters)
            ]

        cluster_colors = {
            cluster_id: base_colors[i % len(base_colors)]
            for i, cluster_id in enumerate(unique_clusters)
        }
        
        node_colors = [
            cluster_colors[sbert_df[sbert_df['url'] == node]['cluster'].iloc[0]] 
            if node in sbert_df['url'].values else "#58a6ff"
            for node in G_loaded.nodes()
        ]

        # Build Plotly graph
        edge_x, edge_y = [], []
        for edge in G_loaded.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x, node_y, node_text = [], [], []
        for node in G_loaded.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            title = title_dict.get(node, 'Unknown')
            node_text.append(f"Title: {title}<br>URL: {node}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=False,
                color=node_colors,
                size=28,
                line_width=2
            )
        )

        plotly_fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117'
            )
        )

        # ==================== HOME TAB ====================
        with tab1:
            # Format display values
            mht_display = "∞" if mht == float("inf") else f"{mht:.1f}"
            eci_display = f"{eci:.2f}"
            rwc_display = f"{rwc:.2f}"

            # -------------------- METRICS --------------------
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="tooltip-container">
                        <h3>Echo Chamber Index</h3>
                        <span class="tooltip-text">
                            Measures overall risk of algorithmic amplification into echo chambers.
                            Range: 0 (safe) to 1 (maximum risk).
                        </span>
                    </div>
                    <h1 style="color:#ff4c4c">{eci_display}</h1>
                    <p>Algorithmic Risk Score</p>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="tooltip-container">
                        <h3>Random Walk Controversy</h3>
                        <span class="tooltip-text">
                            Probability that random browsing gets trapped in controversial clusters.
                            Higher = stronger funneling effect.
                        </span>
                    </div>
                    <h1 style="color:#e3b341">{rwc_display}</h1>
                    <p>Probability of Trap</p>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="tooltip-container">
                        <h3>Semantic Consensus</h3>
                        <span class="tooltip-text">
                            Average agreement level in clusters. Shows how strongly videos
                            share the same sentiment or opinion.
                        </span>
                    </div>
                    <h1 style="color:#3fb950">{consensus:.1f}%</h1>
                    <p>Average Agreement</p>
                </div>
                """, unsafe_allow_html=True)

            with c4:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="tooltip-container">
                        <h3>Mean Hitting Time</h3>
                        <span class="tooltip-text">
                            Average clicks to reach dangerous content.
                            Lower means danger is more accessible. ∞ indicates no danger detected.
                        </span>
                    </div>
                    <h1 style="color:#d2a8ff">{mht_display}</h1>
                    <p>Clicks to Danger</p>
                </div>
                """, unsafe_allow_html=True)

            # Risk level
            risk_color = {
                "LOW": "#3fb950",
                "MEDIUM": "#e3b341",
                "HIGH": "#ff4c4c"
            }[risk_level]

            st.divider()
            st.subheader("🚨 Overall Algorithmic Risk Level")

            st.markdown(f"""
            <div class="metric-box tooltip-container" style="width:100%; margin: 0 auto;">
                <h1 style="color:{risk_color}">
                    {risk_level}
                    <span class="tooltip-text">
                        Combines MHT and RWC into overall risk assessment.
                        <br><br>
                        <b>LOW</b> (ECI < 0.3): Minimal risk<br>
                        <b>MEDIUM</b> (0.3 ≤ ECI < 0.6): Noticeable patterns<br>
                        <b>HIGH</b> (ECI ≥ 0.6): Strong amplification
                    </span>
                </h1>
                <p>Based on ECI (MHT + RWC)</p>
            </div>
            """, unsafe_allow_html=True)

        # ==================== PATHWAYS & CLUSTERS TAB ====================
        with tab2:
            st.subheader("🕸 Recommendation Pathways")
            st.plotly_chart(plotly_fig, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.divider()
            
            st.subheader("📈 Semantic Cluster Geometry")
            cluster_img_path = OUTPUT_DIR / "clusters_2d.png"
            
            if cluster_img_path.exists():
                st.image(
                    str(cluster_img_path),
                    use_container_width=True,
                    caption="2D projection of recommendation clusters"
                )
            else:
                st.warning("Cluster visualization not found.")

        # ==================== DATA LOGS TAB ====================
        with tab3:
            st.subheader("📋 Recommendation Walk Data")
            st.caption("Sequential list of videos encountered during the audit walk")
            
            walk_display = walk_df[['step', 'title', 'url']].copy()
            walk_display.columns = ['Step', 'Video Title', 'URL']
            
            st.dataframe(
                walk_display,
                use_container_width=True,
                hide_index=True,
                height=350
            )
            
            st.divider()
            
            st.subheader("🔍 Cluster Risk Classification")
            st.caption("Summary of semantic clusters and their risk assessment")
            
            validated_display = validated_df[['cluster_id', 'size', 'label', 'consensus', 'subjectivity']].copy()
            validated_display.columns = ['Cluster', 'Videos', 'Risk', 'Agreement', 'Emotion']
            
            validated_display['Agreement'] = validated_display['Agreement'].round(2)
            validated_display['Emotion'] = validated_display['Emotion'].round(2)
            
            st.dataframe(
                validated_display,
                use_container_width=True,
                hide_index=True,
                height=250
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Videos", len(walk_df), delta=None)
            with col2:
                st.metric("Total Clusters", len(validated_df), delta=None)
            with col3:
                radicalized_count = (validated_df['label'] == 'Radicalization Node').sum()
                
                # Smart risk labeling based on count
                if radicalized_count >= 4:
                    delta_text = "⚠️ High Risk"
                elif radicalized_count >= 1:
                    delta_text = "⚠️ Echo Chamber"
                else:
                    delta_text = "✓ Safe"
                    
                st.metric("Radicalization Nodes", radicalized_count, delta=delta_text)

    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("Make sure all pipeline phases completed successfully.")
    
    except Exception as e:
        st.error(f"❌ Error during audit: {e}")
        st.exception(e)
        st.info("Check the console for detailed error messages.")

else:
    with tab1:
        st.info("👆 Enter a YouTube link above and click **Run Algorithmic Audit** to begin.")