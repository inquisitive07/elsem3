import pandas as pd
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================

MODEL_NAME = "all-MiniLM-L6-v2"
RANDOM_STATE = 42
RADICAL_TIGHTNESS_THRESHOLD = 0.65


# ===============================
# ECHO CHAMBER ANALYZER CLASS
# ===============================

class EchoChamberAnalyzer:
    """
    Analyzes video recommendations for echo chamber patterns using SBERT embeddings
    """
    
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.df = None
        self.embeddings = None
        self.clusters = None
        
    def load_data(self, csv_path):
        """Load recommendation walk data from CSV"""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"❌ CSV not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        
        if "title" not in self.df.columns:
            raise ValueError("CSV must contain a 'title' column")
        
        # Handle descriptions
        if "description" not in self.df.columns:
            self.df["description"] = ""
        
        self.df["title"] = self.df["title"].fillna("").astype(str)
        self.df["description"] = self.df["description"].fillna("").astype(str)
        
        print(f"✓ Loaded {len(self.df)} videos")
        return self.df
    
    def create_embeddings(self):
        """Create SBERT embeddings from titles and descriptions"""
        if self.df is None:
            raise ValueError("Must load data first using load_data()")
        
        # Combine title and description
        texts = (self.df["title"] + " [SEP] " + self.df["description"]).tolist()
        print(f"📄 Creating embeddings for {len(texts)} documents...")
        
        # Load model if not already loaded
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        
        # Generate embeddings
        self.embeddings = self.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        print(f"✓ Created embeddings with shape {self.embeddings.shape}")
        return self.embeddings
    
    def cluster_videos(self, n_clusters=None):
        """Cluster videos using K-Means"""
        if self.embeddings is None:
            raise ValueError("Must create embeddings first using create_embeddings()")
        
        # Auto-determine k if not provided
        if n_clusters is None:
            n = len(self.df)
            if n < 3:
                n_clusters = 1
            else:
                n_clusters = min(8, max(3, int(np.sqrt(n))))
        
        print(f"📊 Clustering into {n_clusters} clusters...")
        
        if n_clusters == 1:
            self.df["cluster"] = 0
        else:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=RANDOM_STATE,
                n_init="auto"
            )
            self.df["cluster"] = kmeans.fit_predict(self.embeddings)
        
        # Calculate cluster tightness
        tightness = self._compute_cluster_tightness()
        self.df["cluster_tightness"] = self.df["cluster"].map(tightness)
        self.df["potentially_radical"] = self.df["cluster_tightness"] >= RADICAL_TIGHTNESS_THRESHOLD
        
        print(f"✓ Clustered videos into {self.df['cluster'].nunique()} clusters")
        return self.df["cluster"]
    
    def _compute_cluster_tightness(self):
        """Compute tightness score for each cluster"""
        scores = {}
        labels = self.df["cluster"].values
        
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) < 2:
                scores[c] = 0.0
            else:
                cluster_embeds = self.embeddings[idx]
                scores[c] = np.dot(cluster_embeds, cluster_embeds.T).mean()
        
        return scores
    
    def visualize_clusters(self, output_path):
        """Create 2D visualization of clusters"""
        if self.embeddings is None or self.df is None:
            raise ValueError("Must create embeddings and cluster first")
        
        if len(self.df) <= 1:
            print("⚠️ Not enough data points to visualize")
            return
        
        print("🖼️ Creating cluster visualization...")
        
        # Reduce to 2D
        reduced = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(self.embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced[:, 0], 
            reduced[:, 1], 
            c=self.df["cluster"], 
            cmap="tab10", 
            alpha=0.7,
            s=100
        )
        plt.colorbar(scatter, label="Cluster")
        plt.title("SBERT Semantic Clusters", fontsize=16, fontweight='bold')
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved visualization to {output_path}")
    
    def generate_sbert_dataset(self, output_path):
        """Save the dataset with cluster labels to CSV"""
        if self.df is None:
            raise ValueError("Must load and process data first")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        print(f"✓ Saved SBERT dataset to {output_path}")
        
        return output_path
    
    def calculate_user_echo_score(self, user_indices):
        """
        Calculate echo chamber score for a user's viewing history
        
        Args:
            user_indices: List of indices representing user's watch history
            
        Returns:
            str: Formatted report of echo chamber metrics
        """
        if self.df is None:
            raise ValueError("Must load and process data first")
        
        user_df = self.df.iloc[user_indices]
        
        # Calculate metrics
        total_videos = len(user_df)
        unique_clusters = user_df["cluster"].nunique()
        total_clusters = self.df["cluster"].nunique()
        
        cluster_diversity = unique_clusters / total_clusters if total_clusters > 0 else 0
        avg_tightness = user_df["cluster_tightness"].mean()
        radical_count = user_df["potentially_radical"].sum()
        radical_percentage = (radical_count / total_videos * 100) if total_videos > 0 else 0
        
        # Generate report
        report = f"""
╔════════════════════════════════════════════════════════════╗
║           USER ECHO CHAMBER ANALYSIS REPORT                ║
╚════════════════════════════════════════════════════════════╝

📊 VIEWING STATISTICS
─────────────────────────────────────────────────────────────
  Total videos watched        : {total_videos}
  Unique clusters explored    : {unique_clusters} / {total_clusters}
  Cluster diversity score     : {cluster_diversity:.2%}

🎯 ECHO CHAMBER METRICS
─────────────────────────────────────────────────────────────
  Average cluster tightness   : {avg_tightness:.3f}
  Potentially radical content : {radical_count} videos ({radical_percentage:.1f}%)

⚠️  RISK ASSESSMENT
─────────────────────────────────────────────────────────────
  Diversity Risk    : {"HIGH" if cluster_diversity < 0.3 else "MEDIUM" if cluster_diversity < 0.6 else "LOW"}
  Echo Chamber Risk : {"HIGH" if avg_tightness > 0.7 else "MEDIUM" if avg_tightness > 0.5 else "LOW"}
  Content Risk      : {"HIGH" if radical_percentage > 30 else "MEDIUM" if radical_percentage > 10 else "LOW"}

═══════════════════════════════════════════════════════════════
"""
        return report


# ===============================
# STANDALONE EXECUTION
# ===============================

def main():
    """Main function for standalone execution"""
    print("🔍 Starting SBERT Echo Chamber Analysis\n")
    
    # Setup paths
    BASE_DIR = Path(__file__).resolve().parent
    INPUT_CSV = BASE_DIR / "main" / "data" / "recommendation_walk.csv"
    OUTPUT_DIR = BASE_DIR / "output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create analyzer
    analyzer = EchoChamberAnalyzer()
    
    # Run pipeline
    analyzer.load_data(INPUT_CSV)
    analyzer.create_embeddings()
    analyzer.cluster_videos(n_clusters=5)
    analyzer.visualize_clusters(OUTPUT_DIR / "clusters_2d.png")
    analyzer.generate_sbert_dataset(OUTPUT_DIR / "sbert_dataset.csv")
    
    # Calculate echo score for entire walk
    user_indices = list(range(len(analyzer.df)))
    score_report = analyzer.calculate_user_echo_score(user_indices)
    print(score_report)
    
    print("\n✅ SBERT analysis complete")


if __name__ == "__main__":
    main()