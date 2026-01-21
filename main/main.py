import os
import sys

# Add project root to sys.path for absolute imports
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# HARD DISABLE THREADING (important for macOS + PyTorch)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

from analyzer import EchoChamberAnalyzer
from pathlib import Path

def run_analysis():
    print("=" * 70)
    print("ECHO CHAMBER & ALGORITHMIC AMPLIFICATION AUDITOR")
    print("=" * 70, "\n")

    analyzer = EchoChamberAnalyzer()

    # Load REAL YouTube recommendation data
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "output"
    OUTPUT_DIR = DATA_DIR
    input_file = DATA_DIR / "recommendation_walk.csv"
    
    if not input_file.exists():
        raise FileNotFoundError(
            "recommendation_walk.csv not found. Please run scraper first."
        )

    # Load data (only once, removed duplicate)
    print(f"📂 Loading data from {input_file}...")
    analyzer.load_data(input_file)
    
    # Check if descriptions are present
    if "description" in analyzer.df.columns:
        desc_count = analyzer.df["description"].notna().sum()
        print(f"✓ Loaded {len(analyzer.df)} videos with {desc_count} descriptions")
    else:
        print(f"✓ Loaded {len(analyzer.df)} videos (titles only)")
    
    # SBERT pipeline
    print("\n🔬 Creating embeddings...")
    analyzer.create_embeddings()
    
    print("📊 Clustering videos...")
    analyzer.cluster_videos(n_clusters=5)

    # Geometry
    print("🖼️ Generating visualizations...")
    analyzer.visualize_clusters(OUTPUT_DIR / "clusters_2d.png")
    
    print("💾 Saving SBERT dataset...")
    analyzer.generate_sbert_dataset(OUTPUT_DIR / "sbert_dataset.csv")
    
    # Echo score for entire walk (simulated user)
    print("\n📈 Calculating user echo chamber score...")
    user_indices = list(range(len(analyzer.df)))
    score = analyzer.calculate_user_echo_score(user_indices)

    print("\n" + "=" * 70)
    print("USER ECHO CHAMBER REPORT")
    print("=" * 70)
    print(score)
    print("\n✅ PIPELINE COMPLETE")
    print("=" * 70)

    return score

if __name__ == "__main__":
    run_analysis()