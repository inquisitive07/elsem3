"""
Phase 3: Semantic Validator
Role: Distinguish harmful ideological echo chambers from benign clusters
Method: Sentiment consensus + subjectivity using VADER
"""
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

# -----------------------------
# TOPIC FILTER (LEVEL 1)
# -----------------------------
SENSITIVE_TOPICS = [
    "politics", "election", "vote", "biden", "trump",
    "religion", "islam", "christian", "atheist",
    "vaccine", "covid", "pandemic", "mask",
    "war", "conflict", "israel", "palestine",
    "conspiracy", "qanon", "deepstate",
    "hate", "racism", "extremism", "radical"
]


def contains_sensitive_topic(texts):
    """Check if cluster discusses potentially polarizing topics"""
    full_text = " ".join(texts).lower()
    matches = [topic for topic in SENSITIVE_TOPICS if topic in full_text]
    return len(matches) > 0, matches


# -----------------------------
# SENTIMENT ANALYSIS
# -----------------------------
def analyze_cluster_sentiment(titles, descriptions=None):
    """
    Calculate sentiment metrics for a cluster
    NOW INCLUDES DESCRIPTIONS for deeper analysis
    
    Returns:
        consensus: How much the cluster agrees (0-1)
        subjectivity: How emotional/opinionated the content is (0-1)
        polarity: Average sentiment direction (-1 to +1)
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    
    # Combine titles and descriptions for richer analysis
    texts_to_analyze = []
    for i, title in enumerate(titles):
        if pd.isna(title):
            continue
        
        # Combine title + description if available
        combined_text = str(title)
        if descriptions is not None and i < len(descriptions) and not pd.isna(descriptions.iloc[i]):
            desc = str(descriptions.iloc[i])
            if desc and desc != "":
                combined_text += " " + desc
        
        texts_to_analyze.append(combined_text)
    
    # Analyze sentiment
    for text in texts_to_analyze:
        score = analyzer.polarity_scores(text)
        sentiments.append({
            "compound": score["compound"],
            "pos": score["pos"],
            "neg": score["neg"],
            "neu": score["neu"]
        })
    
    if len(sentiments) == 0:
        return 0.0, 0.0, 0.0
    
    # Extract compound scores
    compounds = [s["compound"] for s in sentiments]
    
    # 1. POLARITY: Average sentiment direction
    polarity = np.mean(compounds)
    
    # 2. SUBJECTIVITY: Emotional intensity (how far from neutral)
    #    High subjectivity = strong opinions (either positive or negative)
    subjectivity = np.mean([abs(c) for c in compounds])
    
    # 3. CONSENSUS: Agreement in sentiment direction
    #    Method: What % of videos share the dominant sentiment?
    if polarity >= 0:
        # Dominant sentiment is positive
        agreement_count = sum(1 for c in compounds if c > 0.1)
    else:
        # Dominant sentiment is negative
        agreement_count = sum(1 for c in compounds if c < -0.1)
    
    consensus = agreement_count / len(compounds)
    
    return consensus, subjectivity, polarity


# -----------------------------
# RADICALIZATION DETECTION
# -----------------------------
def classify_cluster(cluster_data, consensus_threshold=0.65, subjectivity_threshold=0.35):
    """
    Determines if a cluster is a "Radicalization Node"
    NOW ANALYZES BOTH TITLES AND DESCRIPTIONS
    
    Criteria (ALL must be true):
    1. Discusses sensitive/polarizing topics
    2. High consensus (everyone agrees)
    3. High subjectivity (strong emotions)
    
    This identifies echo chambers: groups that are both
    opinionated AND in agreement on controversial topics.
    """
    titles = cluster_data["title"].dropna().tolist()
    
    # Get descriptions if available
    descriptions = None
    if "description" in cluster_data.columns:
        descriptions = cluster_data["description"]
        # Also check descriptions for sensitive topics
        desc_texts = descriptions.dropna().tolist()
        combined_texts = titles + desc_texts
    else:
        combined_texts = titles
    
    # Check 1: Topic sensitivity (now includes descriptions)
    has_sensitive, matched_topics = contains_sensitive_topic(combined_texts)
    
    # Check 2 & 3: Sentiment analysis (now includes descriptions)
    consensus, subjectivity, polarity = analyze_cluster_sentiment(titles, descriptions)
    
    # Classification logic (STRICT + SOFT echo chambers)
    is_radicalized = (
        # STRICT: ideological radicalization (paper definition)
        (has_sensitive and
        consensus >= consensus_threshold and
        subjectivity >= subjectivity_threshold)

        # SOFT: strong opinion echo chambers (demo / analysis)
        or
        (consensus >= 0.85 and subjectivity >= 0.6)
    )

    
    label = "Radicalization Node" if is_radicalized else "Benign Cluster"
    
    return {
        "label": label,
        "consensus": consensus,
        "subjectivity": subjectivity,
        "polarity": polarity,
        "has_sensitive_topic": has_sensitive,
        "matched_topics": matched_topics if has_sensitive else []
    }


# -----------------------------
# MAIN VALIDATION FUNCTION
# -----------------------------
def run_validation(input_file=None, output_file=None, 
                   consensus_threshold=0.65, 
                   subjectivity_threshold=0.35):
    """
    Validates semantic clusters and identifies radicalization nodes
    """

    # -------------------------------------------------
    # FIXED PATH RESOLUTION (ONLY CHANGE THAT MATTERS)
    # validator.py is in main/Validator/validator.py
    # parents[2] -> project root
    # -------------------------------------------------
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = BASE_DIR / "output"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # DEFAULT FILES (THIS WAS MISSING)
    # -------------------------------------------------
    if input_file is None:
        input_file = DATA_DIR / "sbert_dataset.csv"
    if output_file is None:
        output_file = DATA_DIR / "validated_clusters.csv"

    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------
    print(f"Loading data from {input_file}...")

    if not Path(input_file).exists():
        raise FileNotFoundError(f"❌ Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    if "cluster" not in df.columns:
        raise ValueError("Input CSV must contain 'cluster' column")

    # -------------------------------------------------
    # DESCRIPTION CHECK
    # -------------------------------------------------
    has_descriptions = "description" in df.columns and df["description"].notna().any()
    if has_descriptions:
        desc_count = df["description"].notna().sum()
        print(f"✓ Found descriptions for {desc_count}/{len(df)} videos")
    else:
        print("⚠️ No descriptions found - using titles only")

    print(f"Found {df['cluster'].nunique()} clusters to validate\n")

    # -------------------------------------------------
    # CLUSTER ANALYSIS (UNCHANGED)
    # -------------------------------------------------
    results = []

    for cluster_id, group in df.groupby("cluster"):
        print(f"Validating Cluster {cluster_id}...")

        classification = classify_cluster(
            group, 
            consensus_threshold, 
            subjectivity_threshold
        )

        result = {
            "cluster_id": cluster_id,
            "size": len(group),
            "label": classification["label"],
            "consensus": round(classification["consensus"], 3),
            "subjectivity": round(classification["subjectivity"], 3),
            "polarity": round(classification["polarity"], 3),
            "sensitive_topics": ", ".join(classification["matched_topics"])
            if classification["matched_topics"] else "None"
        }

        results.append(result)

        status = "⚠️ RADICALIZED" if result["label"] == "Radicalization Node" else "✓ Benign"
        print(
            f"  {status} | "
            f"Consensus: {result['consensus']:.2f} | "
            f"Subjectivity: {result['subjectivity']:.2f}"
        )
        if classification["matched_topics"]:
            print(f"  Topics: {', '.join(classification['matched_topics'][:3])}")
        print()

    # -------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------
    output_df = pd.DataFrame(results)

    radicalized_count = (output_df["label"] == "Radicalization Node").sum()

    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total clusters        : {len(output_df)}")
    print(f"Radicalization nodes  : {radicalized_count}")
    print(f"Benign clusters       : {len(output_df) - radicalized_count}")
    print(f"Average consensus     : {output_df['consensus'].mean():.2f}")
    print(f"Average subjectivity  : {output_df['subjectivity'].mean():.2f}")
    print(f"{'='*60}\n")

    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    return output_df


# -----------------------------
# STANDALONE EXECUTION
# -----------------------------
if __name__ == "__main__":
    # Run with default parameters
    run_validation()
    
    # Optional: Run with custom thresholds
    # run_validation(consensus_threshold=0.7, subjectivity_threshold=0.4)