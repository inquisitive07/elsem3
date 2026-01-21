import subprocess
import sys
from pathlib import Path

# -------------------------------------------------
# Project root
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

def require_file(rel_path: str) -> Path:
    path = BASE_DIR / rel_path
    if not path.exists():
        raise FileNotFoundError(f"❌ Required file not found: {path}")
    return path

print("=" * 60)
print("🚀 YOUTUBE ECHO CHAMBER ANALYSIS PIPELINE")
print("=" * 60)

# -------------------------------------------------
# Phase 1: Scraper
# -------------------------------------------------
print("\n▶ Phase 1: Running scraper (collecting videos + descriptions)...")
scraper = require_file("main/collection/scraper.py")
subprocess.run(
    [sys.executable, scraper],
    check=True
)
print("✓ Scraper complete - CSV saved to main/data/recommendation_walk.csv")

# -------------------------------------------------
# Phase 2: SBERT Analyzer  ✅ FIXED
# -------------------------------------------------
print("\n▶ Phase 2: Running SBERT clustering (analyzing semantic similarity)...")
analyzer = require_file("analyzer.py")
subprocess.run(
    [sys.executable, analyzer],
    check=True
)
print("✓ SBERT clustering complete - embeddings and clusters generated")

# -------------------------------------------------
# Phase 3: Validator
# -------------------------------------------------
print("\n▶ Phase 3: Running semantic validator (quality control)...")
validator = require_file("main/Validator/validator.py")

subprocess.run(
    [sys.executable, validator],
    check=True
)
print("✓ Validation complete - clusters verified")

# -------------------------------------------------
# Phase 4: ECI Analyzer
# -------------------------------------------------
print("\n▶ Phase 4: Running ECI risk analysis (echo chamber detection)...")
eci = require_file("eci_analyzer.py")
subprocess.run(
    [sys.executable, eci],
    check=True
)
print("✓ ECI analysis complete - risk scores calculated")

print("\n" + "=" * 60)
print("✅ PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)
