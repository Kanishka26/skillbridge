"""
Run this AFTER starting main_rgcn.py locally (python main_rgcn.py, or
uvicorn main_rgcn:app --reload), in a separate terminal/script.

Checks two things flagged as risks before the Thursday integration test:
1. Does cs_skills_list.csv still align with the skills in heterodata_rebuilt.pkl?
2. Does the confidence score (sigmoid over DistMult-style scores) actually
   spread out, or does it saturate near 0/1 for everything?
"""
import requests
import pickle
import pandas as pd

BASE_URL = "http://localhost:8000"

# ══════════════════════════════════════════════════════════════
# Check 1 — skill list alignment
# ══════════════════════════════════════════════════════════════
print("=== Check 1: cs_skills_list.csv vs heterodata_rebuilt.pkl alignment ===")

with open('outputs/heterodata_rebuilt.pkl', 'rb') as f:
    saved = pickle.load(f)
skill_label_map = saved['skill_label_map']
graph_skill_labels = set(v.lower() for v in skill_label_map.values())

skills_df = pd.read_csv('outputs/cs_skills_list.csv', index_col=0)
col = 'preferredLabel' if 'preferredLabel' in skills_df.columns else 'skill'
csv_skill_labels = set(skills_df[col].str.lower())

overlap = graph_skill_labels & csv_skill_labels
only_in_graph = graph_skill_labels - csv_skill_labels
only_in_csv = csv_skill_labels - graph_skill_labels

print(f"Skills in graph (heterodata_rebuilt.pkl): {len(graph_skill_labels)}")
print(f"Skills in cs_skills_list.csv: {len(csv_skill_labels)}")
print(f"Overlap: {len(overlap)}")
print(f"In graph but NOT in CSV (category lookup will show 'Uncategorized' for these): {len(only_in_graph)}")
print(f"In CSV but NOT in graph (irrelevant, but suggests stale file): {len(only_in_csv)}")

overlap_pct = len(overlap) / len(graph_skill_labels) * 100 if graph_skill_labels else 0
print(f"\nOverlap: {overlap_pct:.1f}% of graph skills have a category match in the CSV.")
if overlap_pct < 80:
    print("WARNING: low overlap. cs_skills_list.csv is likely stale relative to the "
          "rebuilt graph. Missing skills will show category='Uncategorized' in the "
          "API response, which may look broken in Nriti's UI even though predictions "
          "themselves are correct.")
else:
    print("Overlap looks reasonable.")

# ══════════════════════════════════════════════════════════════
# Check 2 — confidence score spread
# ══════════════════════════════════════════════════════════════
print("\n=== Check 2: /predict confidence score spread ===")

test_payload = {
    "extracted_skills": ["Python", "SQL", "Git"],
    "target_occupation": "software developer"
}

try:
    resp = requests.post(f"{BASE_URL}/predict", json=test_payload, timeout=10)
    resp.raise_for_status()
    result = resp.json()
except Exception as e:
    print(f"Request failed: {e}")
    print("Is main_rgcn.py actually running on localhost:8000? Start it first.")
    raise SystemExit(1)

if not result.get("success", False):
    print(f"API returned success=False: {result}")
    raise SystemExit(1)

confidences = [m["confidence"] for m in result["missing_skills"]]
print(f"Matched skills: {result['matched_skills']}")
print(f"Missing skills returned: {len(result['missing_skills'])}")
print(f"Confidence scores: {confidences}")
print(f"  min: {min(confidences):.3f}  max: {max(confidences):.3f}  "
      f"spread: {max(confidences) - min(confidences):.3f}")

if max(confidences) - min(confidences) < 0.05:
    print("\nWARNING: confidence scores are nearly identical across all predictions. "
          "The sigmoid calibration is likely saturated (all scores mapping to ~0 or ~1). "
          "Every skill will look equally (un)confident in the UI, which defeats the "
          "purpose of ranking. Needs recalibration against the model's actual score "
          "distribution (pull min/max scores from a validation batch and rescale "
          "explicitly, rather than using raw sigmoid).")
else:
    print("\nConfidence scores show reasonable spread.")

print("\n=== Full response for manual inspection ===")
print(result)