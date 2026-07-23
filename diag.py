"""
Run this locally (not on Render) to diagnose the category lookup mismatch.
Needs outputs/heterodata_rebuilt.pkl and outputs/cs_skills_list.csv.
"""
import pickle
import pandas as pd

with open('outputs/heterodata_rebuilt.pkl', 'rb') as f:
    saved = pickle.load(f)
skill_label_map = saved['skill_label_map']  # uri -> label, used by main.py

skills_df = pd.read_csv('outputs/cs_skills_list.csv', index_col=0)
col = 'preferredLabel' if 'preferredLabel' in skills_df.columns else 'skill'
print(f"cs_skills_list.csv columns: {list(skills_df.columns)}")
print(f"Using column '{col}' for skill names, 'category' for categories")
print(f"cs_skills_list.csv row count: {len(skills_df)}")
print(f"Sample rows:")
print(skills_df[[col, 'category']].head(10).to_string())

graph_labels_lower = set(v.lower() for v in skill_label_map.values())
csv_labels_lower = set(skills_df[col].str.lower())

overlap = graph_labels_lower & csv_labels_lower
missing = graph_labels_lower - csv_labels_lower

print(f"\nGraph skill labels: {len(graph_labels_lower)}")
print(f"CSV skill labels: {len(csv_labels_lower)}")
print(f"Overlap (exact, case-insensitive match): {len(overlap)}")
print(f"Overlap %: {100*len(overlap)/len(graph_labels_lower):.1f}%")

print(f"\n--- Sample of graph labels with NO exact match in CSV (first 15) ---")
for label in list(missing)[:15]:
    print(f"  '{label}'")

# Check for near-misses: does a fuzzy/substring match exist even if exact
# doesn't? This tells us if it's a formatting difference (extra
# parentheses, different casing conventions, etc.) vs. genuinely
# different skill sets.
print(f"\n--- Checking for near-misses (substring match) on 5 missing labels ---")
csv_labels_list = list(csv_labels_lower)
for label in list(missing)[:5]:
    # strip parenthetical suffixes like "(computer programming)" for the check
    base = label.split('(')[0].strip()
    close_matches = [c for c in csv_labels_list if base in c or c in base]
    print(f"  '{label}' -> base='{base}' -> near-matches in CSV: {close_matches[:3]}")