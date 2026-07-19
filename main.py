print("Starting...")
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import numpy as np
import pickle
import pandas as pd
from thefuzz import process
import uvicorn
import tempfile
import os
print("Imports done")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract-skills")
async def extract_skills(file: UploadFile = File(...)):
    # Lazy import: sentence_transformers is heavy and was contributing to
    # startup OOM on Render's free tier when imported at module level
    # alongside torch/torch_geometric. Only load it when this endpoint is
    # actually hit, not every time the server starts.
    import resume_skill_extractor
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        matches = resume_skill_extractor.extract_skills_from_resume(
            pdf_path=tmp_path,
            csv_path='outputs/cs_skills_list.csv'
        )
        skill_names = [m['skill'] for m in matches if m['source'] == 'esco_match']
        return {"success": True, "extracted_skills": skill_names}
    finally:
        os.unlink(tmp_path)

# ── Load heterodata — MUST be the same pickle the R-GCN checkpoint was
# trained against (the rebuild script's output), not the old Week 1 pickle.
# Mismatched entity indices here would silently return wrong predictions.
print("Loading heterodata (rebuilt version, matching R-GCN training)...")
with open('outputs/heterodata_rebuilt.pkl', 'rb') as f:
    saved = pickle.load(f)

all_skill_nodes = saved['all_skill_nodes']
all_occ_nodes   = saved['all_occ_nodes']
skill_idx       = saved['skill_idx']
occ_idx         = saved['occ_idx']
skill_label_map = saved['skill_label_map']   # uri -> label
occ_label_map   = saved['occ_label_map']     # uri -> label

num_skills    = len(all_skill_nodes)
num_occs      = len(all_occ_nodes)
num_entities  = num_skills + num_occs
occ_offset    = num_skills
num_relations = 4
EMBEDDING_DIM = 100
NUM_BASES     = 2   # must match the checkpoint's training config

# uri -> label lookup used for fuzzy matching, and index -> label for output
uri_to_label = {**skill_label_map, **occ_label_map}
skill_label_to_idx = {skill_label_map.get(uri, uri): i for i, uri in enumerate(all_skill_nodes)}
occ_label_to_idx   = {occ_label_map.get(uri, uri): i for i, uri in enumerate(all_occ_nodes)}

print(f"Skills: {num_skills}  Occupations: {num_occs}")

# ── Skills CSV for category lookup (unchanged from original) ────
skills_df = pd.read_csv('outputs/cs_skills_list.csv', index_col=0)
col = 'preferredLabel' if 'preferredLabel' in skills_df.columns else 'skill'
cat_map = dict(zip(skills_df[col].str.lower(), skills_df['category']))

# ── R-GCN model definition — MUST match the winning architecture exactly:
# basis decomposition (num_bases=2) + DistMult-style scoring.
# This is NOT the original R-GCN class -- it's the tuned variant that beat
# both TransE and DistMult after basis decomposition + multi-negative
# training (see rgcn_round3_fixed_loss.py).
print("Loading R-GCN model...")

class RGCN_Combined(nn.Module):
    def __init__(self, n_ent, n_rel, dim, num_bases):
        super().__init__()
        self.entity_embeddings = nn.Embedding(n_ent, dim)
        self.conv1 = RGCNConv(dim, dim, n_rel, num_bases=num_bases)
        self.conv2 = RGCNConv(dim, dim, n_rel, num_bases=num_bases)
        self.relation_embeddings = nn.Embedding(n_rel, dim)
    def encode(self, edge_index, edge_type):
        x = F.relu(self.conv1(self.entity_embeddings.weight, edge_index, edge_type))
        return self.conv2(x, edge_index, edge_type)
    def score(self, h_emb, r_emb, t_emb):
        return torch.sum(h_emb * r_emb * t_emb, dim=1)  # DistMult-style scoring

checkpoint = torch.load('outputs/rgcn_final_winning.pt', map_location='cpu')
model = RGCN_Combined(checkpoint['num_entities'], checkpoint['num_relations'],
                       checkpoint['embedding_dim'], checkpoint['num_bases'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Precompute entity embeddings ONCE at startup using the saved training
# graph -- NOT per-request. This is the same encode()-per-mini-batch bug
# from the very first message in this project; don't reintroduce it here.
train_edge_index = checkpoint['train_edge_index']
train_edge_type  = checkpoint['train_edge_type']
with torch.no_grad():
    entity_embs = model.encode(train_edge_index, train_edge_type)
    r_embs      = model.relation_embeddings.weight

print(f"Model loaded. Architecture: {checkpoint.get('architecture', 'unknown')}")
print(f"Saved test performance: {checkpoint.get('test_results', {})}")

# ── Match helpers (unchanged logic, now over label maps built fresh) ────
def match_skill(skill_name: str, threshold: int = 70):
    result = process.extractOne(skill_name.lower(), list(skill_label_to_idx.keys()))
    if result and result[1] >= threshold:
        matched_label = result[0]
        return matched_label, skill_label_to_idx[matched_label]
    return None, None

def match_occupation(occ_name: str):
    result = process.extractOne(occ_name.lower(), list(occ_label_to_idx.keys()))
    if result and result[1] >= 60:
        return occ_label_to_idx[result[0]]
    return None

# ── Request / Response models (unchanged) ────────────────────────
class PredictRequest(BaseModel):
    extracted_skills:  List[str]
    target_occupation: str
    class Config:
        extra = "allow"

class MissingSkill(BaseModel):
    skill:             str
    category:          str
    confidence:        float
    reason:            str
    learning_priority: str

class PredictResponse(BaseModel):
    success:           bool
    target_occupation: str
    matched_skills:    List[str]
    missing_skills:    List[MissingSkill]
    readiness_score:   float
    learning_roadmap:  List[dict]

# ── Predict endpoint — now scores with R-GCN embeddings ──────────
@app.post("/predict")
def predict(request: PredictRequest):
    input_skills = []
    for skill_list in (
        getattr(request, "selected_skills", []),
        getattr(request, "resume_skills", []),
        request.extracted_skills,
    ):
        for skill in skill_list:
            cleaned_skill = skill.strip()
            if cleaned_skill:
                input_skills.append(cleaned_skill)
    input_skills = list(dict.fromkeys(input_skills))

    matched_skills     = []
    matched_skill_idxs = []
    for skill in input_skills:
        label, idx = match_skill(skill)
        if idx is not None:
            matched_skills.append(label)
            matched_skill_idxs.append(idx)

    occ_local_idx = match_occupation(request.target_occupation)
    if occ_local_idx is None:
        return {"success": False, "error": f"Occupation '{request.target_occupation}' not found"}

    # global index: occupations offset by num_skills, matching training convention
    occ_global_idx = occ_local_idx + occ_offset
    occ_tensor = torch.tensor([occ_global_idx], dtype=torch.long)
    r_tensor   = torch.tensor([0], dtype=torch.long)  # relation 0 = required_for

    all_skill_tensor = torch.arange(0, num_skills, dtype=torch.long)
    occ_exp = occ_tensor.expand(num_skills)
    r_exp   = r_tensor.expand(num_skills)

    with torch.no_grad():
        scores = model.score(entity_embs[all_skill_tensor], r_embs[r_exp], entity_embs[occ_exp])
    scores_np = scores.numpy()

    known_set  = set(matched_skill_idxs)
    candidates = [(i, float(scores_np[i])) for i in range(num_skills) if i not in known_set]
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_missing = candidates[:10]

    missing_skills = []
    for skill_i, score in top_missing:
        uri   = all_skill_nodes[skill_i]
        label = uri_to_label.get(uri, f'skill_{skill_i}')
        cat = cat_map.get(label.lower(), 'Uncategorized')
        if not isinstance(cat, str) or str(cat) == 'nan':
            cat = 'Uncategorized'
        # NOTE: DistMult-style scores are unbounded dot products, not
        # negative distances like TransE -- the old (score+2)/4 normalization
        # doesn't map to [0,1] correctly for this scoring function. Using a
        # sigmoid instead; recalibrate against your actual score distribution
        # (min/max over a validation batch) before trusting these confidence
        # values in the demo.
        conf = float(torch.sigmoid(torch.tensor(score)).item())

        missing_skills.append(MissingSkill(
            skill=label,
            category=cat,
            confidence=round(conf, 3),
            reason="Coming soon",
            learning_priority="TBD"
        ))

    readiness = round(len(matched_skills) / max(len(matched_skills) + len(top_missing), 1), 2)

    return PredictResponse(
        success=True,
        target_occupation=request.target_occupation,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        readiness_score=readiness,
        learning_roadmap=[]
    )

@app.get("/health")
def health():
    return {"status": "ok", "model": "R-GCN (basis=2, DistMult-scoring, multi-neg trained)"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)