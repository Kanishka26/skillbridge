"""
resume_skill_extractor.py

Standalone module: takes a resume PDF + the ESCO skills CSV, and returns
the list of matched skill names (preferredLabel values).

This module is independent of Kanishka's TransE/DistMult code. It is
meant to be called BEFORE that pipeline — its output (a list of skill
names) becomes the input to the graph model.

Usage:
    from resume_skill_extractor import extract_skills_from_resume

    matched = extract_skills_from_resume(
        pdf_path="sample_resume.pdf",
        csv_path="cs_skills_list.csv",
    )
    print(matched)  # ["Python", "Agile project management", ...]
"""

import re
import pdfplumber
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Loaded once and reused across calls if the module is imported into a
# long-running server (like Kanishka's FastAPI app), so the model isn't
# reloaded on every request.
_model = None


def _get_model():
    global _model
    if _model is None:
        # Small, fast, good general-purpose sentence embedding model.
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def extract_resume_text(pdf_path: str) -> str:
    """Pulls all text out of the resume PDF."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_resume(text: str) -> tuple[list[str], list[str]]:
    """
    Splits resume text into short chunks (roughly one phrase/bullet/
    sentence per chunk) so each chunk can be embedded and compared
    independently. Resume formatting is inconsistent, so this splits
    on newlines, bullet characters at the start of a line, and
    sentence-ending periods — but NOT on hyphens inside words like
    "cross-functional" or "sprint-based".

    Returns a tuple: (sentence_chunks, explicit_skill_items)

    sentence_chunks: full-sentence/bullet chunks, used for semantic
        (embedding) matching against the ESCO CSV.
    explicit_skill_items: items pulled out of "Label: item, item, item"
        lines (e.g. "Tools: GitHub, VS Code, Claude") — these are
        treated as skills the resume explicitly claims, regardless of
        whether they get a strong embedding match against the CSV.
    """
    lines = re.split(r"\n", text)
    pieces = []
    explicit_items = []
    for line in lines:
        line = re.sub(r"^[\s•\u2022\-\*]+", "", line)  # strip leading bullet
        pieces.extend(re.split(r"\.\s+", line))

        # Lines like "Tools: GitHub, VS Code, Claude" — pull out each
        # item as an explicitly-claimed skill.
        if ":" in line and "," in line:
            _, _, items_part = line.partition(":")
            items = [i.strip() for i in items_part.split(",")]
            explicit_items.extend([i for i in items if 1 < len(i) <= 40])

    sentence_chunks = [p.strip() for p in pieces if len(p.strip()) > 8]
    return sentence_chunks, explicit_items


def load_skills(csv_path: str) -> pd.DataFrame:
    """
    Loads the ESCO skills CSV. Expects at least a 'preferredLabel'
    column; 'category' is used for extra context in the output but
    not for matching.
    """
    df = pd.read_csv(csv_path)
    if "preferredLabel" not in df.columns:
        raise ValueError(
            f"Expected a 'preferredLabel' column. Found columns: {list(df.columns)}"
        )
    df = df.dropna(subset=["preferredLabel"]).drop_duplicates(subset=["preferredLabel"])
    return df.reset_index(drop=True)


def extract_skills_from_resume(
    pdf_path: str,
    csv_path: str,
    threshold: float = 0.6,
    top_k_per_chunk: int = 3,
    explicit_match_threshold: float = 0.6,
) -> list[dict]:
    """
    Main entry point. Returns a list of matched skills, each as:
        {"skill": str, "category": str, "score": float, "source": str}

    source is either:
      "esco_match"       - matched semantically against an ESCO skill
      "resume_explicit"  - pulled directly from a labeled skill list in
                            the resume (e.g. "Tools: GitHub, ...") that
                            did NOT have a strong ESCO match. Kept as-is
                            because ESCO doesn't track every modern tool
                            by name (e.g. GitHub, Firebase, Power BI).

    threshold: minimum cosine similarity to count an ESCO skill as
        matched (tune by testing against real resumes).
    top_k_per_chunk: how many candidate ESCO skills to consider per
        resume sentence-chunk before applying the threshold.
    explicit_match_threshold: for items explicitly listed in the resume
        (e.g. "Tools: GitHub"), the minimum score needed to report them
        as an ESCO match instead of a raw resume_explicit entry. Higher
        than `threshold` because here we already know the item is a
        real skill — we're only deciding whether to relabel it under
        an ESCO name or keep the original term.
    """
    model = _get_model()

    # 1. Resume -> text -> (sentence chunks, explicitly-listed items)
    resume_text = extract_resume_text(pdf_path)
    sentence_chunks, explicit_items = chunk_resume(resume_text)
    if not sentence_chunks and not explicit_items:
        return []

    # 2. Load skills
    skills_df = load_skills(csv_path)
    skill_labels = skills_df["preferredLabel"].tolist()
    label_to_category = dict(zip(skills_df["preferredLabel"], skills_df["category"]))

    skill_embeddings = model.encode(skill_labels, show_progress_bar=False)

    # 3. Semantic match: sentence chunks against ESCO skills
    best_score_per_skill: dict[str, float] = {}

    if sentence_chunks:
        chunk_embeddings = model.encode(sentence_chunks, show_progress_bar=False)
        similarity_matrix = cosine_similarity(chunk_embeddings, skill_embeddings)

        for chunk_idx in range(len(sentence_chunks)):
            row = similarity_matrix[chunk_idx]
            top_indices = row.argsort()[::-1][:top_k_per_chunk]

            for skill_idx in top_indices:
                score = float(row[skill_idx])
                if score < threshold:
                    continue
                skill_name = skill_labels[skill_idx]
                if skill_name not in best_score_per_skill or score > best_score_per_skill[skill_name]:
                    best_score_per_skill[skill_name] = score

    results = [
        {
            "skill": skill,
            "category": label_to_category.get(skill, "General"),
            "score": round(score, 3),
            "source": "esco_match",
        }
        for skill, score in best_score_per_skill.items()
    ]

    # 4. Hybrid step: explicitly-listed items (e.g. "Tools: GitHub, ...")
    # Each one either strengthens/confirms an ESCO match, or — if ESCO
    # has no good match for it — gets included as-is. This is what
    # catches modern tool names (GitHub, Firebase, Power BI, MongoDB)
    # that ESCO doesn't track individually.
    already_included = {r["skill"].lower() for r in results}

    if explicit_items:
        explicit_embeddings = model.encode(explicit_items, show_progress_bar=False)
        explicit_sim = cosine_similarity(explicit_embeddings, skill_embeddings)

        for i, item in enumerate(explicit_items):
            row = explicit_sim[i]
            best_idx = row.argmax()
            best_score = float(row[best_idx])
            best_label = skill_labels[best_idx]

            if best_score >= explicit_match_threshold:
                # Good ESCO match exists — already captured (or now added)
                # under its ESCO name; skip adding the raw term separately.
                if best_label.lower() not in already_included:
                    results.append({
                        "skill": best_label,
                        "category": label_to_category.get(best_label, "General"),
                        "score": round(best_score, 3),
                        "source": "esco_match",
                    })
                    already_included.add(best_label.lower())
            else:
                # No good ESCO match — keep the resume's own term as-is.
                #
                # IMPORTANT: "score" here is set to 1.0 (full confidence),
                # NOT best_score. best_score measures similarity to ESCO's
                # *closest* label, which is naturally low for modern tools
                # ESCO doesn't track (GitHub, Firebase, etc). That's an
                # ESCO-coverage gap, not a sign the skill is uncertain —
                # the resume explicitly listed it, so confidence is high.
                # The raw ESCO-similarity is kept separately as
                # "esco_similarity" for debugging only; it should never
                # be shown to end users as a confidence/match score.
                if item.lower() not in already_included:
                    results.append({
                        "skill": item,
                        "category": "Resume-listed (no ESCO match)",
                        "score": 1.0,
                        "esco_similarity": round(best_score, 3),  # debug only
                        "source": "resume_explicit",
                    })
                    already_included.add(item.lower())

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


if __name__ == "__main__":
    # Quick manual test:
    #   python resume_skill_extractor.py path/to/resume.pdf path/to/cs_skills_list.csv
    import sys

    if len(sys.argv) != 3:
        print("Usage: python resume_skill_extractor.py <resume.pdf> <skills.csv>")
        sys.exit(1)


    matches = extract_skills_from_resume(sys.argv[1], sys.argv[2])
    print(f"\nFound {len(matches)} matching skills:\n")
    for m in matches:
        if m["source"] == "esco_match":
            print(f"  [ESCO]           {m['skill']:50s} [{m['category']:30s}] score={m['score']}")
        else:
            print(
                f"  [RESUME-DIRECT]  {m['skill']:50s} [{m['category']:30s}] "
                f"confidence={m['score']} (esco_similarity={m['esco_similarity']}, for debug only)"
            )