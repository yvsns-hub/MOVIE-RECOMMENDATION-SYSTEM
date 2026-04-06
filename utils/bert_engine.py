"""
Step 3: BERT Embedding Engine.
Encodes movie overviews using sentence-transformers and saves embeddings.npy.

Usage:
    python utils/bert_engine.py

Downloads ~90MB model on first run. Encoding takes ~3-5 minutes.
"""

import pandas as pd
import numpy as np
import os
import ast
import sys
import time
from difflib import get_close_matches


def load_clean_data(data_dir="data"):
    path = os.path.join(data_dir, "movies_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run utils/data_loader.py first.")
    df = pd.read_csv(path)
    for col in ["genres_list", "keywords_list", "cast_list"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    df["overview"] = df["overview"].fillna("")
    df["director"] = df["director"].fillna("")
    df["mood"]     = df["mood"].fillna("Mixed")
    print(f"Loaded {len(df)} movies.")
    return df


def build_bert_text(df):
    texts = []
    for _, row in df.iterrows():
        genres   = ", ".join(row["genres_list"]) if isinstance(row["genres_list"], list) else ""
        director = row.get("director", "")
        overview = row.get("overview", "")
        text = overview
        if genres:
            text += f" Genre: {genres}."
        if director:
            text += f" Directed by {director}."
        texts.append(text.strip())
    return texts


def encode_movies(texts, batch_size=64):
    print("Loading sentence-transformer model (downloads ~90MB on first run)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Encoding {len(texts)} movies...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  Shape: {embeddings.shape} | Time: {time.time()-t0:.1f}s")
    return embeddings


def save_embeddings(embeddings, data_dir="data"):
    path = os.path.join(data_dir, "embeddings.npy")
    np.save(path, embeddings)
    print(f"  Saved → {path} ({os.path.getsize(path)/1024/1024:.1f} MB)")


def get_bert_recommendations(title, df, embeddings, n=10,
                             genre_filter=None, mood_filter=None,
                             min_rating=0.0, max_rating=10.0,
                             year_min=1900, year_max=2025):
    title_lower  = title.strip().lower()
    title_to_idx = pd.Series(df.index, index=df["title"].str.lower()).to_dict()

    if title_lower in title_to_idx:
        idx = title_to_idx[title_lower]
    else:
        matches = get_close_matches(title_lower, title_to_idx.keys(), n=1, cutoff=0.5)
        if not matches:
            return pd.DataFrame(), f"No match found for '{title}'"
        idx = title_to_idx[matches[0]]

    query_vec  = embeddings[idx]
    sim_scores = embeddings @ query_vec
    sim_scores[idx] = -1
    ranked_idx = np.argsort(sim_scores)[::-1]

    candidates = []
    for i in ranked_idx:
        row   = df.iloc[i]
        score = float(sim_scores[i])
        if genre_filter:
            genres = row["genres_list"] if isinstance(row["genres_list"], list) else []
            if not any(g in genres for g in genre_filter):
                continue
        if mood_filter and mood_filter != "Any":
            if row.get("mood") != mood_filter:
                continue
        rating = float(row["vote_average"]) if pd.notna(row["vote_average"]) else 0.0
        if not (min_rating <= rating <= max_rating):
            continue
        year = int(row["release_year"]) if pd.notna(row["release_year"]) and row["release_year"] > 0 else 0
        if year > 0 and not (year_min <= year <= year_max):
            continue
        candidates.append({
            "title":        row["title"],
            "genres":       ", ".join(row["genres_list"]) if isinstance(row["genres_list"], list) else "",
            "genres_list":  row["genres_list"] if isinstance(row["genres_list"], list) else [],
            "director":     row.get("director", ""),
            "vote_average": rating,
            "release_year": year if year > 0 else "N/A",
            "mood":         row.get("mood", ""),
            "overview":     row.get("overview", ""),
            "poster_path":  row.get("poster_path", ""),
            "id":           row.get("id", ""),
            "similarity":   round(score, 4),
        })
        if len(candidates) >= n:
            break
    return pd.DataFrame(candidates), None


if __name__ == "__main__":
    print("── Step 3: BERT Embedding Engine ──\n")
    df    = load_clean_data("data")
    texts = build_bert_text(df)
    print(f"\nSample: {texts[0][:100]}...\n")

    embeddings = encode_movies(texts)
    print("\nSaving...")
    save_embeddings(embeddings)

    print("\n── Quick Test ──")
    results, err = get_bert_recommendations("Inception", df, embeddings, n=5)
    if not err:
        print("\nBERT → 'Inception' top 5:")
        for _, r in results.iterrows():
            print(f"  {r['title']:<40} {r['vote_average']:.1f}  sim:{r['similarity']:.3f}")

    print("\nDone. Next: streamlit run app.py")
