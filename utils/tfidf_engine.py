"""
Step 2: TF-IDF Recommendation Engine.
Builds cosine similarity matrix and saves to models/sim_tfidf.pkl.

Usage:
    python utils/tfidf_engine.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import ast
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches


def load_clean_data(data_dir="data"):
    path = os.path.join(data_dir, "movies_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run utils/data_loader.py first.")
    df = pd.read_csv(path)
    for col in ["genres_list", "keywords_list", "cast_list"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    df["soup"]     = df["soup"].fillna("")
    df["overview"] = df["overview"].fillna("")
    df["director"] = df["director"].fillna("")
    print(f"Loaded {len(df)} movies.")
    return df


def build_tfidf_matrix(df):
    print("Building TF-IDF matrix...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(df["soup"])
    print(f"  Shape: {tfidf_matrix.shape} | Time: {time.time()-t0:.1f}s")
    return tfidf_matrix, vectorizer


def build_cosine_similarity(tfidf_matrix):
    print("Computing cosine similarity...")
    t0 = time.time()
    sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"  Shape: {sim.shape} | Time: {time.time()-t0:.1f}s")
    return sim


def save_model(sim_matrix, vectorizer, df, models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    sim_path = os.path.join(models_dir, "sim_tfidf.pkl")
    with open(sim_path, "wb") as f:
        pickle.dump(sim_matrix, f, protocol=4)
    print(f"  Saved sim matrix  → {sim_path} ({os.path.getsize(sim_path)/1024/1024:.1f} MB)")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f, protocol=4)
    print(f"  Saved vectorizer  → {vec_path}")
    idx_path = os.path.join(models_dir, "title_index.pkl")
    title_to_idx = pd.Series(df.index, index=df["title"].str.lower()).to_dict()
    with open(idx_path, "wb") as f:
        pickle.dump(title_to_idx, f, protocol=4)
    print(f"  Saved title index → {idx_path}")


def get_recommendations(title, df, sim_matrix, n=10,
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

    sim_scores = sorted(enumerate(sim_matrix[idx]), key=lambda x: x[1], reverse=True)[1:]
    candidates = []
    for i, score in sim_scores:
        row = df.iloc[i]
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
    print("── Step 2: TF-IDF Recommendation Engine ──\n")
    df = load_clean_data("data")
    tfidf_mat, vectorizer = build_tfidf_matrix(df)
    sim_matrix = build_cosine_similarity(tfidf_mat)
    print("\nSaving model files...")
    save_model(sim_matrix, vectorizer, df)

    print("\n── Quick Test ──")
    for title in ["Inception", "The Dark Knight", "Toy Story"]:
        results, err = get_recommendations(title, df, sim_matrix, n=3)
        if not err:
            print(f"\n'{title}' →")
            for _, r in results.iterrows():
                print(f"  {r['title']:<40} {r['vote_average']:.1f}  sim:{r['similarity']:.3f}")

    print("\nDone. Next: python utils/bert_engine.py")
