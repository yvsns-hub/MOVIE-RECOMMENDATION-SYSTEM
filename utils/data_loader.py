"""
Step 1: Data loading and cleaning.
Downloads TMDB 5000 dataset from Kaggle and produces movies_clean.csv.

Usage:
    python utils/data_loader.py

Dataset: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
Files needed in /data/:
    tmdb_5000_movies.csv
    tmdb_5000_credits.csv
"""

import pandas as pd
import numpy as np
import ast
import os
from collections import Counter


def safe_parse(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


def extract_names(val, key="name", limit=None):
    parsed = safe_parse(val)
    names = [item[key] for item in parsed if isinstance(item, dict) and key in item]
    return names[:limit] if limit else names


def extract_director(crew_val):
    for member in safe_parse(crew_val):
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name", "")
    return ""


def extract_top_cast(cast_val, n=5):
    cast = safe_parse(cast_val)
    return [m["name"] for m in cast[:n] if isinstance(m, dict) and "name" in m]


def assign_mood(genres):
    mood_map = {
        "Comedy": "Happy", "Romance": "Happy", "Animation": "Happy",
        "Family": "Happy", "Music": "Happy",
        "Thriller": "Dark", "Horror": "Dark", "Crime": "Dark",
        "Mystery": "Dark", "War": "Dark",
        "Action": "Epic", "Adventure": "Epic", "Science Fiction": "Epic",
        "Fantasy": "Epic",
        "Drama": "Emotional", "History": "Emotional",
        "Documentary": "Informative",
    }
    for g in genres:
        if g in mood_map:
            return mood_map[g]
    return "Mixed"


def clean_and_merge(movies: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicate title column from credits before merging
    credits = credits.rename(columns={"movie_id": "id"})
    if "title" in credits.columns:
        credits = credits.drop(columns=["title"])

    df = movies.merge(credits, on="id", how="left")
    print(f"After merge: {len(df)} rows")

    df["genres_list"]   = df["genres"].apply(lambda x: extract_names(x, "name"))
    df["keywords_list"] = df["keywords"].apply(lambda x: extract_names(x, "name"))
    df["cast_list"]     = df["cast"].apply(extract_top_cast)
    df["director"]      = df["crew"].apply(extract_director)
    df["overview"]      = df["overview"].fillna("").str.strip()
    df["release_year"]  = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
    df["vote_average"]  = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["vote_count"]    = pd.to_numeric(df["vote_count"],   errors="coerce").fillna(0).astype(int)
    df["popularity"]    = pd.to_numeric(df["popularity"],   errors="coerce").fillna(0.0)
    df["budget"]        = pd.to_numeric(df["budget"],       errors="coerce").fillna(0)
    df["revenue"]       = pd.to_numeric(df["revenue"],      errors="coerce").fillna(0)

    before = len(df)
    df = df[df["title"].notna() & (df["overview"] != "")].reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with missing title/overview")

    df["mood"] = df["genres_list"].apply(assign_mood)

    def make_soup(row):
        genres   = " ".join(row["genres_list"]).replace(" ", "")
        keywords = " ".join(row["keywords_list"][:10]).replace(" ", "")
        cast     = " ".join(row["cast_list"]).replace(" ", "")
        director = row["director"].replace(" ", "")
        return f"{genres} {keywords} {cast} {director} {row['overview']}"

    df["soup"] = df.apply(make_soup, axis=1)

    keep = ["id", "title", "overview", "soup", "genres_list", "keywords_list",
            "cast_list", "director", "vote_average", "vote_count", "popularity",
            "release_year", "budget", "revenue", "mood", "runtime"]
    keep = [c for c in keep if c in df.columns]
    df   = df[keep]

    print(f"Final dataset: {len(df)} movies, {len(df.columns)} columns")
    print("\nMood distribution:")
    print(df["mood"].value_counts().to_string())
    print("\nTop genres:")
    all_genres = [g for gl in df["genres_list"] for g in gl]
    for genre, count in Counter(all_genres).most_common(10):
        print(f"  {genre}: {count}")
    return df


def load_raw_data(data_dir="data"):
    movies_path  = os.path.join(data_dir, "tmdb_5000_movies.csv")
    credits_path = os.path.join(data_dir, "tmdb_5000_credits.csv")
    if not os.path.exists(movies_path):
        raise FileNotFoundError(
            f"Missing: {movies_path}\n"
            "Download from: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata\n"
            "Place both CSVs inside the /data/ folder."
        )
    movies  = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    print(f"Movies:  {movies.shape[0]} rows, {movies.shape[1]} columns")
    print(f"Credits: {credits.shape[0]} rows, {credits.shape[1]} columns")
    return movies, credits


def save_clean_data(df, data_dir="data"):
    out = os.path.join(data_dir, "movies_clean.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    print("── Step 1: Data Loading & Cleaning ──\n")
    movies, credits = load_raw_data("data")
    df = clean_and_merge(movies, credits)
    save_clean_data(df, "data")
    print("\nDone. Next: python utils/tfidf_engine.py")
