"""
Filters and Mood Engine — used by the Streamlit app.
"""

import pandas as pd
import os
import ast

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History",
    "Horror", "Music", "Mystery", "Romance", "Science Fiction",
    "Thriller", "War", "Western",
]

MOOD_MAP = {
    "Happy":       ["Comedy", "Romance", "Animation", "Family", "Music"],
    "Dark":        ["Thriller", "Horror", "Crime", "Mystery", "War"],
    "Epic":        ["Action", "Adventure", "Science Fiction", "Fantasy"],
    "Emotional":   ["Drama", "History", "Romance", "Music"],
    "Informative": ["Documentary", "History"],
    "Mixed":       [],
}

MOOD_EMOJI = {
    "Happy": "😄", "Dark": "🌑", "Epic": "⚡",
    "Emotional": "🥹", "Informative": "🎓", "Mixed": "🎲", "Any": "🎬",
}

MOOD_DESCRIPTIONS = {
    "Happy":       "Feel-good, fun, uplifting",
    "Dark":        "Suspense, crime, horror",
    "Epic":        "Action, sci-fi, fantasy",
    "Emotional":   "Drama, romance, moving",
    "Informative": "Documentary, history",
    "Mixed":       "Anything goes",
    "Any":         "No mood filter",
}


def _safe_list(val):
    if isinstance(val, list):
        return val
    try:
        r = ast.literal_eval(str(val))
        return r if isinstance(r, list) else []
    except Exception:
        return []


def load_clean_data(data_dir="data"):
    path = os.path.join(data_dir, "movies_clean.csv")
    df = pd.read_csv(path)
    for col in ["genres_list", "keywords_list", "cast_list"]:
        if col in df.columns:
            df[col] = df[col].apply(_safe_list)
    df["overview"] = df["overview"].fillna("")
    df["director"] = df["director"].fillna("")
    df["mood"]     = df["mood"].fillna("Mixed")
    return df


def filter_by_genre(df, genres):
    if not genres:
        return df
    mask = df["genres_list"].apply(
        lambda gl: any(g in gl for g in genres) if isinstance(gl, list) else False
    )
    return df[mask].reset_index(drop=True)


def filter_by_mood(df, mood):
    if not mood or mood == "Any":
        return df
    return df[df["mood"] == mood].reset_index(drop=True)


def filter_by_rating(df, min_rating, max_rating):
    return df[df["vote_average"].between(min_rating, max_rating)].reset_index(drop=True)


def filter_by_year(df, year_min, year_max):
    valid    = df["release_year"].between(1, 2025)
    in_range = df["release_year"].between(year_min, year_max)
    return df[valid & in_range].reset_index(drop=True)


def get_top_by_mood(df, mood, n=10, min_votes=200, sort_by="vote_average"):
    filtered = filter_by_mood(df, mood)
    filtered = filtered[filtered["vote_count"] >= min_votes]
    filtered = filtered.sort_values(sort_by, ascending=False)
    cols = ["title", "genres_list", "director", "vote_average",
            "release_year", "mood", "overview", "id", "vote_count"]
    cols = [c for c in cols if c in filtered.columns]
    return filtered.head(n)[cols]


if __name__ == "__main__":
    df = load_clean_data("data")
    print(f"Loaded {len(df)} movies.")
    print(f"Genre=Action:   {len(filter_by_genre(df, ['Action']))} movies")
    print(f"Mood=Dark:      {len(filter_by_mood(df, 'Dark'))} movies")
    print(f"Rating 7.5-10:  {len(filter_by_rating(df, 7.5, 10.0))} movies")
    print(f"Year 2000-2017: {len(filter_by_year(df, 2000, 2017))} movies")
    print("\nTop 5 Happy movies:")
    for _, r in get_top_by_mood(df, "Happy", n=5).iterrows():
        print(f"  {r['title']:<40} {r['vote_average']:.1f}")
    print("\nFilters OK.")
