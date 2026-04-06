"""
TMDB Live API Utility.
Handles poster fetching, live recommendations, reviews, trending.
Requires TMDB_API_KEY in .env file.
Get free key at: https://www.themoviedb.org/settings/api
"""

import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY  = os.getenv("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.tmdb.org/3"
POSTER_BASE   = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER   = "https://placehold.co/300x450/1a1a1a/555?text=No+Poster"

_cache = {}


def _get(endpoint, params=None):
    if not TMDB_API_KEY:
        return None
    params = params or {}
    cache_key = endpoint + str(sorted(params.items()))
    if cache_key in _cache:
        return _cache[cache_key]
    params["api_key"] = TMDB_API_KEY
    try:
        resp = requests.get(f"{TMDB_BASE_URL}/{endpoint}", params=params, timeout=6)
        if resp.status_code == 429:
            time.sleep(2)
            resp = requests.get(f"{TMDB_BASE_URL}/{endpoint}", params=params, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            _cache[cache_key] = data
            return data
    except requests.exceptions.RequestException:
        pass
    return None


def get_poster_url(poster_path):
    if poster_path and str(poster_path) not in ["nan", "None", ""]:
        return f"{POSTER_BASE}{poster_path}"
    return PLACEHOLDER


def search_movie(title):
    data = _get("search/movie", {"query": title, "language": "en-US", "page": 1})
    if data and data.get("results"):
        return data["results"][0]
    return None


def get_live_recommendations(tmdb_id, n=10):
    results = []
    for page in [1, 2]:
        data = _get(f"movie/{tmdb_id}/recommendations", {"page": page})
        if data:
            results.extend(data.get("results", []))
        if len(results) >= n:
            break
    if len(results) < n:
        for page in [1, 2]:
            data = _get(f"movie/{tmdb_id}/similar", {"page": page})
            if data:
                existing_ids = {r["id"] for r in results}
                for m in data.get("results", []):
                    if m["id"] not in existing_ids:
                        results.append(m)
            if len(results) >= n:
                break

    normalized = []
    for m in results[:n]:
        normalized.append({
            "title":        m.get("title", ""),
            "genres":       "",
            "genres_list":  [],
            "director":     "",
            "vote_average": m.get("vote_average", 0.0),
            "release_year": m.get("release_date", "")[:4] if m.get("release_date") else "N/A",
            "mood":         "",
            "overview":     m.get("overview", ""),
            "poster_path":  m.get("poster_path", ""),
            "id":           m.get("id", ""),
            "similarity":   0,
            "source":       "TMDB Live API",
        })
    return normalized


def get_movie_details(tmdb_id):
    data = _get(f"movie/{tmdb_id}", {"append_to_response": "credits,videos"})
    if not data:
        return None
    genres   = [g["name"] for g in data.get("genres", [])]
    director = ""
    for crew in data.get("credits", {}).get("crew", []):
        if crew.get("job") == "Director":
            director = crew.get("name", "")
            break
    cast = [c["name"] for c in data.get("credits", {}).get("cast", [])[:5]]
    details = {
        "id":           data.get("id"),
        "title":        data.get("title", ""),
        "overview":     data.get("overview", ""),
        "genres":       ", ".join(genres),
        "genres_list":  genres,
        "director":     director,
        "cast":         cast,
        "vote_average": data.get("vote_average", 0.0),
        "release_year": data.get("release_date", "")[:4] if data.get("release_date") else "N/A",
        "runtime":      data.get("runtime", 0),
        "tagline":      data.get("tagline", ""),
        "poster_path":  data.get("poster_path", ""),
        "poster_url":   get_poster_url(data.get("poster_path", "")),
        "source":       "TMDB Live API",
    }
    
    # Extract trailer if available
    for video in data.get("videos", {}).get("results", []):
        if video.get("site") == "YouTube" and video.get("type") == "Trailer":
            details["trailer_url"] = f"https://www.youtube.com/watch?v={video['key']}"
            break

    return details


def get_movie_reviews(tmdb_id, max_reviews=5):
    data = _get(f"movie/{tmdb_id}/reviews", {"page": 1})
    if not data:
        return []
    reviews = []
    for r in data.get("results", [])[:max_reviews]:
        rating = r.get("author_details", {}).get("rating")
        reviews.append({
            "author":  r.get("author", "Anonymous"),
            "rating":  float(rating) if rating else None,
            "content": r.get("content", "")[:500],
            "date":    r.get("created_at", "")[:10],
        })
    return reviews


def get_trending(window="week", n=10):
    data = _get(f"trending/movie/{window}")
    if not data:
        return []
    results = []
    for m in data.get("results", [])[:n]:
        results.append({
            "title":        m.get("title", ""),
            "overview":     m.get("overview", ""),
            "vote_average": m.get("vote_average", 0.0),
            "release_year": m.get("release_date", "")[:4] if m.get("release_date") else "N/A",
            "poster_path":  m.get("poster_path", ""),
            "id":           m.get("id", ""),
            "genres": "", "mood": "", "source": "TMDB Live API",
        })
    return results


def get_now_playing(n=10):
    data = _get("movie/now_playing", {"page": 1})
    if not data:
        return []
    results = []
    for m in data.get("results", [])[:n]:
        results.append({
            "title":        m.get("title", ""),
            "overview":     m.get("overview", ""),
            "vote_average": m.get("vote_average", 0.0),
            "release_year": m.get("release_date", "")[:4] if m.get("release_date") else "N/A",
            "poster_path":  m.get("poster_path", ""),
            "id":           m.get("id", ""),
            "genres": "", "mood": "", "source": "TMDB Live API",
        })
    return results


def get_full_movie_profile(title):
    movie = search_movie(title)
    if not movie:
        return None
    tmdb_id = movie["id"]
    details = get_movie_details(tmdb_id)
    reviews = get_movie_reviews(tmdb_id, max_reviews=3)
    if not details:
        return None
    details["reviews"]      = reviews
    details["review_count"] = len(reviews)
    rated = [r["rating"] for r in reviews if r.get("rating")]
    details["avg_review_rating"] = round(sum(rated) / len(rated), 1) if rated else None
    return details
