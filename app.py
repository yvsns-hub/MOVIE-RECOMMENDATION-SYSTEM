"""
CineMatch — Movie Recommender System
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.filters import (
    ALL_GENRES, MOOD_MAP, MOOD_EMOJI, MOOD_DESCRIPTIONS,
    get_top_by_mood, filter_by_mood,
)
from utils.chatbot import render_chatbot_widget
from utils.tmdb_api import (
    get_poster_url, search_movie, get_live_recommendations,
    get_movie_details, get_full_movie_profile, get_trending,
    get_now_playing, TMDB_API_KEY, POSTER_BASE, PLACEHOLDER,
)
from utils.youtube import get_trailer_url

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e0e0e; }
    section[data-testid="stSidebar"] { background-color: #161616; }
    .movie-card {
        background:#1a1a1a; border-radius:12px; padding:12px;
        margin-bottom:8px; border:1px solid #2a2a2a;
    }
    .movie-title { font-size:13px; font-weight:600; color:#fff; margin:8px 0 4px; line-height:1.3; }
    .movie-meta  { font-size:11px; color:#888; margin:2px 0; }
    .badge-red  { display:inline-block; background:#e50914; color:#fff; border-radius:4px; padding:2px 6px; font-size:11px; font-weight:600; }
    .badge-dark { display:inline-block; background:#2a2a2a; color:#ccc; border-radius:4px; padding:2px 6px; font-size:11px; margin-left:3px; }
    .badge-blue { display:inline-block; background:#0f4c75; color:#90caf9; border-radius:4px; padding:2px 6px; font-size:11px; margin-left:3px; }
    .section-hdr { font-size:18px; font-weight:700; color:#fff; margin:20px 0 12px; padding-bottom:6px; border-bottom:2px solid #e50914; }
    .info-box { background:#1a1a2e; border-left:3px solid #0f4c75; padding:10px 14px; border-radius:0 8px 8px 0; margin:8px 0; font-size:13px; color:#90caf9; }
    .warn-box { background:#2a1a00; border-left:3px solid #e65c00; padding:10px 14px; border-radius:0 8px 8px 0; margin:8px 0; font-size:13px; color:#ffb347; }
    .match-bar-bg { background:#2a2a2a; border-radius:2px; height:3px; margin-top:4px; }
    .match-bar    { background:#e50914; border-radius:2px; height:3px; }
    .stButton button { background:#e50914 !important; color:#fff !important; border:none !important; border-radius:6px !important; font-weight:600 !important; }
    .stButton button:hover { background:#b00710 !important; }
    div[data-testid="stImage"] img { border-radius:8px; }
    h1 { color:#fff !important; }
    h2, h3 { color:#ddd !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def safe_list(val):
    if isinstance(val, list):
        return val
    try:
        r = ast.literal_eval(str(val))
        return r if isinstance(r, list) else []
    except Exception:
        return []


def fmt_rating(val):
    try:
        r = float(val)
        return f"{r:.1f}" if r > 0 else "N/A"
    except Exception:
        return "N/A"


def fmt_year(val):
    try:
        y = int(val)
        return str(y) if y > 1800 else "N/A"
    except Exception:
        return str(val) if val else "N/A"


def fmt_genres(val):
    gl = safe_list(val)
    if gl:
        return ", ".join(gl[:3])
    if isinstance(val, str) and val.strip():
        return val[:40]
    return "—"


def find_suggestion(title, df):
    from difflib import get_close_matches
    matches = get_close_matches(
        title.strip().lower(),
        df["title"].str.lower().tolist(),
        n=1, cutoff=0.6,
    )
    if matches:
        idx = df[df["title"].str.lower() == matches[0]].index
        if len(idx):
            return df.loc[idx[0], "title"]
    return None


# ─────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Loading movie database...")
def load_data():
    df = pd.read_csv("data/movies_clean.csv")
    for col in ["genres_list", "keywords_list", "cast_list"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_list)
    df["overview"]    = df["overview"].fillna("")
    df["director"]    = df["director"].fillna("")
    df["mood"]        = df["mood"].fillna("Mixed")
    df["soup"]        = df["soup"].fillna("")
    df["title"]       = df["title"].fillna("Unknown")
    df["poster_path"] = ""   # fetched live via TMDB API per card
    return df


@st.cache_resource(show_spinner="Loading TF-IDF model...")
def load_tfidf():
    with open("models/sim_tfidf.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading BERT embeddings...")
def load_bert():
    return np.load("data/embeddings.npy")


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster(movie_id, poster_path=""):
    """Fetch poster URL. Cached 24hrs per movie — API only called once."""
    path = str(poster_path).strip()
    if path and path not in ["nan", "None", ""]:
        return f"{POSTER_BASE}{path}"
    if TMDB_API_KEY and movie_id:
        try:
            import requests as _req
            r = _req.get(
                f"https://api.tmdb.org/3/movie/{movie_id}",
                params={"api_key": TMDB_API_KEY},
                timeout=4,
            )
            if r.status_code == 200:
                p = r.json().get("poster_path", "")
                if p:
                    return f"{POSTER_BASE}{p}"
        except Exception:
            pass
    return PLACEHOLDER


@st.cache_data(ttl=1800, show_spinner=False)
def cached_trending():
    return get_trending("week", n=10)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_now_playing():
    return get_now_playing(n=10)


# ─────────────────────────────────────────────
# RECOMMENDATION RUNNER
# ─────────────────────────────────────────────

def run_recommendations(title, df, sim_matrix, embeddings, mode, n, filter_kwargs):
    from difflib import get_close_matches

    title_lower  = title.strip().lower()
    title_to_idx = pd.Series(df.index, index=df["title"].str.lower()).to_dict()
    in_dataset   = title_lower in title_to_idx or bool(
        get_close_matches(title_lower, title_to_idx.keys(), n=1, cutoff=0.5)
    )

    if in_dataset:
        if mode == "BERT":
            from utils.bert_engine import get_bert_recommendations
            result_df, err = get_bert_recommendations(
                title, df, embeddings, n=n, **filter_kwargs)
        else:
            from utils.tfidf_engine import get_recommendations
            result_df, err = get_recommendations(
                title, df, sim_matrix, n=n, **filter_kwargs)

        if err:
            return [], err, False, title
        if result_df is None or result_df.empty:
            return [], "No results match your filters. Try relaxing them.", False, title

        records = result_df.to_dict("records")
        for r in records:
            r["poster_url"] = fetch_poster(r.get("id", ""), r.get("poster_path", ""))
            if not r.get("genres"):
                r["genres"] = fmt_genres(r.get("genres_list", []))
        matched = find_suggestion(title, df) or title
        return records, None, False, matched

    # TMDB Live fallback
    if not TMDB_API_KEY:
        return [], (
            f"'{title}' is not in our dataset (pre-2017). "
            "Add TMDB_API_KEY to .env to search live."
        ), False, title

    movie = search_movie(title)
    if not movie:
        return [], f"'{title}' not found in dataset or on TMDB.", False, title

    live_recs = get_live_recommendations(movie["id"], n=n)
    for r in live_recs:
        r["poster_url"] = fetch_poster(r.get("id", ""), r.get("poster_path", ""))
    return live_recs, None, True, movie.get("title", title)


# ─────────────────────────────────────────────
# MOVIE CARD
# ─────────────────────────────────────────────

def render_card(movie, show_sim=True, key_prefix=""):
    poster_url = movie.get("poster_url", PLACEHOLDER)
    title      = movie.get("title", "Unknown")
    rating     = fmt_rating(movie.get("vote_average", 0))
    year       = fmt_year(movie.get("release_year", ""))
    genres     = fmt_genres(movie.get("genres") or movie.get("genres_list", []))
    overview   = movie.get("overview", "")
    mood       = movie.get("mood", "")
    sim        = float(movie.get("similarity") or 0)
    is_live    = movie.get("source", "") == "TMDB Live API"

    st.markdown('<div class="movie-card">', unsafe_allow_html=True)

    st.image(poster_url if poster_url else PLACEHOLDER, use_container_width=True)

    st.markdown(f'<div class="movie-title">{title}</div>', unsafe_allow_html=True)

    badges = f'<span class="badge-red">⭐ {rating}</span>'
    if mood:
        badges += f'<span class="badge-dark">{MOOD_EMOJI.get(mood, "")} {mood}</span>'
    if is_live:
        badges += '<span class="badge-blue">🌐 Live</span>'
    st.markdown(badges, unsafe_allow_html=True)

    st.markdown(
        f'<div class="movie-meta">📅 {year} &nbsp;·&nbsp; 🎭 {genres}</div>',
        unsafe_allow_html=True
    )

    if show_sim and sim > 0 and not is_live:
        pct = int(sim * 100)
        st.markdown(
            f'<div class="movie-meta">Match: {pct}%</div>'
            f'<div class="match-bar-bg"><div class="match-bar" style="width:{pct}%"></div></div>',
            unsafe_allow_html=True
        )

    if st.button("📖 View Full Profile", key=f"btn_profile_{key_prefix}_{movie.get('id', title)}_{title[:5]}", use_container_width=True):
        movie_profile_dialog(title)

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🎬 CineMatch")
        st.caption("AI-powered movie recommendations")
        st.divider()

        st.markdown("**🤖 Engine**")
        mode = st.radio(
            "Engine", ["BERT", "TF-IDF"],
            label_visibility="collapsed",
            help="BERT = semantic similarity. TF-IDF = keyword matching.",
        )
        st.caption("✨ Semantic similarity" if mode == "BERT" else "⚡ Keyword matching")

        st.divider()
        st.markdown("**😄 Mood**")
        mood_options = ["Any"] + list(MOOD_MAP.keys())
        mood_labels  = [f"{MOOD_EMOJI.get(m, '🎬')} {m}" for m in mood_options]
        mood_idx = st.selectbox(
            "Mood", range(len(mood_options)),
            format_func=lambda i: mood_labels[i],
            label_visibility="collapsed",
        )
        mood = mood_options[mood_idx]
        if mood != "Any":
            st.caption(MOOD_DESCRIPTIONS.get(mood, ""))

        st.divider()
        st.markdown("**🎭 Genres**")
        genres = st.multiselect(
            "Genres", ALL_GENRES,
            label_visibility="collapsed",
            placeholder="All genres",
        )

        st.divider()
        st.markdown("**⭐ Min Rating**")
        min_rating = st.slider(
            "Min Rating", 0.0, 10.0, 0.0, 0.5,
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**📅 Release Year**")
        year_range = st.slider(
            "Year", 1950, 2024, (1990, 2024), 1,
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**🔢 Results**")
        n_results = st.slider(
            "Results", 5, 20, 10, 1,
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("📦 Dataset: TMDB 5000 (1950–2017)")
        if TMDB_API_KEY:
            st.caption("✅ TMDB Live API connected")
        else:
            st.caption("⚠️ No TMDB key — add to .env")


    return {
        "mode": mode, "mood": mood, "genres": genres,
        "min_rating": min_rating, "max_rating": 10.0,
        "year_min": year_range[0], "year_max": year_range[1],
        "n_results": n_results,
    }


# ─────────────────────────────────────────────
# TAB 1 — RECOMMEND
# ─────────────────────────────────────────────

def tab_search(df, sim_matrix, embeddings, sidebar):
    st.markdown('<div class="section-hdr">🔍 Find Similar Movies</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Movie title",
            placeholder="e.g. Inception, Oppenheimer, The Dark Knight…",
            label_visibility="collapsed",
        )
    with col2:
        st.button("🎬 Recommend", use_container_width=True)

    if not query:
        st.markdown(
            '<div class="info-box">💡 Type any movie title. '
            'Movies after 2017 are fetched live from TMDB automatically.</div>',
            unsafe_allow_html=True
        )
        return

    filter_kwargs = {
        "genre_filter": sidebar["genres"] or None,
        "mood_filter":  sidebar["mood"],
        "min_rating":   sidebar["min_rating"],
        "max_rating":   sidebar["max_rating"],
        "year_min":     sidebar["year_min"],
        "year_max":     sidebar["year_max"],
    }

    with st.spinner(f"Finding movies like **{query}**…"):
        results, err, is_live, matched = run_recommendations(
            query, df, sim_matrix, embeddings,
            mode=sidebar["mode"],
            n=sidebar["n_results"],
            filter_kwargs=filter_kwargs,
        )

    if not is_live and matched.lower() != query.strip().lower():
        st.markdown(
            f'<div class="info-box">🔎 Matched <b>"{matched}"</b> '
            f'for your search "<b>{query}</b>"</div>',
            unsafe_allow_html=True
        )

    if err:
        st.error(err)
        suggestion = find_suggestion(query, df)
        if suggestion and suggestion.lower() != query.strip().lower():
            st.markdown(
                f'<div class="warn-box">💡 Did you mean: <b>{suggestion}</b>?</div>',
                unsafe_allow_html=True
            )
        return

    if not results:
        st.warning("No results. Try lowering min rating or widening the year range.")
        return

    if is_live:
        st.markdown(
            f'<div class="info-box">🌐 <b>Live results</b> — '
            f'"{matched}" fetched from TMDB in real time.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="info-box">✅ <b>{len(results)} results</b> '
            f'via <b>{sidebar["mode"]}</b> engine</div>',
            unsafe_allow_html=True
        )

    cols = st.columns(5)
    for i, movie in enumerate(results):
        with cols[i % 5]:
            render_card(movie, show_sim=not is_live, key_prefix=f"search_{i}")


# ─────────────────────────────────────────────
# TAB 2 — BROWSE BY MOOD
# ─────────────────────────────────────────────

def tab_browse(df, sidebar):
    st.markdown('<div class="section-hdr">🎭 Browse by Mood</div>', unsafe_allow_html=True)

    if "browse_mood" not in st.session_state:
        st.session_state["browse_mood"] = "Happy"

    mood_cols = st.columns(6)
    for i, mood in enumerate(list(MOOD_MAP.keys())):
        with mood_cols[i % 6]:
            active = st.session_state["browse_mood"] == mood
            label  = f"{'▶ ' if active else ''}{MOOD_EMOJI.get(mood, '🎬')} {mood}"
            if st.button(label, key=f"mb_{mood}", use_container_width=True):
                st.session_state["browse_mood"] = mood
                st.rerun()

    selected = st.session_state["browse_mood"]
    st.markdown(
        f'<div class="section-hdr">{MOOD_EMOJI.get(selected, "🎬")} Top {selected} Movies'
        f'<span style="font-size:13px;font-weight:400;color:#888;margin-left:10px">'
        f'{MOOD_DESCRIPTIONS.get(selected, "")}</span></div>',
        unsafe_allow_html=True
    )

    top = get_top_by_mood(df, selected, n=sidebar["n_results"], min_votes=200)
    if top.empty:
        st.info(f"No movies found for mood: {selected}")
        return

    records = top.to_dict("records")
    for r in records:
        r["poster_url"] = fetch_poster(r.get("id", ""), r.get("poster_path", ""))
        r["genres"]     = fmt_genres(r.get("genres_list", []))

    cols = st.columns(5)
    for i, movie in enumerate(records):
        with cols[i % 5]:
            render_card(movie, show_sim=False, key_prefix=f"browse_{i}")


# ─────────────────────────────────────────────
# TAB 3 — STATS
# ─────────────────────────────────────────────

def tab_stats(df):
    st.markdown('<div class="section-hdr">📊 Dataset Overview</div>', unsafe_allow_html=True)

    valid_years = df["release_year"][df["release_year"] > 0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Movies", f"{len(df):,}")
    c2.metric("Avg Rating",   f"{df['vote_average'].mean():.2f} / 10")
    c3.metric("Year Range",   f"{int(valid_years.min())}–{int(valid_years.max())}")
    c4.metric("Directors",    f"{(df['director'] != '').sum():,}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Mood Distribution**")
        st.bar_chart(df["mood"].value_counts())
    with col2:
        st.markdown("**Rating Distribution**")
        labels = ["0–4", "4–6", "6–7", "7–8", "8–10"]
        cuts   = pd.cut(df["vote_average"], bins=[0, 4, 6, 7, 8, 10], labels=labels)
        st.bar_chart(cuts.value_counts().sort_index())

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 10 Directors**")
        st.bar_chart(df[df["director"] != ""]["director"].value_counts().head(10))
    with col2:
        st.markdown("**Top 10 Genres**")
        from collections import Counter
        all_g = [g for gl in df["genres_list"] for g in (gl if isinstance(gl, list) else [])]
        st.bar_chart(pd.Series(dict(Counter(all_g).most_common(10))))


# ─────────────────────────────────────────────
# TAB 4 — NEW & TRENDING
# ─────────────────────────────────────────────

def render_profile_ui(profile):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(profile.get("poster_url", PLACEHOLDER), use_container_width=True)
    with col2:
        st.markdown(f"### {profile.get('title', 'Unknown')}")
        if profile.get("tagline"):
            st.caption(profile["tagline"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rating",  f"⭐ {profile.get('vote_average', 0):.1f}/10")
        m2.metric("Year",    profile.get("release_year", "N/A"))
        m3.metric("Runtime", f"{profile.get('runtime', 0)} min")
        m4.metric("Reviews", profile.get("review_count", 0))
        st.markdown(f"**Genres:** {profile.get('genres', '—')}")
        st.markdown(f"**Director:** {profile.get('director', '—')}")
        if profile.get("cast"):
            st.markdown(f"**Cast:** {', '.join(profile['cast'])}")
        st.divider()
        st.write(profile.get("overview", ""))
        
        # YouTube Trailer
        trailer_url = profile.get("trailer_url") or get_trailer_url(f"{profile.get('title', '')} {profile.get('release_year', '')}".strip())
        if trailer_url:
            st.video(trailer_url)

    reviews = profile.get("reviews", [])
    if reviews:
        st.markdown("**💬 Audience Reviews**")
        for rev in reviews:
            rating_str = f"⭐ {rev['rating']}/10" if rev.get("rating") else ""
            st.markdown(
                f'<div class="movie-card">'
                f'<div class="movie-meta"><b>{rev["author"]}</b>'
                f'&nbsp;{rating_str}&nbsp;·&nbsp;{rev["date"]}</div>'
                f'<div style="color:#ccc;font-size:13px;margin-top:6px">'
                f'{rev["content"][:400]}{"…" if len(rev["content"]) > 400 else ""}'
                f'</div></div>',
                unsafe_allow_html=True
            )

@st.dialog("🎬 Movie Details", width="large")
def movie_profile_dialog(title):
    with st.spinner(f"Loading full profile for {title}..."):
        profile = get_full_movie_profile(title)
        
    if not profile:
        st.error(f"Could not load full profile from TMDB for {title}.")
        return
    render_profile_ui(profile)

def tab_trending():
    st.markdown('<div class="section-hdr">🔥 New & Trending</div>', unsafe_allow_html=True)

    if not TMDB_API_KEY:
        st.markdown(
            '<div class="warn-box">⚠️ TMDB API key not set. '
            'Add TMDB_API_KEY to your .env file to enable live data.</div>',
            unsafe_allow_html=True
        )
        return

    # ── Movie lookup with reviews ────────────
    st.markdown("**🔎 Look up any movie — full details + reviews**")
    col1, col2 = st.columns([4, 1])
    with col1:
        lookup = st.text_input(
            "Lookup",
            placeholder="e.g. Oppenheimer, Dune Part Two, Inside Out 2…",
            label_visibility="collapsed",
            key="lookup_input",
        )
    with col2:
        st.button("Look up", use_container_width=True, key="lookup_btn")

    if lookup:
        with st.spinner(f"Fetching **{lookup}**…"):
            profile = get_full_movie_profile(lookup)

        if not profile:
            st.error(f"'{lookup}' not found on TMDB.")
        else:
            render_profile_ui(profile)
        st.divider()

    # ── Trending ─────────────────────────────
    st.markdown('<div class="section-hdr">📈 Trending This Week</div>', unsafe_allow_html=True)
    trending = cached_trending()
    if trending:
        cols = st.columns(5)
        for i, movie in enumerate(trending):
            movie["poster_url"] = fetch_poster(movie.get("id", ""), movie.get("poster_path", ""))
            with cols[i % 5]:
                render_card(movie, show_sim=False, key_prefix=f"trend_{i}")
    else:
        st.info("Could not load trending movies.")

    st.divider()

    # ── Now playing ──────────────────────────
    st.markdown('<div class="section-hdr">🎟️ Now Playing in Cinemas</div>', unsafe_allow_html=True)
    now_playing = cached_now_playing()
    if now_playing:
        cols = st.columns(5)
        for i, movie in enumerate(now_playing):
            movie["poster_url"] = fetch_poster(movie.get("id", ""), movie.get("poster_path", ""))
            with cols[i % 5]:
                render_card(movie, show_sim=False, key_prefix=f"now_{i}")
    else:
        st.info("Could not load now playing movies.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
def main():
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("❌ movies_clean.csv not found. Run: `python utils/data_loader.py`")
        st.stop()

    try:
        sim_matrix = load_tfidf()
    except FileNotFoundError:
        st.error("❌ TF-IDF model not found. Run: `python utils/tfidf_engine.py`")
        st.stop()

    try:
        embeddings = load_bert()
    except FileNotFoundError:
        st.error("❌ BERT embeddings not found. Run: `python utils/bert_engine.py`")
        st.stop()

    sidebar = render_sidebar()

    st.markdown("# 🎬 CineMatch")
    st.caption("Content-based recommender · BERT + TF-IDF · TMDB Live API")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Recommend", "🎭 Browse by Mood", "📊 Stats", "🔥 New & Trending"
    ])
    with tab1:
        tab_search(df, sim_matrix, embeddings, sidebar)
    with tab2:
        tab_browse(df, sidebar)
    with tab3:
        tab_stats(df)
    with tab4:
        tab_trending()


    # Floating CineBot widget
    render_chatbot_widget(df)


if __name__ == "__main__":
    main()