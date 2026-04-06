from youtube_search import YoutubeSearch
import streamlit as st

@st.cache_data(ttl=86400, show_spinner=False)
def get_trailer_url(query):
    try:
        # Search for the movie trailer
        search_query = f"{query} official trailer"
        results = YoutubeSearch(search_query, max_results=1).to_dict()
        
        if results:
            video_id = results[0]['id']
            return f"https://www.youtube.com/watch?v={video_id}"
        return None
    except Exception as e:
        # Catch any network or parsing errors
        return None
