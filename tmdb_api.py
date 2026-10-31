"""
TMDB API Client with Fast Graceful Fallback
-------------------------------------------
Provides live movie metadata, backdrops, and discovery from The Movie Database (TMDB).
Includes:
- Fast 1.0s connectivity check to prevent freezing when network is offline/sandboxed
- Streamlit cache decorators (@st.cache_data)
- Graceful delegation to local catalog on connection errors or missing keys
- High quality SVG fallback placeholder for missing posters
"""

from __future__ import annotations

import html
import logging
import os
from typing import Any, Dict, List, Optional
import requests
import streamlit as st

logger = logging.getLogger(__name__)

TMDB_API_BASE = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p"

GENRE_MAP: Dict[str, int] = {
    "Action": 28,
    "Adventure": 12,
    "Animation": 16,
    "Comedy": 35,
    "Crime": 80,
    "Documentary": 99,
    "Drama": 18,
    "Family": 10751,
    "Fantasy": 14,
    "History": 36,
    "Horror": 27,
    "Music": 10402,
    "Mystery": 9648,
    "Romance": 10749,
    "Science Fiction": 878,
    "TV Movie": 10770,
    "Thriller": 53,
    "War": 10752,
    "Western": 37,
}


def get_api_key() -> str:
    """Retrieve the TMDB API key securely from st.secrets or environment variable."""
    if "tmdb_api_key_override" in st.session_state and st.session_state.tmdb_api_key_override:
        return str(st.session_state.tmdb_api_key_override).strip()

    try:
        if "TMDB_API_KEY" in st.secrets:
            return str(st.secrets["TMDB_API_KEY"]).strip()
    except Exception:
        pass

    return os.getenv("TMDB_API_KEY", "").strip()


@st.cache_data(ttl=300, show_spinner=False)
def is_tmdb_reachable() -> bool:
    """
    Fast, cached connectivity check (timeout 1.0s).
    Returns True only if TMDB is reachable and API key is valid.
    """
    key = get_api_key()
    if not key:
        return False
    try:
        resp = requests.get(
            f"{TMDB_API_BASE}/configuration",
            params={"api_key": key},
            timeout=1.0,
        )
        return resp.status_code == 200
    except Exception:
        return False


def is_tmdb_available() -> bool:
    """Check if TMDB is configured and reachable."""
    return is_tmdb_reachable()


@st.cache_data(ttl=1800, show_spinner=False)
def tmdb_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Perform a GET request to TMDB API with short timeout and robust exception handling.
    """
    key = get_api_key()
    if not key or not is_tmdb_reachable():
        return {}

    query_params = dict(params or {})
    query_params["api_key"] = key

    url = f"{TMDB_API_BASE}/{endpoint.lstrip('/')}"
    try:
        resp = requests.get(url, params=query_params, timeout=1.5)
        if resp.status_code == 200:
            return resp.json()
        logger.debug(f"TMDB request returned status {resp.status_code} for {endpoint}")
        return {}
    except (requests.RequestException, Exception) as exc:
        logger.debug(f"TMDB network error for {endpoint}: {exc}")
        return {}


def get_poster_url(path: Optional[str], size: str = "w500") -> Optional[str]:
    """Build full TMDB image URL."""
    if not path or not str(path).strip() or path == "None":
        return None
    p = str(path).strip()
    if not p.startswith("/"):
        p = "/" + p
    return f"{IMAGE_BASE_URL}/{size}{p}"


def get_backdrop_url(path: Optional[str], size: str = "w1280") -> Optional[str]:
    """Build full TMDB backdrop image URL."""
    return get_poster_url(path, size=size)


def get_live_discovery(category: str, page: int = 1) -> List[Dict[str, Any]]:
    """Fetch discovery category results from TMDB if connected."""
    if not is_tmdb_reachable():
        return []

    category_endpoints = {
        "Trending": ("trending/movie/day", {"page": page}),
        "Popular": ("movie/popular", {"page": page}),
        "Highest Rated": ("movie/top_rated", {"page": page}),
        "Now Playing": ("movie/now_playing", {"page": page}),
        "Upcoming": ("movie/upcoming", {"page": page}),
    }

    if category not in category_endpoints:
        return []

    endpoint, params = category_endpoints[category]
    data = tmdb_get(endpoint, params)
    raw_results = data.get("results", [])
    return [_normalize_tmdb_record(m) for m in raw_results if m.get("id")]


def search_tmdb_movies(query: str, page: int = 1) -> List[Dict[str, Any]]:
    """Search movies using TMDB API."""
    if not query or not query.strip() or not is_tmdb_reachable():
        return []

    data = tmdb_get("search/movie", {
        "query": query.strip(),
        "include_adult": "false",
        "page": page,
    })
    raw_results = data.get("results", [])
    return [_normalize_tmdb_record(m) for m in raw_results if m.get("id")]


def get_tmdb_movie_details(movie_id: int) -> Dict[str, Any]:
    """Fetch movie details including runtime, genres, credits and backdrop from TMDB."""
    if not movie_id or not is_tmdb_reachable():
        return {}

    details = tmdb_get(f"movie/{int(movie_id)}", {"append_to_response": "credits,recommendations"})
    if not details or not details.get("id"):
        return {}

    credits = details.get("credits", {})
    cast_list = [c.get("name") for c in credits.get("cast", [])[:5] if c.get("name")]
    crew_list = credits.get("crew", [])
    directors = [c.get("name") for c in crew_list if c.get("job") == "Director"]
    director = directors[0] if directors else ""

    genres = [g.get("name") for g in details.get("genres", []) if g.get("name")]

    recs_raw = details.get("recommendations", {}).get("results", [])
    recs = [_normalize_tmdb_record(r) for r in recs_raw[:10] if r.get("id")]

    return {
        "id": int(details["id"]),
        "title": str(details.get("title", "Untitled")),
        "release_date": str(details.get("release_date", "")),
        "release_year": int(str(details.get("release_date", ""))[:4]) if str(details.get("release_date", ""))[:4].isdigit() else 0,
        "vote_average": round(float(details.get("vote_average", 0.0)), 1),
        "vote_count": int(details.get("vote_count", 0)),
        "popularity": round(float(details.get("popularity", 0.0)), 1),
        "runtime": details.get("runtime"),
        "overview": str(details.get("overview", "")),
        "tagline": str(details.get("tagline", "")),
        "poster_path": details.get("poster_path") or "",
        "backdrop_path": details.get("backdrop_path") or "",
        "genres": genres,
        "director": director,
        "cast": cast_list,
        "similar_movies": recs,
    }


def _normalize_tmdb_record(m: Dict[str, Any]) -> Dict[str, Any]:
    release = str(m.get("release_date", "") or "")
    year = int(release[:4]) if release[:4].isdigit() else 0

    return {
        "id": int(m["id"]),
        "title": str(m.get("title") or m.get("original_title") or "Untitled"),
        "release_date": release,
        "release_year": year,
        "vote_average": round(float(m.get("vote_average", 0.0)), 1),
        "vote_count": int(m.get("vote_count", 0)),
        "popularity": round(float(m.get("popularity", 0.0)), 1),
        "overview": str(m.get("overview", "") or ""),
        "poster_path": m.get("poster_path") or "",
        "backdrop_path": m.get("backdrop_path") or "",
        "genres": [],
        "director": "",
        "cast": [],
        "from_tmdb": True,
    }


def render_poster_placeholder_html(title: str, height_px: int = 280) -> str:
    """Generate an elegant, dark cinematic poster fallback placeholder."""
    safe_title = html.escape(title[:36] + ("..." if len(title) > 36 else ""))
    return f"""
    <div style="
        height: {height_px}px;
        width: 100%;
        background: linear-gradient(135deg, #181d28 0%, #0d1017 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 16px;
        text-align: center;
        box-sizing: border-box;
    ">
        <div style="font-size: 32px; margin-bottom: 8px; opacity: 0.65;">🎬</div>
        <div style="
            font-size: 13px;
            font-weight: 600;
            color: #cbd5e1;
            line-height: 1.3;
            max-height: 2.6em;
            overflow: hidden;
        ">{safe_title}</div>
        <div style="font-size: 11px; color: #64748b; margin-top: 6px;">No Poster Available</div>
    </div>
    """
