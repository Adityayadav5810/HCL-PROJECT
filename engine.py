"""
CineMatch Recommendation Engine
-------------------------------
Core hybrid recommendation pipeline combining:
1. Content-based filtering using TF-IDF and cosine similarity on movie tags
   (genres, overview, keywords, cast, and director).
2. Collaborative filtering using TruncatedSVD latent factors trained on
   MovieLens user-movie co-ratings.
3. Hybrid ranking with customizable weights and graceful cold-start fallbacks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


def is_scalar_na(val: Any) -> bool:
    """
    Safely check if a scalar value is null/NaN without array truth value ambiguity.
    Prevents ValueError: The truth value of an array with more than one element is ambiguous.
    """
    if val is None:
        return True
    if isinstance(val, (list, tuple, np.ndarray, dict, set)):
        return False
    try:
        res = pd.isna(val)
        if isinstance(res, (bool, np.bool_)):
            return bool(res)
        if isinstance(res, np.ndarray):
            return bool(res.any()) if res.size > 0 else False
        return bool(res)
    except (ValueError, TypeError):
        return False


def local_movie_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert movie rows into standard dictionary records.
    Safely handles array and list columns without triggering pd.isna ambiguity.
    """
    if frame is None or frame.empty:
        return []

    result: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        movie: Dict[str, Any] = {}
        for column, value in row.items():
            if is_scalar_na(value):
                if column == "vote_average":
                    value = 0.0
                elif column in ("genres", "genres_clean", "cast_names", "genres_list", "cast_list"):
                    value = []
                else:
                    value = ""
            movie[column] = value

        try:
            movie["id"] = int(movie.get("id", 0))
        except (ValueError, TypeError):
            continue

        movie["title"] = str(movie.get("title", "Untitled"))
        movie["overview"] = str(movie.get("overview", "") or "")
        movie["release_date"] = str(movie.get("release_date", "") or "")
        try:
            movie["vote_average"] = float(movie.get("vote_average", 0) or 0)
        except (ValueError, TypeError):
            movie["vote_average"] = 0.0

        result.append(movie)

    return result


@st.cache_resource(show_spinner=False)
def load_core_models() -> Dict[str, Any]:
    """
    Load core movie catalog and content/item-collaborative models.
    Lazy-loads large user matrix separately to keep startup fast.
    """
    required_files = [
        "movies_model.pkl",
        "tfidf_matrix.pkl",
        "movie_factors.pkl",
        "movie_to_idx.pkl",
    ]

    missing = [f for f in required_files if not (MODEL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required model files in {MODEL_DIR}: {', '.join(missing)}"
        )

    movies_raw: pd.DataFrame = joblib.load(MODEL_DIR / "movies_model.pkl")
    tfidf_matrix = joblib.load(MODEL_DIR / "tfidf_matrix.pkl")
    movie_factors = np.asarray(joblib.load(MODEL_DIR / "movie_factors.pkl"))
    movie_to_idx: Dict[int, int] = joblib.load(MODEL_DIR / "movie_to_idx.pkl")

    # Clean and standardize movies dataframe
    movies = movies_raw.copy()

    # Ensure integer movie IDs
    movies["id"] = pd.to_numeric(movies["id"], errors="coerce").fillna(-1).astype(int)
    movies = movies[movies["id"] > 0].reset_index(drop=True)

    # Standardize string fields
    movies["title"] = movies["title"].fillna("Untitled").astype(str).str.strip()
    movies["overview"] = movies["overview"].fillna("").astype(str)
    movies["tagline"] = movies["tagline"].fillna("").astype(str)

    # Convert numeric fields properly (resolves string sorting bugs)
    movies["popularity"] = pd.to_numeric(movies["popularity"], errors="coerce").fillna(0.0).astype(float)
    movies["vote_average"] = pd.to_numeric(movies["vote_average"], errors="coerce").fillna(0.0).astype(float)
    movies["vote_count"] = pd.to_numeric(movies["vote_count"], errors="coerce").fillna(0.0).astype(float)

    # Release dates and years
    movies["release_date"] = movies["release_date"].fillna("").astype(str)
    parsed_dates = pd.to_datetime(movies["release_date"], errors="coerce")
    movies["release_year"] = parsed_dates.dt.year.fillna(0).astype(int)

    # Clean genres list safely without calling pd.isna on list objects
    if "genres_clean" in movies.columns:
        movies["genres_list"] = [
            val if isinstance(val, list) else ([str(val).strip()] if str(val).strip() else [])
            for val in movies["genres_clean"]
        ]
    else:
        movies["genres_list"] = [[] for _ in range(len(movies))]

    # Clean cast safely
    if "cast_names" in movies.columns:
        movies["cast_list"] = [
            val[:5] if isinstance(val, list) else []
            for val in movies["cast_names"]
        ]
    else:
        movies["cast_list"] = [[] for _ in range(len(movies))]

    if "director" not in movies.columns:
        movies["director"] = ""
    else:
        movies["director"] = movies["director"].fillna("").astype(str)

    # Build O(1) index mappings
    id_to_row = {int(mid): i for i, mid in enumerate(movies["id"])}
    title_to_rows: Dict[str, List[int]] = {}
    for i, t in enumerate(movies["title"]):
        key = t.lower()
        title_to_rows.setdefault(key, []).append(i)

    # Precalculate L2 norms for collaborative factors
    cf_norms = np.linalg.norm(movie_factors, axis=1)

    return {
        "movies": movies,
        "tfidf_matrix": tfidf_matrix,
        "movie_factors": movie_factors,
        "cf_norms": cf_norms,
        "movie_to_idx": movie_to_idx,
        "id_to_row": id_to_row,
        "title_to_rows": title_to_rows,
    }


@st.cache_resource(show_spinner=False)
def load_user_collaborative_models() -> Optional[Dict[str, Any]]:
    """
    Lazy load user factors and user rating history on demand.
    Only called when the user explores the 'Movie DNA / User Profile' tab.
    """
    user_files = ["user_factors.pkl", "user_to_idx.pkl", "user_rated_movies.pkl"]
    missing = [f for f in user_files if not (MODEL_DIR / f).exists()]
    if missing:
        return None

    try:
        user_factors = np.asarray(joblib.load(MODEL_DIR / "user_factors.pkl"))
        user_to_idx = joblib.load(MODEL_DIR / "user_to_idx.pkl")
        user_rated = joblib.load(MODEL_DIR / "user_rated_movies.pkl")
        return {
            "user_factors": user_factors,
            "user_to_idx": user_to_idx,
            "user_rated_movies": user_rated,
        }
    except Exception as e:
        logger.warning(f"Could not load user collaborative models: {e}")
        return None


def get_movie_row_index(models: Dict[str, Any], query: Union[int, str]) -> Optional[int]:
    """Safely resolve a movie ID or title to its row index in the catalog."""
    id_to_row = models["id_to_row"]
    title_to_rows = models["title_to_rows"]

    try:
        mid = int(query)
        if mid in id_to_row:
            return id_to_row[mid]
    except (ValueError, TypeError):
        pass

    query_str = str(query).strip().lower()
    if query_str in title_to_rows:
        return title_to_rows[query_str][0]

    for title, rows in title_to_rows.items():
        if query_str in title:
            return rows[0]

    return None


@st.cache_data(show_spinner=False, max_entries=256)
def compute_hybrid_recommendations(
    movie_id: int,
    top_n: int = 15,
    w_content: float = 0.6,
    w_collab: float = 0.4,
    genre_filter: Optional[str] = None,
    min_rating: float = 0.0,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Generate deterministic, vectorized hybrid recommendations for a movie.
    Returns: (list of recommendation dicts, mode_description).
    """
    try:
        models = load_core_models()
    except Exception as e:
        return [], f"Model loading error: {e}"

    movies: pd.DataFrame = models["movies"]
    tfidf_matrix = models["tfidf_matrix"]
    movie_factors = models["movie_factors"]
    cf_norms = models["cf_norms"]
    id_to_row = models["id_to_row"]

    row_idx = id_to_row.get(int(movie_id))
    if row_idx is None:
        return [], "Movie not found in catalog."

    seed_movie = movies.iloc[row_idx]
    seed_genres = set(seed_movie.get("genres_list", []))

    # 1. Content-based similarity via TF-IDF cosine similarity
    content_sim = cosine_similarity(tfidf_matrix[row_idx], tfidf_matrix).flatten()
    content_sim[row_idx] = 0.0  # Exclude seed movie

    c_max = content_sim.max()
    if c_max > 0:
        c_norm = content_sim / c_max
    else:
        c_norm = content_sim

    # 2. Collaborative item-item similarity
    collab_k = seed_movie.get("collab_idx")
    has_collab = not is_scalar_na(collab_k)

    full_cf = np.zeros(len(movies), dtype=np.float32)

    if has_collab:
        k = int(collab_k)
        target = movie_factors[k]
        target_norm = cf_norms[k]

        if target_norm > 0:
            denom = cf_norms * target_norm
            dots = movie_factors @ target
            raw_cf = np.divide(dots, denom, out=np.zeros_like(dots), where=denom != 0)
            raw_cf[k] = -1.0  # Exclude seed movie
            cf_norm = np.clip((raw_cf + 1.0) / 2.0, 0.0, 1.0)

            collab_mask = movies["collab_idx"].notna()
            full_cf[collab_mask] = cf_norm[movies.loc[collab_mask, "collab_idx"].astype(int)]

        hybrid_scores = (w_content * c_norm) + (w_collab * full_cf)
        mode = f"Hybrid ({int(w_content * 100)}% Content / {int(w_collab * 100)}% Collaborative)"
    else:
        hybrid_scores = c_norm.copy()
        mode = "Content-Based (Seed title has no collaborative factor history)"

    hybrid_scores[row_idx] = -1.0

    # Apply optional genre and rating filters if requested
    candidate_mask = np.ones(len(movies), dtype=bool)
    candidate_mask[row_idx] = False

    if min_rating > 0.0:
        candidate_mask &= movies["vote_average"].values >= min_rating

    if genre_filter and genre_filter != "All genres":
        genre_lower = genre_filter.lower()
        genre_matches = np.array([
            any(genre_lower == g.lower() for g in gl) for gl in movies["genres_list"]
        ], dtype=bool)
        candidate_mask &= genre_matches

    filtered_scores = np.where(candidate_mask, hybrid_scores, -1.0)

    num_candidates = int(np.sum(filtered_scores >= 0))
    if num_candidates == 0:
        return [], "No matching movies found with the current filter settings."

    top_count = min(top_n, num_candidates)
    top_indices = filtered_scores.argsort()[-top_count:][::-1]

    recommendations: List[Dict[str, Any]] = []
    for rank, idx in enumerate(top_indices, 1):
        row = movies.iloc[idx]
        rec_genres = set(row.get("genres_list", []))
        shared_genres = list(seed_genres & rec_genres)

        explanations = []
        if shared_genres:
            explanations.append(f"Shared genres: {', '.join(shared_genres)}")
        if has_collab and full_cf[idx] > 0.65:
            explanations.append("Strong audience co-rating pattern")
        if c_norm[idx] > 0.60:
            explanations.append("High plot/theme similarity")

        explanation_str = " · ".join(explanations) if explanations else "Content & audience affinity"

        recommendations.append({
            "rank": rank,
            "id": int(row["id"]),
            "title": str(row["title"]),
            "release_date": str(row.get("release_date", "")),
            "release_year": int(row.get("release_year", 0)),
            "vote_average": round(float(row.get("vote_average", 0.0)), 1),
            "vote_count": int(row.get("vote_count", 0)),
            "popularity": round(float(row.get("popularity", 0.0)), 1),
            "overview": str(row.get("overview", "")),
            "poster_path": str(row.get("poster_path", "")) if not is_scalar_na(row.get("poster_path")) else "",
            "genres": row.get("genres_list", []),
            "director": str(row.get("director", "")),
            "cast": row.get("cast_list", []),
            "hybrid_score": round(float(hybrid_scores[idx]), 3),
            "content_score": round(float(c_norm[idx]), 3),
            "collab_score": round(float(full_cf[idx]), 3) if has_collab else 0.0,
            "explanation": explanation_str,
        })

    return recommendations, mode


def get_side_by_side_comparison(
    movie_id: int, top_n: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    """Side-by-side comparison of pure Content, pure Collaborative, and Hybrid."""
    content_recs, _ = compute_hybrid_recommendations(
        movie_id, top_n=top_n, w_content=1.0, w_collab=0.0
    )
    collab_recs, _ = compute_hybrid_recommendations(
        movie_id, top_n=top_n, w_content=0.0, w_collab=1.0
    )
    hybrid_recs, _ = compute_hybrid_recommendations(
        movie_id, top_n=top_n, w_content=0.6, w_collab=0.4
    )

    return {
        "content": content_recs,
        "collaborative": collab_recs,
        "hybrid": hybrid_recs,
    }


def get_user_profile_recommendations(user_id: int, top_n: int = 12) -> Tuple[List[Dict[str, Any]], str]:
    """Generate collaborative recommendations based on a MovieLens user's latent taste vector."""
    user_models = load_user_collaborative_models()
    if user_models is None:
        return [], "User collaborative models are not available."

    core_models = load_core_models()
    movies: pd.DataFrame = core_models["movies"]
    movie_factors = core_models["movie_factors"]
    movie_to_idx = core_models["movie_to_idx"]

    user_to_idx = user_models["user_to_idx"]
    user_factors = user_models["user_factors"]
    user_rated = user_models["user_rated_movies"]

    if user_id not in user_to_idx:
        return [], f"User ID {user_id} was not found in the MovieLens dataset (valid range: 1 to 265,917)."

    user_idx = user_to_idx[user_id]
    user_vec = user_factors[user_idx]
    rated_set = set(user_rated.get(user_id, []))

    scores = movie_factors @ user_vec
    ranked_indices = scores.argsort()[::-1]

    idx_to_movie_id = {idx: mid for mid, idx in movie_to_idx.items()}
    candidate_movie_ids: List[int] = []

    for idx in ranked_indices:
        mid = idx_to_movie_id.get(idx)
        if mid and mid not in rated_set:
            candidate_movie_ids.append(mid)
        if len(candidate_movie_ids) >= top_n:
            break

    id_to_row = core_models["id_to_row"]
    results: List[Dict[str, Any]] = []

    for rank, mid in enumerate(candidate_movie_ids, 1):
        if mid in id_to_row:
            row = movies.iloc[id_to_row[mid]]
            results.append({
                "rank": rank,
                "id": int(row["id"]),
                "title": str(row["title"]),
                "release_year": int(row.get("release_year", 0)),
                "vote_average": round(float(row.get("vote_average", 0.0)), 1),
                "poster_path": str(row.get("poster_path", "")) if not is_scalar_na(row.get("poster_path")) else "",
                "genres": row.get("genres_list", []),
                "overview": str(row.get("overview", "")),
                "explanation": f"Matches User {user_id}'s latent preference dimensions",
            })

    return results, f"Collaborative recommendations based on User {user_id}'s history ({len(rated_set)} rated films)."


def search_local_catalog(query: str, limit: int = 24) -> List[Dict[str, Any]]:
    """Fast local search over the 46,625 movie catalog."""
    if not query or not query.strip():
        return []

    models = load_core_models()
    movies: pd.DataFrame = models["movies"]
    q = query.strip().lower()

    exact_mask = movies["title"].str.lower() == q
    starts_mask = movies["title"].str.lower().str.startswith(q) & ~exact_mask
    contains_mask = (
        movies["title"].str.lower().str.contains(q, regex=False, na=False)
        & ~exact_mask
        & ~starts_mask
    )

    matches = pd.concat([
        movies[exact_mask],
        movies[starts_mask].sort_values("popularity", ascending=False),
        movies[contains_mask].sort_values("popularity", ascending=False),
    ]).drop_duplicates(subset=["id"]).head(limit)

    return _dataframe_to_records(matches)


def filter_local_catalog(
    genre: str = "All genres",
    year_range: Tuple[int, int] = (1970, 2026),
    min_rating: float = 0.0,
    sort_by: str = "Popularity",
    page: int = 1,
    per_page: int = 20,
) -> Tuple[List[Dict[str, Any]], int]:
    """Filter and paginate movies from the local 46,625 catalog."""
    models = load_core_models()
    movies: pd.DataFrame = models["movies"]

    filtered = movies.copy()

    if min_rating > 0.0:
        filtered = filtered[filtered["vote_average"] >= min_rating]

    if "release_year" in filtered.columns:
        filtered = filtered[
            (filtered["release_year"] >= year_range[0])
            & (filtered["release_year"] <= year_range[1])
        ]

    if genre != "All genres":
        genre_lower = genre.lower()
        mask = np.array([
            any(genre_lower == g.lower() for g in gl) for gl in filtered["genres_list"]
        ], dtype=bool)
        filtered = filtered[mask]

    if sort_by == "Rating":
        filtered = filtered[filtered["vote_count"] >= 50].sort_values(
            "vote_average", ascending=False
        )
    elif sort_by == "Release date":
        filtered = filtered.sort_values("release_date", ascending=False)
    elif sort_by == "Vote Count":
        filtered = filtered.sort_values("vote_count", ascending=False)
    else:
        filtered = filtered.sort_values("popularity", ascending=False)

    total_count = len(filtered)
    start_idx = max(0, (page - 1) * per_page)
    paged_df = filtered.iloc[start_idx : start_idx + per_page]

    return _dataframe_to_records(paged_df), total_count


def get_curated_collection(category: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Curated collections from the local catalog."""
    models = load_core_models()
    movies: pd.DataFrame = models["movies"]

    if category == "Trending":
        df = movies.sort_values(["popularity", "vote_count"], ascending=[False, False]).head(limit)
    elif category == "Popular":
        df = movies[movies["vote_count"] >= 100].sort_values("popularity", ascending=False).head(limit)
    elif category == "Highest Rated":
        df = movies[movies["vote_count"] >= 500].sort_values("vote_average", ascending=False).head(limit)
    elif category == "Latest":
        df = movies[movies["release_year"] >= 2014].sort_values("release_date", ascending=False).head(limit)
    elif category == "Sci-Fi & Fantasy":
        mask = np.array([any(g in ["Science Fiction", "Fantasy"] for g in gl) for gl in movies["genres_list"]], dtype=bool)
        df = movies[mask & (movies["vote_count"] >= 200)].sort_values("vote_average", ascending=False).head(limit)
    elif category == "Action & Adventure":
        mask = np.array([any(g in ["Action", "Adventure"] for g in gl) for gl in movies["genres_list"]], dtype=bool)
        df = movies[mask & (movies["vote_count"] >= 300)].sort_values("popularity", ascending=False).head(limit)
    elif category == "Animation & Family":
        mask = np.array([any(g in ["Animation", "Family"] for g in gl) for gl in movies["genres_list"]], dtype=bool)
        df = movies[mask & (movies["vote_count"] >= 100)].sort_values("vote_average", ascending=False).head(limit)
    else:
        df = movies.head(limit)

    return _dataframe_to_records(df)


def get_movie_by_id(movie_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve complete movie details for a given TMDB movie ID from local catalog."""
    models = load_core_models()
    id_to_row = models["id_to_row"]
    if movie_id not in id_to_row:
        return None
    row = models["movies"].iloc[id_to_row[movie_id]]
    return _series_to_record(row)


def get_engine_diagnostics() -> Dict[str, Any]:
    """Produce catalog coverage and model health metrics."""
    models = load_core_models()
    movies: pd.DataFrame = models["movies"]
    tfidf_matrix = models["tfidf_matrix"]
    movie_factors = models["movie_factors"]
    movie_to_idx = models["movie_to_idx"]

    total_catalog = len(movies)
    collab_count = len(movie_to_idx)
    coverage_pct = round((collab_count / total_catalog) * 100, 1)

    return {
        "total_movies": total_catalog,
        "tfidf_features": tfidf_matrix.shape[1],
        "tfidf_rows": tfidf_matrix.shape[0],
        "collab_movies": collab_count,
        "latent_dimensions": movie_factors.shape[1],
        "catalog_coverage_pct": coverage_pct,
        "avg_rating": round(float(movies["vote_average"].mean()), 2),
        "total_genres": len({g for gl in movies["genres_list"] for g in gl}),
    }


def _dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [_series_to_record(row) for _, row in df.iterrows()]


def _series_to_record(row: pd.Series) -> Dict[str, Any]:
    poster = row.get("poster_path")
    poster_str = str(poster) if not is_scalar_na(poster) and str(poster).strip() else ""

    return {
        "id": int(row["id"]),
        "title": str(row.get("title", "Untitled")),
        "release_date": str(row.get("release_date", "")),
        "release_year": int(row.get("release_year", 0)),
        "vote_average": round(float(row.get("vote_average", 0.0)), 1),
        "vote_count": int(row.get("vote_count", 0)),
        "popularity": round(float(row.get("popularity", 0.0)), 1),
        "overview": str(row.get("overview", "")),
        "tagline": str(row.get("tagline", "")),
        "poster_path": poster_str,
        "genres": row.get("genres_list", []),
        "director": str(row.get("director", "")),
        "cast": row.get("cast_list", []),
        "has_collaborative_factors": not is_scalar_na(row.get("collab_idx")),
    }
