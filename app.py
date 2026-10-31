"""
CineMatch - Modern Cinematic Movie Discovery & Recommendation Platform
------------------------------------------------------------------------
A full-stack movie discovery platform powered by hybrid machine learning:
- TF-IDF Content-Based Filtering (Genres, Overview, Keywords, Cast, Director)
- TruncatedSVD Collaborative Filtering (MovieLens co-rating latent factors)
- Hybrid ranking with real-time customizable weights and cold-start fallback
- Live TMDB API integration with offline catalog resilience (46,625 movies)
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Ensure local imports work regardless of execution working directory
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import engine
import styles
import tmdb_api

# ==========================================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================================

st.set_page_config(
    page_title="CineMatch · AI Movie Discovery & Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

styles.apply_cinematic_styles()

# ==========================================================
# 2. SESSION STATE INITIALIZATION
# ==========================================================

if "bookmarks" not in st.session_state:
    st.session_state.bookmarks = {}

if "watchlist" not in st.session_state:
    st.session_state.watchlist = {}

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "rec_seed_id" not in st.session_state:
    st.session_state.rec_seed_id = 862  # Toy Story default

# ==========================================================
# 3. REUSABLE UI COMPONENTS
# ==========================================================

def render_movie_card(movie: Dict[str, Any], section: str = "default", index: int = 0):
    """
    Render a responsive movie card with 2:3 poster, rating badge,
    title, and quick actions.
    """
    mid = movie.get("id", index)
    title = str(movie.get("title") or "Untitled")
    poster_path = movie.get("poster_path")
    vote_avg = movie.get("vote_average", 0.0)
    year = movie.get("release_year") or str(movie.get("release_date", ""))[:4] or "????"
    genres = movie.get("genres", [])
    hybrid_score = movie.get("hybrid_score")

    with st.container(border=True):
        # 1. Poster Image or Elegant Fallback Placeholder
        poster_url = tmdb_api.get_poster_url(poster_path) if poster_path else None
        if poster_url:
            st.image(poster_url, use_container_width=True)
        else:
            st.markdown(
                tmdb_api.render_poster_placeholder_html(title, height_px=260),
                unsafe_allow_html=True,
            )

        # 2. Match Score Badge (if present from recommendation engine)
        if hybrid_score is not None and hybrid_score > 0:
            pct = int(round(hybrid_score * 100))
            badge_class = "high" if pct >= 70 else "medium"
            st.markdown(
                f'<div class="match-badge {badge_class}">⚡ {pct}% Match</div>',
                unsafe_allow_html=True,
            )

        # 3. Title with 2-line clamp
        safe_title = html.escape(title)
        st.markdown(f'<div class="card-title" title="{safe_title}">{safe_title}</div>', unsafe_allow_html=True)

        # 4. Rating and Release Year Meta
        rating_str = f"⭐ {float(vote_avg):.1f}" if vote_avg and vote_avg > 0 else "⭐ Unrated"
        st.markdown(
            f'<div class="card-meta"><span>{rating_str}</span><span>{year}</span></div>',
            unsafe_allow_html=True,
        )

        # 5. Genre Pills (up to 2)
        if genres and isinstance(genres, list):
            pills = "".join(f'<span class="genre-pill">{html.escape(str(g))}</span>' for g in genres[:2])
            st.markdown(f'<div style="margin-bottom:8px;">{pills}</div>', unsafe_allow_html=True)

        # 6. Action Buttons
        if st.button("Details", key=f"det_{section}_{mid}_{index}", use_container_width=True):
            st.session_state.selected_movie = movie
            st.session_state.page = "Movie Details"
            st.rerun()

        col_a, col_b = st.columns(2)
        with col_a:
            is_bm = str(mid) in st.session_state.bookmarks
            bm_label = "★ Saved" if is_bm else "☆ Save"
            if st.button(bm_label, key=f"bm_{section}_{mid}_{index}", use_container_width=True):
                if is_bm:
                    st.session_state.bookmarks.pop(str(mid), None)
                else:
                    st.session_state.bookmarks[str(mid)] = movie
                st.rerun()

        with col_b:
            is_wl = str(mid) in st.session_state.watchlist
            wl_label = "✓ Listed" if is_wl else "＋ Later"
            if st.button(wl_label, key=f"wl_{section}_{mid}_{index}", use_container_width=True):
                if is_wl:
                    st.session_state.watchlist.pop(str(mid), None)
                else:
                    st.session_state.watchlist[str(mid)] = movie
                st.rerun()


def render_movie_grid(movies: List[Dict[str, Any]], columns: int = 5, section: str = "grid"):
    """
    Render a list of movie records in a clean responsive grid.
    """
    if not movies:
        st.info("No movies found matching the criteria.")
        return

    cols_count = max(1, min(columns, 6))
    for start in range(0, len(movies), cols_count):
        batch = movies[start : start + cols_count]
        cols = st.columns(len(batch), gap="small")
        for idx, (col, movie) in enumerate(zip(cols, batch)):
            with col:
                render_movie_card(movie, section=section, index=start + idx)


# ==========================================================
# 4. VIEW: HOME PAGE
# ==========================================================

def render_home_page():
    # Hero Spotlight Banner
    spotlight_movies = engine.get_curated_collection("Highest Rated", limit=5)
    hero = spotlight_movies[0] if spotlight_movies else {}

    hero_title = hero.get("title", "Discover Your Next Great Movie")
    hero_desc = hero.get("overview") or (
        "Explore 46,000+ cinematic titles powered by advanced hybrid machine learning. "
        "Combine plot semantics, creative DNA, and collective audience taste for precision recommendations."
    )
    hero_genres = " · ".join(hero.get("genres", [])[:3]) if hero.get("genres") else "Cinematic Universe"

    st.markdown(
        f"""
        <div class="hero-spotlight">
            <div class="hero-badge">✦ Spotlight Feature · {html.escape(hero_genres)}</div>
            <div class="hero-title">{html.escape(hero_title)}</div>
            <div class="hero-desc">{html.escape(hero_desc)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        if hero and st.button("Explore Recommendations", type="primary", use_container_width=True):
            st.session_state.rec_seed_id = hero.get("id", 862)
            st.session_state.page = "Recommendations"
            st.rerun()
    with col2:
        if hero and st.button("View Movie Details", use_container_width=False):
            st.session_state.selected_movie = hero
            st.session_state.page = "Movie Details"
            st.rerun()

    # Home Feeds
    feed_sections = [
        ("Trending Worldwide", "Trending"),
        ("Popular Blockbusters", "Popular"),
        ("Highest Rated Classics", "Highest Rated"),
        ("Action & High Octane", "Action & Adventure"),
        ("Sci-Fi & Cyberpunk", "Sci-Fi & Fantasy"),
        ("Animation & Family", "Animation & Family"),
    ]

    for title, cat in feed_sections:
        st.markdown(
            f"""
            <div class="section-header">
                <div class="section-title"><span>🎬</span> {html.escape(title)}</div>
                <div class="section-subtitle">Curated Catalog Selection</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Try live TMDB first if available and applicable, fallback to local catalog
        feed_movies = []
        if cat in ["Trending", "Popular", "Highest Rated"] and tmdb_api.is_tmdb_available():
            feed_movies = tmdb_api.get_live_discovery(cat, page=1)[:5]

        if not feed_movies:
            feed_movies = engine.get_curated_collection(cat, limit=5)

        render_movie_grid(feed_movies, columns=5, section=f"home_{cat.lower().replace(' ', '_')}")


# ==========================================================
# 5. VIEW: DISCOVER / BROWSE CATALOG
# ==========================================================

def render_discover_page():
    st.title("🧭 Discover Movies")
    st.caption("Filter and explore through the 46,625 movie catalog.")

    with st.container(border=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns([2, 3, 2, 2])

        with f_col1:
            genres_list = ["All genres"] + list(tmdb_api.GENRE_MAP.keys())
            selected_genre = st.selectbox("Genre", genres_list, index=0)

        with f_col2:
            year_range = st.slider("Release Year", 1920, 2026, (1980, 2024))

        with f_col3:
            min_rating = st.slider("Min Rating ⭐", 0.0, 10.0, 6.0, step=0.5)

        with f_col4:
            sort_options = ["Popularity", "Rating", "Release date", "Vote Count"]
            selected_sort = st.selectbox("Sort By", sort_options, index=0)

    # Page controls
    if "discover_page_num" not in st.session_state:
        st.session_state.discover_page_num = 1

    movies, total_found = engine.filter_local_catalog(
        genre=selected_genre,
        year_range=year_range,
        min_rating=min_rating,
        sort_by=selected_sort,
        page=st.session_state.discover_page_num,
        per_page=20,
    )

    st.markdown(
        f'<div style="margin: 14px 0; color: #94a3b8; font-size: 13px;">Found <strong>{total_found:,}</strong> movies matching criteria (Page {st.session_state.discover_page_num})</div>',
        unsafe_allow_html=True,
    )

    render_movie_grid(movies, columns=5, section="discover")

    # Pagination buttons
    p_prev, p_info, p_next = st.columns([1, 4, 1])
    with p_prev:
        if st.session_state.discover_page_num > 1:
            if st.button("← Previous", use_container_width=True):
                st.session_state.discover_page_num -= 1
                st.rerun()
    with p_info:
        st.caption(f"Showing items {min(1, total_found)} - {min(len(movies) + (st.session_state.discover_page_num - 1) * 20, total_found)} of {total_found:,}")
    with p_next:
        if st.session_state.discover_page_num * 20 < total_found:
            if st.button("Next Page →", use_container_width=True):
                st.session_state.discover_page_num += 1
                st.rerun()


# ==========================================================
# 6. VIEW: SEARCH MOVIES
# ==========================================================

def render_search_page():
    st.title("🔍 Search Movies")
    st.caption("Search across the catalog or live TMDB database by title or keyword.")

    query = st.text_input(
        "Search movie title",
        value=st.session_state.get("search_query", ""),
        placeholder="e.g. Inception, The Dark Knight, Interstellar, Pulp Fiction...",
        key="search_input_box",
    )

    if query.strip():
        # Try local search first
        local_results = engine.search_local_catalog(query, limit=20)

        # If connected to TMDB, search TMDB as well
        tmdb_results = []
        if tmdb_api.is_tmdb_available():
            tmdb_results = tmdb_api.search_tmdb_movies(query, page=1)[:10]

        # Combine results safely, preferring catalog records
        seen_ids = set()
        merged_results = []
        for m in local_results:
            seen_ids.add(m["id"])
            merged_results.append(m)

        for m in tmdb_results:
            if m["id"] not in seen_ids:
                seen_ids.add(m["id"])
                merged_results.append(m)

        if merged_results:
            st.markdown(
                f'<div style="margin: 12px 0; color: #94a3b8; font-size: 13px;">Found <strong>{len(merged_results)}</strong> matching movies for "<em>{html.escape(query)}</em>":</div>',
                unsafe_allow_html=True,
            )
            render_movie_grid(merged_results, columns=5, section="search_res")
        else:
            st.info(f"No movies found matching '{query}'. Try searching for a common keyword or title.")
    else:
        st.markdown(
            """
            <div style="padding: 40px; text-align: center; color: #64748b;">
                <div style="font-size: 40px; margin-bottom: 10px;">🍿</div>
                <div style="font-size: 16px; font-weight: 600; color: #cbd5e1;">Start typing to search movies</div>
                <div style="font-size: 13px; margin-top: 4px;">Supports instant title search, partial keywords, and character queries.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==========================================================
# 7. VIEW: HYBRID RECOMMENDATIONS WORKBENCH
# ==========================================================

def render_recommendations_page():
    st.title("🎯 Recommendation Engine")
    st.caption("AI-powered hybrid recommendation workbench combining content semantics and collaborative latent factors.")

    # Seed movie selector
    core_models = engine.load_core_models()
    movies_df: pd.DataFrame = core_models["movies"]

    popular_picks = [
        (862, "Toy Story (1995)"),
        (155, "The Dark Knight (2008)"),
        (157336, "Interstellar (2014)"),
        (27205, "Inception (2010)"),
        (680, "Pulp Fiction (1994)"),
        (603, "The Matrix (1999)"),
        (550, "Fight Club (1999)"),
        (120, "The Lord of the Rings: The Fellowship of the Ring (2001)"),
    ]

    st.subheader("1. Select Seed Movie")
    col_pick, col_search = st.columns([1, 2])

    with col_pick:
        pick_labels = {mid: label for mid, label in popular_picks}
        default_index = 0
        current_id = st.session_state.rec_seed_id
        if current_id in pick_labels:
            default_index = list(pick_labels.keys()).index(current_id)

        selected_quick = st.selectbox(
            "Quick Popular Picks",
            options=list(pick_labels.keys()),
            format_func=lambda x: pick_labels.get(x, f"Movie #{x}"),
            index=default_index,
        )
        if selected_quick != st.session_state.rec_seed_id:
            st.session_state.rec_seed_id = selected_quick
            st.rerun()

    with col_search:
        search_kw = st.text_input("Or search by movie title", placeholder="Type title to find seed movie...")
        if search_kw.strip():
            matches = engine.search_local_catalog(search_kw, limit=8)
            if matches:
                opt_map = {m["id"]: f"{m['title']} ({m['release_year']})" for m in matches}
                chosen_id = st.selectbox("Select match", options=list(opt_map.keys()), format_func=lambda x: opt_map[x])
                if st.button("Use Selected Movie", type="primary"):
                    st.session_state.rec_seed_id = chosen_id
                    st.rerun()

    # Retrieve current seed movie details
    seed_movie = engine.get_movie_by_id(st.session_state.rec_seed_id)
    if not seed_movie:
        st.error("Selected movie could not be located in the catalog.")
        return

    # Seed Movie Card Banner
    st.markdown(
        f"""
        <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 18px; margin: 16px 0;">
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #38bdf8; font-weight: 700;">Currently Analyzing Seed Movie</div>
            <div style="font-size: 22px; font-weight: 800; color: #ffffff; margin: 4px 0;">{html.escape(seed_movie['title'])} ({seed_movie['release_year']})</div>
            <div style="font-size: 13px; color: #cbd5e1; margin-bottom: 8px;">{html.escape(seed_movie['overview'][:240])}...</div>
            <div style="font-size: 12px; color: #94a3b8;">
                ⭐ {seed_movie['vote_average']} · {', '.join(seed_movie['genres'])} · Director: {seed_movie['director'] or 'N/A'} · 
                Collaborative Factors: {'✓ Available (MovieLens SVD)' if seed_movie['has_collaborative_factors'] else 'Cold-Start (TF-IDF Content Fallback)'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("2. Algorithmic Tuning")
    w_col1, w_col2, w_col3 = st.columns([3, 2, 2])

    with w_col1:
        content_weight_pct = st.slider(
            "Content Similarity Weight (Plot, Cast, Director)",
            min_value=0,
            max_value=100,
            value=60,
            step=5,
            help="Higher value favors semantic tag similarity; lower value favors audience co-rating patterns.",
        )
        w_content = content_weight_pct / 100.0
        w_collab = 1.0 - w_content

    with w_col2:
        top_n = st.slider("Number of Recommendations", 5, 25, 10, step=5)

    with w_col3:
        filter_genre = st.selectbox(
            "Filter Recommendations by Genre",
            ["All genres"] + list(tmdb_api.GENRE_MAP.keys()),
            index=0,
        )

    # Compute Recommendations
    recommendations, mode = engine.compute_hybrid_recommendations(
        movie_id=seed_movie["id"],
        top_n=top_n,
        w_content=w_content,
        w_collab=w_collab,
        genre_filter=filter_genre,
    )

    st.markdown(
        f'<div style="margin: 18px 0 10px 0; font-size: 14px; color: #38bdf8; font-weight: 600;">Engine Mode: {html.escape(mode)}</div>',
        unsafe_allow_html=True,
    )

    # Tabs: Recommendations Grid vs Deep Dive Breakdown vs Side-by-Side
    tab_grid, tab_table, tab_compare = st.tabs(["🖼️ Visual Recommendations", "📊 Score Breakdown & Explanations", "⚖️ Content vs Collab Comparison"])

    with tab_grid:
        render_movie_grid(recommendations, columns=5, section="recs_grid")

    with tab_table:
        if recommendations:
            table_data = []
            for r in recommendations:
                table_data.append({
                    "Rank": r["rank"],
                    "Movie Title": r["title"],
                    "Year": r["release_year"],
                    "Rating": r["vote_average"],
                    "Hybrid Score": f"{r['hybrid_score']:.3f}",
                    "Content Score": f"{r['content_score']:.3f}",
                    "Collab Score": f"{r['collab_score']:.3f}",
                    "Why Recommended": r["explanation"],
                })
            st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)
        else:
            st.info("No recommendations generated.")

    with tab_compare:
        st.caption("Compare how pure Content-Based filtering contrasts with pure Collaborative filtering for this movie.")
        comparison = engine.get_side_by_side_comparison(seed_movie["id"], top_n=5)
        c_left, c_mid, c_right = st.columns(3)

        with c_left:
            st.markdown("#### 1. Pure Content-Based (100% TF-IDF)")
            for rec in comparison["content"]:
                st.markdown(f"**{rec['rank']}. {rec['title']}** ({rec['release_year']}) — ⭐ {rec['vote_average']}<br><small style='color:#94a3b8;'>Sim: {rec['content_score']:.3f} · {', '.join(rec['genres'][:2])}</small>", unsafe_allow_html=True)

        with c_mid:
            st.markdown("#### 2. Pure Collaborative (100% SVD)")
            for rec in comparison["collaborative"]:
                st.markdown(f"**{rec['rank']}. {rec['title']}** ({rec['release_year']}) — ⭐ {rec['vote_average']}<br><small style='color:#94a3b8;'>CF Affinity: {rec['collab_score']:.3f} · {', '.join(rec['genres'][:2])}</small>", unsafe_allow_html=True)

        with c_right:
            st.markdown("#### 3. Hybrid Blend (60% / 40%)")
            for rec in comparison["hybrid"]:
                st.markdown(f"**{rec['rank']}. {rec['title']}** ({rec['release_year']}) — ⭐ {rec['vote_average']}<br><small style='color:#38bdf8;'>Match: {rec['hybrid_score']:.3f} · {', '.join(rec['genres'][:2])}</small>", unsafe_allow_html=True)


# ==========================================================
# 8. VIEW: MOVIE DNA / USER TASTE PROFILE
# ==========================================================

def render_movie_dna_page():
    st.title("🧬 Movie DNA · User Taste Profiling")
    st.caption("Explore recommendations generated from 265,917 real MovieLens learned user latent preference vectors.")

    st.markdown(
        """
        In production recommender systems, collaborative filtering learns a low-dimensional latent space
        representing implicit user tastes (e.g. affinity for noir cinematography, Nolan thrillers, or 90s animation).
        Enter any user ID between 1 and 265,917 to generate recommendations tailored to that user's historical ratings.
        """
    )

    col_in, col_btn = st.columns([2, 1])
    with col_in:
        user_id = st.number_input("MovieLens User ID", min_value=1, max_value=265917, value=1, step=1)
    with col_btn:
        st.write("")
        st.write("")
        run_dna = st.button("Generate User Recommendations", type="primary", use_container_width=True)

    if run_dna or "dna_recs" in st.session_state:
        if run_dna:
            with st.spinner("Analyzing user preference vector..."):
                recs, info = engine.get_user_profile_recommendations(int(user_id), top_n=10)
                st.session_state.dna_recs = (recs, info)
        else:
            recs, info = st.session_state.dna_recs

        st.markdown(f'<div style="margin: 16px 0; color: #38bdf8; font-weight: 600;">{html.escape(info)}</div>', unsafe_allow_html=True)
        if recs:
            render_movie_grid(recs, columns=5, section="user_dna")


# ==========================================================
# 9. VIEW: MOVIE DETAILS
# ==========================================================

def render_movie_details_page():
    movie = st.session_state.selected_movie
    if not movie:
        st.info("No movie selected. Return to Home or Search to pick a movie.")
        if st.button("← Back to Home"):
            st.session_state.page = "Home"
            st.rerun()
        return

    mid = movie.get("id")

    # Header and back button
    col_back, col_actions = st.columns([1, 4])
    with col_back:
        if st.button("← Back to Catalog"):
            st.session_state.page = "Home"
            st.rerun()

    # Try fetching fresh TMDB details (backdrop, runtime, director, full cast)
    details = {}
    if tmdb_api.is_tmdb_available() and mid:
        details = tmdb_api.get_tmdb_movie_details(mid)

    merged = {**movie, **details}

    # Backdrop
    backdrop_path = merged.get("backdrop_path")
    if backdrop_path:
        backdrop_url = tmdb_api.get_backdrop_url(backdrop_path)
        if backdrop_url:
            st.markdown(
                f'<div class="details-backdrop" style="background-image: url(\'{backdrop_url}\');"></div>',
                unsafe_allow_html=True,
            )

    # Title & Metadata
    title = merged.get("title", "Untitled")
    year = merged.get("release_year") or str(merged.get("release_date", ""))[:4] or "????"
    vote_avg = merged.get("vote_average", 0.0)
    vote_count = merged.get("vote_count", 0)
    runtime = merged.get("runtime")
    director = merged.get("director", "")
    cast = merged.get("cast", [])
    genres = merged.get("genres", [])
    overview = merged.get("overview", "")
    tagline = merged.get("tagline", "")

    m_col1, m_col2 = st.columns([1, 3])

    with m_col1:
        poster_path = merged.get("poster_path")
        poster_url = tmdb_api.get_poster_url(poster_path) if poster_path else None
        if poster_url:
            st.image(poster_url, use_container_width=True)
        else:
            st.markdown(tmdb_api.render_poster_placeholder_html(title, height_px=380), unsafe_allow_html=True)

        is_bm = str(mid) in st.session_state.bookmarks
        if st.button("★ Remove Bookmark" if is_bm else "☆ Save to Bookmarks", use_container_width=True):
            if is_bm:
                st.session_state.bookmarks.pop(str(mid), None)
            else:
                st.session_state.bookmarks[str(mid)] = merged
            st.rerun()

        is_wl = str(mid) in st.session_state.watchlist
        if st.button("✓ In Watchlist" if is_wl else "＋ Add to Watchlist", use_container_width=True):
            if is_wl:
                st.session_state.watchlist.pop(str(mid), None)
            else:
                st.session_state.watchlist[str(mid)] = merged
            st.rerun()

        if st.button("🎯 Open in Recommender", type="primary", use_container_width=True):
            st.session_state.rec_seed_id = mid
            st.session_state.page = "Recommendations"
            st.rerun()

    with m_col2:
        st.title(title)
        if tagline:
            st.markdown(f'<div style="font-style: italic; color: #94a3b8; font-size: 16px; margin-bottom: 12px;">"{html.escape(tagline)}"</div>', unsafe_allow_html=True)

        meta_parts = [f"⭐ {vote_avg:.1f} / 10 ({vote_count:,} votes)", str(year)]
        if runtime:
            meta_parts.append(f"{runtime} min")
        st.markdown(f'<div style="font-size: 14px; font-weight: 600; color: #f59e0b; margin-bottom: 14px;">{" · ".join(meta_parts)}</div>', unsafe_allow_html=True)

        # Genres
        if genres:
            genre_pills = "".join(f'<span class="genre-pill" style="font-size:13px; padding: 4px 12px;">{html.escape(str(g))}</span>' for g in genres)
            st.markdown(f'<div style="margin-bottom: 16px;">{genre_pills}</div>', unsafe_allow_html=True)

        # Overview
        st.subheader("Overview")
        st.write(overview or "No synopsis available for this title.")

        # Cast & Crew
        c_crew1, c_crew2 = st.columns(2)
        with c_crew1:
            if director:
                st.markdown(f"**Director:** {html.escape(director)}")
        with c_crew2:
            if cast:
                st.markdown(f"**Top Cast:** {html.escape(', '.join(str(c) for c in cast[:5]))}")

    # Similar & Recommended Movies section
    st.divider()
    st.subheader(f"More Like '{title}'")
    st.caption("AI-generated recommendations matching plot semantics, genres, and audience tastes.")

    similar_movies, _ = engine.compute_hybrid_recommendations(movie_id=mid, top_n=10)
    if similar_movies:
        render_movie_grid(similar_movies, columns=5, section="details_sim")
    else:
        st.info("No similar movies available for this title.")


# ==========================================================
# 10. VIEW: WATCHLIST & BOOKMARKS
# ==========================================================

def render_saved_collection_page(title: str, collection_key: str):
    st.title(f"🔖 {title}")
    st.caption(f"Your personalized {title.lower()} saved during this session.")

    collection = getattr(st.session_state, collection_key, {})
    movies = list(collection.values())

    if not movies:
        st.markdown(
            f"""
            <div style="padding: 50px; text-align: center; border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 12px; margin: 20px 0;">
                <div style="font-size: 36px; margin-bottom: 8px;">📑</div>
                <div style="font-size: 16px; font-weight: 600; color: #cbd5e1;">Your {title.lower()} is currently empty</div>
                <div style="font-size: 13px; color: #64748b; margin-top: 4px;">Click the bookmark or watchlist button on any movie card to save it here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    c_stat1, c_stat2, c_stat3 = st.columns(3)
    c_stat1.metric("Saved Movies", len(movies))
    avg_r = sum(m.get("vote_average", 0.0) for m in movies) / len(movies)
    c_stat2.metric("Average Rating", f"⭐ {avg_r:.1f}")
    if st.button(f"Clear All {title}", use_container_width=False):
        collection.clear()
        st.rerun()

    st.divider()
    render_movie_grid(movies, columns=5, section=f"saved_{collection_key}")


# ==========================================================
# 11. VIEW: MODEL DIAGNOSTICS & EVALUATION
# ==========================================================

def render_diagnostics_page():
    st.title("📊 CineMatch Model Intelligence & Diagnostics")
    st.caption("Architecture, coverage statistics, and algorithmic verification.")

    stats = engine.get_engine_diagnostics()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Catalog Movies", f"{stats['total_movies']:,}")
    m2.metric("TF-IDF Features", f"{stats['tfidf_features']:,}")
    m3.metric("Collaborative Factor Coverage", f"{stats['collab_movies']:,} titles")
    m4.metric("Latent Dimensions (SVD)", f"{stats['latent_dimensions']} dims")

    st.divider()

    st.subheader("Model Architecture Specifications")
    arch_data = [
        {"Component": "Content Vectorizer", "Algorithm": "TfidfVectorizer (n-grams 1-2, sublinear tf)", "Dimensions": "50,000 sparse features", "Status": "Active & Vectorized"},
        {"Component": "Collaborative Model", "Algorithm": "TruncatedSVD Matrix Factorization", "Dimensions": "50 latent components", "Status": "Active & Pre-normalized"},
        {"Component": "Hybrid Ranker", "Algorithm": "Weighted Linear Interpolation + Cold-Start Fallback", "Weights": "Default 60% Content / 40% Collab", "Status": "Operational (0.08s latency)"},
        {"Component": "User Taste Embeddings", "Algorithm": "MovieLens User Vectors (265,917 profiles)", "Dimensions": "50 latent preferences", "Status": "Loaded On-Demand"},
        {"Component": "External Discovery", "Algorithm": "The Movie Database (TMDB) REST v3", "Cache TTL": "1,800 seconds (Safe Fallback)", "Status": "Connected" if tmdb_api.is_tmdb_available() else "Offline Fallback"},
    ]
    st.dataframe(pd.DataFrame(arch_data), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Algorithmic Methodology & Integrity")
    st.markdown(
        """
        - **Deterministic Ranking**: Cosine similarities and dot products are deterministic and reproducible.
        - **Self-Exclusion Guarantee**: Queried seed movies are strictly masked out (`score = -1.0`) to avoid self-recommendation.
        - **Cold-Start Resilience**: 7,565 movies have high-confidence collaborative factors. When querying a title outside this subset, the engine automatically falls back to full 50,000-dimensional TF-IDF content similarity without dropping recommendations.
        - **Real-Time Performance**: Vectorized matrix calculations execute in under 100 milliseconds without reloading expensive model pickles.
        """
    )


# ==========================================================
# 12. VIEW: SETTINGS
# ==========================================================

def render_settings_page():
    st.title("⚙️ Settings & System Configuration")
    st.caption("Manage API keys, cache, and diagnostic preferences.")

    with st.container(border=True):
        st.subheader("The Movie Database (TMDB) API Configuration")
        st.write(
            "An API key allows CineMatch to fetch live trending movies, backdrop artwork, and real-time cast photos. "
            "If no key is provided, the application runs seamlessly from the local 46,625 movie catalog."
        )

        current_key = tmdb_api.get_api_key()
        masked_key = (current_key[:4] + "..." + current_key[-4:]) if len(current_key) > 8 else ("Configured" if current_key else "Not Set")

        st.markdown(f"**Current API Key Status:** `{'✓ Active (' + masked_key + ')' if current_key else '✗ Inactive (Using Local Catalog)'}`")

        new_key = st.text_input(
            "Override TMDB API Key",
            value=st.session_state.get("tmdb_api_key_override", ""),
            type="password",
            placeholder="Enter custom TMDB v3 API Key...",
        )

        if st.button("Save API Key Override"):
            st.session_state.tmdb_api_key_override = new_key.strip()
            st.success("API key updated successfully!")
            st.rerun()

    with st.container(border=True):
        st.subheader("Performance & Cache Management")
        st.write("Streamlit caches recommendation matrix computations and TMDB requests to deliver sub-second responses.")

        if st.button("Clear Application Cache", type="primary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared successfully!")
            st.rerun()


# ==========================================================
# 13. SIDEBAR NAVIGATION
# ==========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="font-size: 28px;">🎬</span>
            <span style="font-size: 22px; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">CineMatch</span>
        </div>
        <div style="font-size: 12px; color: #64748b; margin-bottom: 18px;">AI-Powered Cinematic Discovery</div>
        """,
        unsafe_allow_html=True,
    )

    # API & Catalog Status Pill
    tmdb_online = tmdb_api.is_tmdb_available()
    status_color = "#10b981" if tmdb_online else "#f59e0b"
    status_label = "Live API Active" if tmdb_online else "Local Catalog Mode"
    st.markdown(
        f"""
        <div style="display: inline-flex; align-items: center; gap: 6px; background: rgba(255,255,255,0.05); padding: 4px 10px; border-radius: 6px; font-size: 11px; margin-bottom: 14px;">
            <span style="width: 8px; height: 8px; border-radius: 50%; background-color: {status_color};"></span>
            <span style="color: #cbd5e1;">{status_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_pages = [
        ("⌂ Home", "Home"),
        ("🧭 Discover", "Discover"),
        ("🔍 Search", "Search"),
        ("🎯 Recommendations", "Recommendations"),
        ("🧬 Movie DNA", "Movie DNA"),
        (f"★ Saved ({len(st.session_state.bookmarks)})", "Bookmarks"),
        (f"＋ Watchlist ({len(st.session_state.watchlist)})", "Watchlist"),
        ("📊 Intelligence", "Model Diagnostics"),
        ("⚙️ Settings", "Settings"),
    ]

    for label, page_key in nav_pages:
        is_active = st.session_state.page == page_key
        if st.button(
            label,
            key=f"nav_{page_key}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = page_key
            st.rerun()

    st.divider()
    st.caption("CineMatch Recommender · Portfolio Edition<br>46,625 Catalog Titles · TruncatedSVD + TF-IDF", unsafe_allow_html=True)


# ==========================================================
# 14. ROUTER
# ==========================================================

current_page = st.session_state.page

if current_page == "Home":
    render_home_page()
elif current_page == "Discover":
    render_discover_page()
elif current_page == "Search":
    render_search_page()
elif current_page == "Recommendations":
    render_recommendations_page()
elif current_page == "Movie DNA":
    render_movie_dna_page()
elif current_page == "Bookmarks":
    render_saved_collection_page("Saved Movies", "bookmarks")
elif current_page == "Watchlist":
    render_saved_collection_page("Watchlist", "watchlist")
elif current_page == "Movie Details":
    render_movie_details_page()
elif current_page == "Model Diagnostics":
    render_diagnostics_page()
elif current_page == "Settings":
    render_settings_page()
else:
    render_home_page()
