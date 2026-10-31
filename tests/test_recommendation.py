"""
CineMatch Automated Test Suite
------------------------------
Comprehensive unit and integration tests verifying:
- Model loading and schema validation
- Vectorized content-based and collaborative similarity
- Hybrid recommendation ranking & cold-start fallbacks
- Self-exclusion guarantee
- Search, filter, and pagination logic
- Error handling and edge cases
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Add project directory to sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import engine
import tmdb_api


class TestModelLoading(unittest.TestCase):
    """Verify model files and data structure integrity."""

    def test_core_models_exist_and_load(self):
        models = engine.load_core_models()
        self.assertIn("movies", models)
        self.assertIn("tfidf_matrix", models)
        self.assertIn("movie_factors", models)
        self.assertIn("id_to_row", models)
        self.assertIn("title_to_rows", models)

        movies = models["movies"]
        self.assertGreater(len(movies), 40000)
        self.assertIn("id", movies.columns)
        self.assertIn("title", movies.columns)
        self.assertIn("popularity", movies.columns)
        self.assertIn("vote_average", movies.columns)
        self.assertIn("release_year", movies.columns)

        import numpy as np
        self.assertTrue(issubclass(movies["popularity"].dtype.type, (float, np.floating, int, np.integer)))
        self.assertTrue(issubclass(movies["vote_average"].dtype.type, (float, np.floating, int, np.integer)))
        self.assertTrue(issubclass(movies["release_year"].dtype.type, (int, np.integer)))

    def test_engine_diagnostics(self):
        diag = engine.get_engine_diagnostics()
        self.assertGreater(diag["total_movies"], 40000)
        self.assertEqual(diag["tfidf_features"], 50000)
        self.assertEqual(diag["latent_dimensions"], 50)
        self.assertGreater(diag["collab_movies"], 7000)
        self.assertGreater(diag["catalog_coverage_pct"], 10.0)


class TestSearchAndFilters(unittest.TestCase):
    """Test movie lookup and catalog filtering."""

    def test_search_exact_and_partial(self):
        # Exact match
        exact_results = engine.search_local_catalog("Toy Story", limit=5)
        self.assertTrue(len(exact_results) > 0)
        self.assertEqual(exact_results[0]["title"], "Toy Story")

        # Partial match
        part_results = engine.search_local_catalog("interstell", limit=5)
        self.assertTrue(len(part_results) > 0)
        titles = [m["title"].lower() for m in part_results]
        self.assertTrue(any("interstellar" in t for t in titles))

        # Empty and non-existent
        self.assertEqual(engine.search_local_catalog(""), [])
        self.assertEqual(engine.search_local_catalog("xyznonexistentmovie99999"), [])

    def test_filter_catalog(self):
        # Genre filter
        results, count = engine.filter_local_catalog(
            genre="Animation",
            year_range=(1990, 2010),
            min_rating=6.0,
            sort_by="Rating",
            page=1,
            per_page=10,
        )
        self.assertTrue(len(results) > 0)
        self.assertTrue(count > 0)
        for m in results:
            self.assertIn("Animation", m["genres"])
            self.assertGreaterEqual(m["vote_average"], 6.0)
            self.assertGreaterEqual(m["release_year"], 1990)
            self.assertLessEqual(m["release_year"], 2010)

    def test_curated_collections(self):
        for cat in ["Trending", "Popular", "Highest Rated", "Action & Adventure", "Sci-Fi & Fantasy"]:
            recs = engine.get_curated_collection(cat, limit=5)
            self.assertEqual(len(recs), 5, f"Category '{cat}' should return 5 movies")
            for r in recs:
                self.assertIn("id", r)
                self.assertIn("title", r)
                self.assertIn("vote_average", r)


class TestRecommendationEngine(unittest.TestCase):
    """Test core hybrid, content, and collaborative recommendation pipelines."""

    def test_hybrid_recommendation_toy_story(self):
        # Seed: Toy Story (ID: 862) has both content and collaborative factors
        recs, mode = engine.compute_hybrid_recommendations(movie_id=862, top_n=5, w_content=0.6, w_collab=0.4)
        self.assertEqual(len(recs), 5)
        self.assertIn("Hybrid", mode)

        rec_ids = [r["id"] for r in recs]
        # Self-exclusion: 862 must NOT be in recommendations
        self.assertNotIn(862, rec_ids)

        # Expected top recommendations for Toy Story include Toy Story 2 or 3
        rec_titles = [r["title"] for r in recs]
        has_sequel = any("Toy Story" in t for t in rec_titles)
        self.assertTrue(has_sequel, f"Expected sequel in recommendations for Toy Story, got: {rec_titles}")

        # Check finite score bounds
        for r in recs:
            self.assertGreaterEqual(r["hybrid_score"], 0.0)
            self.assertLessEqual(r["hybrid_score"], 1.0)
            self.assertGreater(len(r["explanation"]), 0)

    def test_hybrid_recommendation_dark_knight(self):
        # Seed: The Dark Knight (ID: 155)
        recs, mode = engine.compute_hybrid_recommendations(movie_id=155, top_n=5, w_content=0.6, w_collab=0.4)
        self.assertEqual(len(recs), 5)
        self.assertNotIn(155, [r["id"] for r in recs])

        rec_titles = [r["title"] for r in recs]
        # Must recommend Batman Begins or other Batman films
        has_batman = any("Batman" in t for t in rec_titles)
        self.assertTrue(has_batman, f"Expected Batman title in recommendations for The Dark Knight, got: {rec_titles}")

    def test_cold_start_movie_fallback(self):
        # Seed: Toy Story 2 (ID: 863) has content vector but no collaborative factors (cold-start)
        recs, mode = engine.compute_hybrid_recommendations(movie_id=863, top_n=5)
        self.assertEqual(len(recs), 5)
        self.assertIn("Content-Based", mode)
        self.assertNotIn(863, [r["id"] for r in recs])

        # Expected to recommend Toy Story 3 or Toy Story
        rec_titles = [r["title"] for r in recs]
        has_toy_story = any("Toy Story" in t for t in rec_titles)
        self.assertTrue(has_toy_story, f"Expected Toy Story title in recommendations for Toy Story 2, got: {rec_titles}")

    def test_invalid_movie_id(self):
        recs, mode = engine.compute_hybrid_recommendations(movie_id=-9999, top_n=5)
        self.assertEqual(recs, [])
        self.assertIn("not found", mode.lower())

    def test_side_by_side_comparison(self):
        res = engine.get_side_by_side_comparison(movie_id=862, top_n=5)
        self.assertIn("content", res)
        self.assertIn("collaborative", res)
        self.assertIn("hybrid", res)
        self.assertEqual(len(res["content"]), 5)
        self.assertEqual(len(res["collaborative"]), 5)
        self.assertEqual(len(res["hybrid"]), 5)


class TestUserCollaborativeProfile(unittest.TestCase):
    """Test MovieLens user taste vector recommendations."""

    def test_user_recommendations(self):
        recs, info = engine.get_user_profile_recommendations(user_id=1, top_n=5)
        self.assertTrue(len(recs) > 0)
        self.assertIn("User 1", info)
        for r in recs:
            self.assertIn("id", r)
            self.assertIn("title", r)

    def test_invalid_user_id(self):
        recs, info = engine.get_user_profile_recommendations(user_id=99999999, top_n=5)
        self.assertEqual(recs, [])
        self.assertIn("not found", info.lower())


class TestTMDBIntegration(unittest.TestCase):
    """Test API helper utilities and poster fallbacks."""

    def test_poster_url_generation(self):
        url = tmdb_api.get_poster_url("/test_poster.jpg")
        self.assertEqual(url, "https://image.tmdb.org/t/p/w500/test_poster.jpg")
        self.assertIsNone(tmdb_api.get_poster_url(None))
        self.assertIsNone(tmdb_api.get_poster_url(""))

    def test_svg_poster_placeholder(self):
        html_code = tmdb_api.render_poster_placeholder_html("The Dark Knight")
        self.assertIn("The Dark Knight", html_code)
        self.assertIn("No Poster Available", html_code)


if __name__ == "__main__":
    unittest.main()
