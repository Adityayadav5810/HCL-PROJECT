# 🎬 CineMatch · AI Movie Discovery & Hybrid Recommendation System

A modern, production-grade movie discovery and recommendation platform built with Python, Scikit-Learn, and Streamlit. CineMatch blends **Content-Based Semantic Filtering** with **TruncatedSVD Collaborative Filtering** over a catalog of **46,625 movies** and **265,917 user profiles**.

---

## 📌 Features

- **🎯 Hybrid Recommender Engine**:
  - Dynamically blends Content-Based Similarity (TF-IDF over plot, genres, director, cast, and keywords) with Collaborative Latent Factors (TruncatedSVD on MovieLens co-ratings).
  - Interactive weight tuning slider ($0\%$ to $100\%$ Content vs. Collaborative).
  - Side-by-side comparison matrix (Content vs. Collaborative vs. Hybrid).
  - Clear algorithmic explanations for each recommendation (e.g. shared genres, audience co-rating patterns, plot affinity).
  - Built-in self-exclusion guarantee and graceful cold-start fallback.
- **🧭 Dynamic Catalog Discovery**:
  - Filter by Genre, Release Year Range, Minimum Rating, and Sort Order (Popularity, Rating, Release Date, Vote Count).
  - Fast pagination across 46,625 catalog titles.
- **🔍 Instant Movie Search**:
  - Exact, prefix, and partial keyword matching with sub-second response times.
- **🧬 Movie DNA (User Taste Profiling)**:
  - Explores learned user preference vectors across 265,917 real MovieLens user profiles.
- **📑 Bookmarks & Watchlist**:
  - Save movies to Bookmarks and Watch Later during your browsing session.
- **📊 Model Intelligence Dashboard**:
  - Live diagnostics covering model architecture, matrix dimensions, catalog coverage, and latent factors.
- **🎨 Premium Cinematic UI**:
  - Dark cinematic theme (obsidian and slate palette with electric cyan accents).
  - Responsive movie cards with 2:3 aspect ratio posters and SVG fallback placeholders.
  - Rich Movie Details view with backdrop artwork, runtime, vote stats, director & cast chips, and synopsis.
- **🌐 Resilient TMDB API Integration**:
  - Fetches live trending movies, backdrop artwork, and real-time metadata when connected.
  - Automatically falls back to the local 46,625 catalog when offline or in restricted environments.

---

## 🛠️ Architecture & Technologies

- **Python 3.11+**
- **Streamlit** — Web interface and reactive state management
- **Scikit-Learn & Scipy** — TF-IDF vectorization, sparse matrix operations, and TruncatedSVD
- **NumPy & Pandas** — Vectorized numerical operations and data processing
- **Joblib** — High-performance model serialization
- **Requests** — TMDB API v3 integration with resilient caching and fast timeout

### Model Architecture

```text
                                 [ Seed Movie / User ID ]
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
         [ Content-Based Pipeline ]                      [ Collaborative Pipeline ]
     50,000-feature TF-IDF Sparse Matrix              50-dimensional SVD Latent Factors
     Plot, Genres, Keywords, Cast, Director             MovieLens User-Movie Co-ratings
                     │                                               │
             Cosine Similarity                               Latent Factor Alignment
                     │                                               │
                     └───────────────────────┬───────────────────────┘
                                             ▼
                             [ Weighted Hybrid Interpolation ]
                                  w_c * Content + w_cf * CF
                                             │
                                     [ Self-Masking ]
                                (Exclude seed movie: -1.0)
                                             │
                                  [ Cold-Start Fallback ]
                           (Pure content when CF unavailable)
                                             │
                                             ▼
                               [ Top-N Ranked Recommendations ]
```

---

## 📁 Project Structure

```text
Movie-Recommendation-System/
│
├── .streamlit/
│   ├── config.toml           # Streamlit theme and server configuration
│   ├── credentials.toml      # Headless credentials (bypasses email prompt)
│   └── secrets.toml          # TMDB API key configuration (git-ignored)
│
├── models/
│   ├── movies_model.pkl      # 46,625 movie records with rich metadata
│   ├── tfidf_matrix.pkl      # 46,625 x 50,000 sparse TF-IDF feature matrix
│   ├── movie_factors.pkl     # 7,565 x 50 SVD item latent factor matrix
│   ├── movie_to_idx.pkl      # TMDB ID to latent factor row mapping
│   ├── svd_model.pkl         # Trained TruncatedSVD decomposition model
│   ├── user_factors.pkl      # 265,917 x 50 user latent factor matrix
│   ├── user_to_idx.pkl       # User ID to latent factor row mapping
│   └── user_rated_movies.pkl # User historical rating sets
│
├── tests/
│   ├── test_recommendation.py # 14 comprehensive unit & integration tests
│   └── verify_startup.py      # Automated server startup and health check probe
│
├── app.py                    # Main Streamlit application entry point
├── engine.py                 # Core recommendation engine and catalog queries
├── styles.py                 # Dark cinematic CSS styling system
├── tmdb_api.py               # TMDB API client with fast offline fallback
├── links.csv                 # MovieLens to TMDB ID mappings
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Installation & Application Startup

### 1. Prerequisites

Ensure **Python 3.10+** (Python 3.11 recommended) is installed on your system.

### 2. Install Dependencies

Open Windows PowerShell in the project directory and install the required packages:

```powershell
pip install -r requirements.txt
```

### 3. Configure TMDB API Key (Optional)

The application runs completely offline using the bundled 46,625 movie catalog. To enable live trending titles and backdrop images from TMDB:

1. Create or edit `.streamlit/secrets.toml`:
   ```toml
   TMDB_API_KEY = "your_tmdb_api_key_here"
   ```
2. Or set an environment variable:
   ```powershell
   $env:TMDB_API_KEY = "your_tmdb_api_key_here"
   ```
3. Or enter your key directly inside the app's **⚙️ Settings** page.

### 4. Launch the Application

Run the application using the Streamlit CLI:

```powershell
# From inside the Movie-Recommendation-System directory:
python -m streamlit run app.py

# Or from the parent directory:
python -m streamlit run Movie-Recommendation-System/app.py
```

Once started, open your web browser at:
👉 **`http://localhost:8501`**

---

## 🧪 Running Automated Tests

To run the unit and integration test suite:

```powershell
python -m unittest tests/test_recommendation.py
```

To run the automated server startup and HTTP health verification:

```powershell
python tests/verify_startup.py
```

Expected output:
```text
Ran 14 tests in ~12s
OK
SUCCESS: Streamlit health endpoint responded 200 OK!
Startup verification result: PASSED
```

---

## ⚡ Performance Optimizations

1. **Vectorized Matrix Computations**: Replaced slow item-by-item Python loops with vectorized NumPy and Scipy operations, dropping recommendation latency to **under 80ms**.
2. **On-Demand Lazy Loading**: Large user matrices (95MB) are loaded on-demand only when accessing the Movie DNA tab, reducing initial startup time from **25s down to ~5.8s**.
3. **Resilient Network Timeouts**: Fast 1.0s connectivity check prevents the application from hanging when offline or sandboxed.
4. **Data Type Hygiene**: Fixed string popularity sorting and array ambiguity errors when parsing genres and cast lists.

---

## 📋 Known Limitations

- **Historical Dataset Range**: The bundled catalog metadata includes films up to 2017. For 2018–2026 releases, active TMDB API connectivity is utilized.
- **Collaborative Factor Subset**: The TruncatedSVD item factors cover the 7,565 most-rated catalog titles to preserve model quality. Titles outside this subset automatically use 50,000-feature TF-IDF content similarity.
- **Session-Based Storage**: Bookmarks and Watchlists are stored in Streamlit `session_state` per browser session.
