# Movie-recommendation-system
Streamlit-based movie recommendation system. Features an automated TMDb API pipeline, rigorous data cleaning, normalization, and an interactive EDA dashboard. Leverages Content-Based Filtering (Cosine Similarity) to suggest top similar films based on genres, ratings, and plot keywords, complete with detailed descriptions.
##  Project Overview

This project spans the complete data science lifecycle—from raw API data ingestion to user-facing deployment. The system recommends the top films similar to a user's chosen title by calculating mathematical similarities across genres, user ratings, and plot keywords.

The project is split into two primary components:
1. **Data Engineering & EDA Pipeline (`FINAL PROJECT (2).ipynb`):** An production-grade notebook that handles data collection via the TMDb API, missing value resolution, column normalization, detailed statistical analysis, and saves data to a structured SQLite database.
2. **Interactive Streamlit Web App (`fproject.py`):** A clean, responsive frontend application allowing users to search for any movie, explore metadata, and receive instant, detailed recommendations for similar films.


##  Tech Stack & Libraries

### 🔹 Data Ingestion & Storage
* `requests`, `json`, `time` — Automating multi-page metadata extraction from the TMDb API with built-in rate-limit handling.
* `sqlite3` — Storing the engineered dataset into a structured SQL database (`movies.db`).

### 🔹 Preprocessing & Machine Learning
* `pandas` & `numpy` — Core data manipulation, deduplication, and feature structuring.
* `scikit-learn (sklearn)`:
  * `MinMaxScaler` — Standardizing skewed numerical columns (e.g., user ratings, vote counts).
  * `TfidfVectorizer` — Converting text plot summaries into dense mathematical vectors.
  * `cosine_similarity` — The backbone algorithm used to calculate structural distances between films.

### 🔹 Visualization & UI Deployment
* `matplotlib` & `seaborn` — Generating comprehensive charts, distribution graphs, and correlation heatmaps during EDA.
* `streamlit` — Powering the production web application interface.


##  Data Pipeline & Core Methodology

### 1. Data Collection & Cleaning
* Dynamically extracted a dataset spanning over **3,800+ films** using the TMDb API.
* Implemented strict data sanitization: dropped duplicate items, handled nested JSON structures, resolved missing descriptions, and encoded multi-categorical data.

### 2. Exploratory Data Analysis (EDA)
* Conducted deep-dive statistical analysis tracking the correlation between movie budgets, release vintages, voting volumes, and critical reception scores.
* Created visualizations mapping out genre frequencies and average viewer rating trends over time.

### 3. Recommendation Engine (Predictive Modeling)
* **Approach:** Content-Based Filtering.
* The system builds a unified feature vector combining normalized numeric metrics and textual plot features. 
* By running **Cosine Similarity** across these vectors, the engine precisely ranks and fetches mathematically adjacent films to form the recommendation pool.


## Streamlit Interface Flow

When running the web app via `fproject.py`:
* **Search & Explore:** An intuitive dropdown/search bar lets you pick a target movie. 
* **Target Details:** The app instantly renders the selected film's details: full plot overview, release year, genres, and normalized audience metrics.
* **Smart Recommendations:** Directly beneath, the UI dynamically generates a dedicated section showcasing the **Top Similar Movies**, complete with matching criteria and comprehensive summaries for each recommended title.
