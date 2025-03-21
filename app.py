import os
import requests
import pandas as pd
import re
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv

# ✅ Load TMDb API Key
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

app = Flask(__name__)

# ✅ Load MovieLens Ratings & Movies
ratings_df = pd.read_csv("ratings.csv")
movies_df = pd.read_csv("movies.csv")

# ✅ Prevent merging movies with the same title
ratings_df = ratings_df.groupby(["userId", "movieId"], as_index=False).rating.mean()

# ✅ Merge movieId back to get correct movie titles
ratings_df = ratings_df.merge(movies_df[["movieId", "title"]], on="movieId", how="left")

# ✅ Limit dataset size to optimize performance (top 50,000 users & movies)
top_users = ratings_df["userId"].value_counts().head(50000).index
top_movies = ratings_df["movieId"].value_counts().head(50000).index
filtered_ratings = ratings_df[(ratings_df["userId"].isin(top_users)) & (ratings_df["movieId"].isin(top_movies))]

# ✅ Ensure movies with the same title but different years stay separate
filtered_ratings = ratings_df.groupby(["userId", "movieId"], as_index=False).rating.mean()

# ✅ Convert ratings into a sparse matrix (No pivot table!)
user_ids = filtered_ratings["userId"].astype("category").cat.codes
movie_ids = filtered_ratings["movieId"].astype("category").cat.codes
ratings = filtered_ratings["rating"].values

user_movie_ratings_sparse = coo_matrix((ratings, (user_ids, movie_ids))).tocsr()

# ✅ Create a mapping of movieId -> internal index
unique_movie_ids = filtered_ratings["movieId"].astype("category")
movie_index_to_id = dict(enumerate(unique_movie_ids.cat.categories))
id_to_movie_index = {v: k for k, v in movie_index_to_id.items()}  # Reverse lookup

# ✅ Use Approximate Nearest Neighbors (KNN) instead of Full Cosine Similarity
knn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=10, n_jobs=-1)
knn_model.fit(user_movie_ratings_sparse.T)

def get_similar_movies(movie_id, num_neighbors=10):
    """Finds similar movies using Approximate Nearest Neighbors."""
    if movie_id not in id_to_movie_index:
        return []
    
    movie_index = id_to_movie_index[movie_id]
    distances, indices = knn_model.kneighbors(user_movie_ratings_sparse.T[movie_index], n_neighbors=num_neighbors)
    
    similar_movies = [movie_index_to_id[idx] for idx in indices.flatten()[1:]]  # Skip first (itself)
    return similar_movies

# ✅ Fetch Genre Mapping from TMDb API
def get_genre_mapping():
    """Fetches genre ID-to-name mapping from TMDb API."""
    url_movie = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US"
    url_tv = f"https://api.themoviedb.org/3/genre/tv/list?api_key={TMDB_API_KEY}&language=en-US"

    movie_genres = requests.get(url_movie).json().get("genres", [])
    tv_genres = requests.get(url_tv).json().get("genres", [])

    return {g["id"]: g["name"] for g in movie_genres + tv_genres}

GENRE_MAPPING = get_genre_mapping()

# ✅ Remove Year from Titles (Fixes TMDb Search)
def clean_title(title):
    """Removes release year from a title (e.g., 'The Avengers (2012)' → 'The Avengers')"""
    return re.sub(r'\(\d{4}\)', '', title).strip()

# ✅ Search MovieLens Dataset Before TMDb
def search_movie_lens(title):
    """Search MovieLens dataset dynamically instead of preloading."""
    for chunk in pd.read_csv("movies.csv", chunksize=10000):  # Load in chunks
        matched_movies = chunk[chunk["title"].str.contains(title, case=False, na=False)]
        if not matched_movies.empty:
            return matched_movies.iloc[0]  # Return first match
    return None  # If not found

# ✅ Search TMDb (Falls Back if MovieLens Doesn’t Find It)
def search_tmdb(query):
    """First checks MovieLens, then queries TMDb if needed."""
    cleaned_query = clean_title(query)

    # ✅ Try Finding in MovieLens First
    matched_movie = search_movie_lens(cleaned_query)
    if matched_movie is not None:
        return [{
            "tmdb_id": matched_movie["movieId"],
            "title": matched_movie["title"],
            "genres": matched_movie["genres"],
            "type": "movie"
        }]

    # ✅ If Not in MovieLens, Query TMDb API
    url = f"https://api.themoviedb.org/3/search/multi?api_key={TMDB_API_KEY}&query={cleaned_query}&language=en-US"
    response = requests.get(url).json()
    results = response.get("results", [])

    valid_results = []
    for item in results:
        if item.get("media_type") in ["movie", "tv"]:
            title = item.get("title") or item.get("name")
            release_date = item.get("release_date") or item.get("first_air_date", "Unknown")
            release_year = release_date.split("-")[0] if release_date and release_date != "Unknown" else "N/A"
            genres = ", ".join([GENRE_MAPPING.get(genre_id, "Unknown") for genre_id in item.get("genre_ids", [])])

            valid_results.append({
                "tmdb_id": item["id"],
                "title": f"{title} ({release_year})",
                "genres": genres,
                "type": item["media_type"]
            })

    return valid_results

# ✅ Fetch Movie/TV Details for Content-Based Filtering
def get_detailed_info(tmdb_id, content_type):
    """Fetch detailed metadata for a movie/TV show from TMDb."""
    url = f"https://api.themoviedb.org/3/{content_type}/{tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=credits"
    response = requests.get(url).json()

    cast = ", ".join([c["name"] for c in response.get("credits", {}).get("cast", [])[:5]])
    director = ", ".join([c["name"] for c in response.get("credits", {}).get("crew", []) if c["job"] == "Director"])
    genres = ", ".join([g["name"] for g in response.get("genres", [])])

    return {
        "title": response.get("title") or response.get("name"),
        "genres": genres,
        "cast": cast,
        "director": director
    }

# ✅ Compute Content-Based Similarity
def compute_similarity(data):
    """Compute TF-IDF similarity between movies based on metadata."""
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    content_matrix = tfidf_vectorizer.fit_transform(data)
    return cosine_similarity(content_matrix)

@app.route("/search")
def search():
    query = request.args.get("query", "").strip().lower()
    if not query:
        return jsonify({"movies": []})

    # ✅ Search MovieLens first
    matching_movies = movies_df[movies_df["title"].str.lower().str.contains(query, na=False)]
    
    # ✅ Fall back to TMDb API if no results
    if matching_movies.empty:
        tmdb_results = search_tmdb(query)
        movies_list = [f"{movie['title']}" for movie in tmdb_results]  # Includes release year
    else:
        movies_list = matching_movies["title"].tolist()

    return jsonify({"movies": movies_list})

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_ratings = request.json.get("ratings", {})
        all_metadata = []
        recommendations = {}

        for title, rating in user_ratings.items():
            search_results = search_tmdb(title)
            if not search_results:
                continue

            tmdb_id = search_results[0]["tmdb_id"]
            content_type = search_results[0]["type"]
            details = get_detailed_info(tmdb_id, content_type)

            metadata = details["genres"] + " " + details["cast"] + " " + details["director"]
            all_metadata.append(metadata)

            # ✅ Create mappings between movie IDs and titles
            title_to_movie_id = dict(zip(movies_df["title"], movies_df["movieId"]))
            movie_id_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))


            # ✅ Collaborative Filtering (Weight = 60%)
            if title in title_to_movie_id:  # Convert title to movieId
                movie_id = title_to_movie_id[title]
                similar_movies_cf = get_similar_movies(movie_id, num_neighbors=10)  # Get similar movies

                for sim_movie_id in similar_movies_cf:
                    sim_movie_title = movie_id_to_title.get(sim_movie_id, "Unknown")
                    recommendations[sim_movie_title] = recommendations.get(sim_movie_title, 0) + (float(rating) / 100) * 0.6


        # ✅ Content-Based Filtering (Weight = 40%)
        if all_metadata:
            content_similarity = compute_similarity(all_metadata)
            for i, title in enumerate(user_ratings.keys()):
                for j, sim_score in enumerate(content_similarity[i]):
                    if i != j:
                        recommendations[list(user_ratings.keys())[j]] = recommendations.get(list(user_ratings.keys())[j], 0) + (sim_score * 0.4)

        # ✅ Sort and return top recommendations
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:10]
        return jsonify({"recommendations": [movie for movie, _ in sorted_recommendations]})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
