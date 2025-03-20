import os
import requests
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

app = Flask(__name__)

# ✅ Fetch Movies & TV Shows from TMDb API
def fetch_tmdb_data(content_type="movie", page=1):
    url = f"https://api.themoviedb.org/3/discover/{content_type}?api_key={TMDB_API_KEY}&language=en-US&page={page}"
    response = requests.get(url)
    data = response.json()
    return data.get("results", [])

# ✅ Get full details (genres, cast, director) for a given movie/TV show
def get_detailed_info(tmdb_id, content_type="movie"):
    url = f"https://api.themoviedb.org/3/{content_type}/{tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=credits"
    response = requests.get(url)
    return response.json()

# ✅ Load Movies & TV Shows from TMDb (First 5 Pages)
def load_tmdb_data():
    all_data = []
    for page in range(1, 6):  # Load 5 pages for more data
        movies = fetch_tmdb_data("movie", page)
        shows = fetch_tmdb_data("tv", page)
        all_data.extend(movies + shows)
    return all_data

# ✅ Fetch genre names from TMDb
def get_genre_mapping():
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US"
    response = requests.get(url).json()
    movie_genres = {g["id"]: g["name"] for g in response.get("genres", [])}

    url_tv = f"https://api.themoviedb.org/3/genre/tv/list?api_key={TMDB_API_KEY}&language=en-US"
    response_tv = requests.get(url_tv).json()
    tv_genres = {g["id"]: g["name"] for g in response_tv.get("genres", [])}

    return {**movie_genres, **tv_genres}  # Merge both movie & TV show genres

# ✅ Get genre mapping before loading content
GENRE_MAPPING = get_genre_mapping()

# ✅ Process TMDb Data into a Pandas DataFrame
def prepare_dataframe():
    tmdb_data = load_tmdb_data()
    processed_data = []
    
    for item in tmdb_data:
        tmdb_id = item["id"]
        title = item.get("title") or item.get("name")  # TV shows use 'name'
        genres = ", ".join([GENRE_MAPPING.get(genre_id, "Unknown") for genre_id in item.get("genre_ids", [])])  # ✅ FIXED
        content_type = "TV Show" if "name" in item else "Movie"

        # Get full metadata (cast, director, genres)
        details = get_detailed_info(tmdb_id, "tv" if content_type == "TV Show" else "movie")
        cast = ", ".join([c["name"] for c in details.get("credits", {}).get("cast", [])[:5]])
        director = ", ".join([c["name"] for c in details.get("credits", {}).get("crew", []) if c["job"] == "Director"])

        processed_data.append({"tmdb_id": tmdb_id, "title": title, "genres": genres, "cast": cast, "director": director, "type": content_type})

    return pd.DataFrame(processed_data)

# ✅ Prepare Data
movies_df = prepare_dataframe()

# ✅ Content-Based Filtering (Genres, Cast, Director)
movies_df["metadata"] = (
    movies_df["genres"].fillna("") + " " +
    movies_df["cast"].fillna("") + " " +
    movies_df["director"].fillna("")
)
vectorizer = TfidfVectorizer(stop_words="english")
content_matrix = vectorizer.fit_transform(movies_df["metadata"])
content_similarity = cosine_similarity(content_matrix)
content_similarity_df = pd.DataFrame(content_similarity, index=movies_df["title"], columns=movies_df["title"])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_ratings = request.json.get("ratings", {})

        # ✅ Normalize ratings from 1-100 scale to 0.5-5
        user_movies = {title: (float(rating) / 100) * 5 for title, rating in user_ratings.items() if title in content_similarity_df.index}

        recommendations = {}

        for movie, rating in user_movies.items():
            similar_movies_cb = content_similarity_df[movie].sort_values(ascending=False)[1:11]  # Content-Based

            for sim_movie, score in similar_movies_cb.items():
                recommendations[sim_movie] = recommendations.get(sim_movie, 0) + score * (rating / 5)

        # ✅ Sort recommendations by score
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:10]

        return jsonify({"recommendations": [movie for movie, _ in sorted_recommendations]})

    return render_template("index.html")

@app.route("/search")
def search_movies():
    query = request.args.get("query", "").lower()
    matching_movies = [movie for movie in movies_df["title"] if query in movie.lower()]
    return jsonify({"movies": matching_movies[:10]})  # Return top 10 matches

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
