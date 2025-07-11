import os
import pandas as pd
import numpy as np
import json
import time
import sqlite3
from flask import Flask, request, jsonify, render_template, Response, session, redirect, g
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback_dev_key")
DATABASE = "users.db"

# ----------------- Database ------------------
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_title TEXT NOT NULL,
                rating REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        db.commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# ----------------- Auth Routes ------------------
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    password_hash = generate_password_hash(password)
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        db.commit()
        user_id = cursor.lastrowid
        session["user_id"] = user_id
        session["username"] = username
        return jsonify({"message": "User created and logged in", "redirect": "/menu"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already taken"}), 409

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    db = get_db()
    user = db.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,)).fetchone()
    if user and check_password_hash(user[1], password):
        session["user_id"] = user[0]
        session["username"] = username
        return jsonify({"message": "Logged in", "redirect": "/menu"})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/menu")
def menu():
    if "user_id" not in session:
        return redirect("/")
    return render_template("menu.html", username=session.get("username"))

@app.route("/recommend")
def recommend_page():
    if "user_id" not in session:
        return redirect("/")
    return render_template("index.html", username=session.get("username"))

@app.route("/one_off_recommendations")
def one_off_recommendations():
    if "user_id" not in session:
        return redirect("/")
    return render_template("one_off_recommendations.html", username=session.get("username"))

@app.route("/recommend_from_library")
def recommend_from_library():
    if "user_id" not in session:
        return redirect("/")
    return render_template("recommendations_from_library.html", username=session.get("username"))

@app.route("/library")
def library():
    if "user_id" not in session:
        return redirect("/")
    return render_template("library.html", username=session.get("username"))

@app.route("/recommendations_from_one_off")
def recommendations_from_one_off():
    if "user_id" not in session:
        return redirect("/")
    return render_template("recommendations_from_one_off.html", username=session.get("username"))

# ----------------- User Recommendations ------------------
@app.route("/user_recommendations", methods=["GET"])
def get_user_recommendations():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    db = get_db()
    recs = db.execute("SELECT id, movie_title, rating FROM recommendations WHERE user_id = ? ORDER BY timestamp DESC", (session["user_id"],)).fetchall()
    return jsonify([{"id": r[0], "title": r[1], "rating": r[2]} for r in recs])

@app.route("/user_recommendations", methods=["POST"])
def save_user_recommendations():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    db = get_db()
    # Update or insert ratings
    for title, rating in data.items():
        existing = db.execute("SELECT id FROM recommendations WHERE user_id = ? AND movie_title = ?", (session["user_id"], title)).fetchone()
        if existing:
            db.execute("UPDATE recommendations SET rating = ?, timestamp = CURRENT_TIMESTAMP WHERE id = ?", (rating, existing[0]))
        else:
            db.execute("INSERT INTO recommendations (user_id, movie_title, rating) VALUES (?, ?, ?)", (session["user_id"], title, rating))
    # Delete movies not in the new list
    if data:
        placeholders = ','.join(['?'] * len(data))
        db.execute(f"DELETE FROM recommendations WHERE user_id = ? AND movie_title NOT IN ({placeholders})", (session["user_id"], *data.keys()))
    else:
        db.execute("DELETE FROM recommendations WHERE user_id = ?", (session["user_id"],))
    db.commit()
    return jsonify({"message": "Recommendations saved or updated"})

@app.route("/user_recommendations/<int:rec_id>", methods=["PATCH"])
def update_user_recommendation(rec_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    new_rating = data.get("rating")
    db = get_db()
    db.execute("UPDATE recommendations SET rating = ? WHERE id = ? AND user_id = ?", (new_rating, rec_id, session["user_id"]))
    db.commit()
    return jsonify({"message": "Recommendation updated"})

@app.route("/user_recommendations/<int:rec_id>", methods=["DELETE"])
def delete_user_recommendation(rec_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    db = get_db()
    db.execute("DELETE FROM recommendations WHERE id = ? AND user_id = ?", (rec_id, session["user_id"]))
    db.commit()
    return jsonify({"message": "Recommendation deleted"})

# ----------------- Recommendation Setup ------------------
movies_df = pd.read_csv("data/movies.csv")
ratings_df = pd.read_csv("data/ratings.csv")

movies_df["genres"] = movies_df["genres"].replace("(no genres listed)", "")

tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
tfidf_matrix = tfidf.fit_transform(movies_df["genres"])

title_to_index = pd.Series(movies_df.index, index=movies_df["title"])

top_users = ratings_df["userId"].value_counts().head(5000).index
top_movies = ratings_df["movieId"].value_counts().head(5000).index
filtered_ratings = ratings_df[ratings_df["userId"].isin(top_users) & ratings_df["movieId"].isin(top_movies)]

user_movie_matrix = filtered_ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)

svd = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd.fit_transform(user_movie_matrix)
movie_factors = svd.components_.T

movieId_to_index = {movie_id: i for i, movie_id in enumerate(user_movie_matrix.columns)}
movie_id_subset = list(user_movie_matrix.columns)
movie_index_subset = movies_df[movies_df['movieId'].isin(movie_id_subset)].index

latest_ratings_payload = {}

@app.route("/search")
def search():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    query = request.args.get("query", "").lower()
    matching = movies_df[movies_df["title"].str.lower().str.contains(query, na=False)].copy()
    return jsonify({"movies": matching["title"].tolist()})

@app.route("/prepare_recommendation", methods=["POST"])
def prepare_recommendation():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    global latest_ratings_payload
    latest_ratings_payload = request.json
    return '', 204

@app.route("/progress_recommend")
def progress_recommend():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_ratings = latest_ratings_payload
    rated_titles = list(user_ratings.keys())
    rated_scores = list(user_ratings.values())

    def generate():
        valid_entries = [(title, float(score)) for title, score in zip(rated_titles, rated_scores) if title in title_to_index]
        if not valid_entries:
            yield f"data: {json.dumps({'progress': 100, 'step': 'No valid titles', 'recommendations': []})}\n\n"
            return

        yield f"data: {json.dumps({'progress': 10, 'step': 'Processing ratings'})}\n\n"
        time.sleep(0.2)

        rated_indices = [title_to_index[title] for title, _ in valid_entries]
        rated_movie_ids = [movies_df.loc[i, "movieId"] for i in rated_indices]
        rated_ratings = [score for _, score in valid_entries]

        rated_svd_indices = [movieId_to_index[movie_id] for movie_id in rated_movie_ids if movie_id in movieId_to_index]
        collaborative_scores = np.zeros(len(movie_id_subset))

        yield f"data: {json.dumps({'progress': 30, 'step': 'Calculating collaborative scores'})}\n\n"
        time.sleep(0.3)

        if rated_svd_indices:
            user_vector = np.zeros(movie_factors.shape[1])
            total_weight = 0.0
            for idx, rating in zip(rated_svd_indices, rated_ratings):
                user_vector += movie_factors[idx] * rating
                total_weight += rating
            if total_weight > 0:
                user_vector /= total_weight
            collaborative_scores = cosine_similarity(user_vector.reshape(1, -1), movie_factors)[0]

        yield f"data: {json.dumps({'progress': 60, 'step': 'Calculating content-based scores'})}\n\n"
        time.sleep(0.3)

        rated_genre_matrix = tfidf_matrix[rated_indices]
        user_profile = rated_genre_matrix.T.dot(np.array(rated_ratings))
        content_scores = cosine_similarity(user_profile.reshape(1, -1), tfidf_matrix)[0]
        content_scores_subset = content_scores[movie_index_subset]

        yield f"data: {json.dumps({'progress': 80, 'step': 'Combining scores'})}\n\n"
        time.sleep(0.2)

        hybrid_scores = 0.6 * collaborative_scores + 0.4 * content_scores_subset
        recommended_titles = (
            movies_df.iloc[movie_index_subset].assign(score=hybrid_scores)
            .sort_values("score", ascending=False)
            .loc[~movies_df.iloc[movie_index_subset]["title"].isin(rated_titles)]
            .head(10)["title"]
            .tolist()
        )

        yield f"data: {json.dumps({'progress': 100, 'step': 'Done', 'recommendations': recommended_titles})}\n\n"

    return Response(generate(), mimetype="text/event-stream")

@app.route("/get_latest_oneoff_ratings")
def get_latest_oneoff_ratings():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    global latest_ratings_payload
    return jsonify(latest_ratings_payload)

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=8080)