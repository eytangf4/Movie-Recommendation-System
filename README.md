# 🎬📺 Movie & TV Show Recommendation System

A **Flask-based web app** that recommends **movies and TV shows** based on your personal preferences using a **hybrid of collaborative filtering and content-based filtering**.

It combines user ratings from the **MovieLens** dataset with real-time metadata from the **TMDb API** to generate intelligent, personalized recommendations.

---

## 🚀 Features

- 🔍 Autocomplete search for movie/TV titles (from MovieLens + TMDb)
- 📊 Hybrid recommendation engine (Collaborative + Content-Based)
- 🎯 User rating input on a scale of 1–100
- 🚫 Input movies are excluded from recommendations
- ⏳ Loading bar with percentage & animation while recommendations are generated
- 💡 Uses real-world ratings + metadata (genres, cast, director)
- 🎨 Clean, responsive UI with Bootstrap

---

## 🧠 How It Works

### 🧾 User Input
You enter titles and assign ratings between 1 and 100.

### 🤝 Collaborative Filtering (60%)
We use real user ratings from the MovieLens dataset to find similar preferences using Approximate Nearest Neighbors on a sparse matrix.

### 🎬 Content-Based Filtering (40%)
We pull metadata from the TMDb API and apply TF-IDF + cosine similarity to compare genres, cast, and directors.

### ⚡ Hybrid Model
Final scores combine both CF and CBF methods and are ranked to generate top 10 recommendations.

---

## 💻 Installation

### 1. Clone the repository
```
git clone https://github.com/yourusername/movie-tv-recommender.git
cd movie-tv-recommender
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Download MovieLens data
Download `ratings.csv` and `movies.csv` from the [MovieLens 20M dataset](https://grouplens.org/datasets/movielens/20m/) and place them in the root folder.

### 4. Get a TMDb API Key
Sign up at [https://www.themoviedb.org/](https://www.themoviedb.org/) and request a free developer API key.

Create a `.env` file in the root of the project with:
```
TMDB_API_KEY=your_api_key_here
```

### 5. Run the Flask app
```
python app.py
```

Open your browser at [http://127.0.0.1:8080](http://127.0.0.1:8080)

---

## 🖼️ UI Features

- 🔍 Search bar with release year support
- ➕ Add/remove inputs dynamically
- 💯 Ratings out of 100
- 🔄 Real-time loading bar while generating results
- 🎬 Personalized, ranked recommendations (excluding what you already rated)

---

## 🔮 Future Ideas

- [ ] Filter by genre, media type, or maturity rating
- [ ] Show movie/TV posters via TMDb
- [ ] Save user profiles and rating history
- [ ] Pagination and mobile optimization

---

## 📄 License

MIT License — Free to use, modify, and share!

---

## 🙌 Acknowledgments

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [TMDb API](https://www.themoviedb.org/)
- Bootstrap, Flask, scikit-learn, pandas
