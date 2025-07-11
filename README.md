# 🎬 Movie Recommendation System

A **Flask-based web app** where you log in, rate movies you’ve seen, and get instant, personalized movie recommendations tailored to your taste. Built with a modern glassmorphism UI and MovieLens data.

---

## 🚀 Features

- 🔐 User authentication (signup, login, logout)
- 📚 Add, update, and delete movies in your personal library
- ➕ Adding a movie saves it instantly; deleting/updating requires pressing "Save Library" or "Discard Changes"
- 🔍 Autocomplete search for movie titles (from MovieLens)
- 📊 Hybrid recommendation engine (Collaborative + Content-Based)
- ⭐ User rating input as a float (1.0–5.0, one decimal)
- 🚫 Input movies are excluded from recommendations
- ⏳ Animated loading bar while recommendations are generated
- 🎨 Clean, modern glassmorphism UI with Bootstrap
- 📝 Library and one-off recommendation modes

---

## 🧠 How It Works

### 🧾 User Input
Log in, add movies to your library and rate them (1.0–5.0, one decimal). Additions save instantly; deletions/updates require saving.

### 🤝 Collaborative Filtering (60%)
Uses real user ratings from the MovieLens dataset and SVD/cosine similarity to find similar preferences.

### 🎬 Content-Based Filtering (40%)
Uses genres from MovieLens and applies TF-IDF + cosine similarity.

### ⚡ Hybrid Model
Final scores combine both collaborative and content-based methods and are ranked to generate the top 10 recommendations.

---

## 💻 Installation

### 1. Clone the repository
```sh
git clone <your-repo-url>
cd Movie Recommendation System
```

### 2. Install dependencies
```sh
pip install -r requirements.txt
```

### 3. Download MovieLens data
Download `ratings.csv` and `movies.csv` from the [MovieLens 20M dataset](https://grouplens.org/datasets/movielens/20m/) and place them in the `data/` folder.

### 4. Set up environment variables
Create a `.env` file in the root of the project with:
```
FLASK_SECRET_KEY=your_secret_key_here
```

### 5. Run the Flask app
```sh
python app.py
```

Open your browser at [http://127.0.0.1:8080](http://127.0.0.1:8080)

---

## 🖼️ UI Features

- Glassmorphism card layout for all pages
- Responsive design with Bootstrap
- Autocomplete and dynamic add/remove for movie inputs
- Real-time loading bar while generating results
- Personalized, ranked recommendations (excluding what you already rated)
- Library and one-off recommendation flows

---

## 🔮 Future Ideas

- [ ] Filter by genre or year
- [ ] Show movie posters
- [ ] Pagination and mobile optimization

---

## 📄 License

MIT License — Free to use, modify, and share!

---

## 🙌 Acknowledgments

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- Bootstrap, Flask, scikit-learn, pandas
