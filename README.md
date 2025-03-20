# 🎬📺 Movie & TV Show Recommendation System

A **Flask-based recommendation system** that suggests **movies & TV shows** based on user preferences using **The Movie Database (TMDb) API**.

## 🚀 Features

- ✅ **Hybrid Recommendation Model** (Content-Based Filtering)  
- ✅ **Live Movie & TV Show Data from TMDb API**  
- ✅ **Search Autocomplete for Movies & TV Shows**  
- ✅ **User Rating System (1-100 Scale)**  
- ✅ **No Local Database Needed – Fully Dynamic**  

---

## 📌 How It Works

### 1️⃣ Fetching Movie & TV Show Data  
- The system **calls TMDb API** to get **the latest movies & TV shows**.  
- It extracts **genres, cast, directors**, and other metadata.  

### 2️⃣ User Input  
- Users **type in movies or TV shows** they've watched and enter a **rating (1-100 scale).**  
- The system **normalizes ratings to match TMDb's 0.5-5 scale**.  

### 3️⃣ Content-Based Filtering (Hybrid Model)  
- Finds **similar movies & TV shows** using **genres, cast, and director metadata**.  
- Uses **TF-IDF & Cosine Similarity** to rank recommendations.  

### 4️⃣ Displaying Recommendations  
- The top **10 recommendations** are shown dynamically **without refreshing the page**.  

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```
git clone https://github.com/yourusername/movie-recommendation.git
cd movie-recommendation
```

### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 3️⃣ Get a TMDb API Key
1. **Sign up for TMDb API** → [https://www.themoviedb.org/signup](https://www.themoviedb.org/signup)  
2. **Request a free API key**  
3. **Create a `.env` file** in the project root and paste:  

```
TMDB_API_KEY=your_api_key_here
```

### 4️⃣ Run the Flask App
```
python app.py
```
Then, open **[`http://127.0.0.1:8080/`](http://127.0.0.1:8080/)** in your browser.  

---

## 🎮 How to Use

1. **Search for movies/TV shows you’ve watched**  
2. **Enter a rating (1-100 scale)**  
3. **Click "Get Recommendations"**  
4. **View personalized movie & TV show suggestions**  

---

## 📌 Example Screenshot
*(Add a screenshot of the UI here!)*  

---

## 🛠 Future Improvements
- ✅ **Filter by Movies vs. TV Shows**  
- ✅ **Sort by Genre or Maturity Rating**  
- ✅ **Collaborative Filtering Integration**  

---

## 🤝 Contributing
Want to improve this project? Fork it & submit a PR! 🚀  

📩 **Contact:** `your.email@example.com`  

---

## 📜 License
MIT License. Feel free to use & modify this project! 🎬🔥  
