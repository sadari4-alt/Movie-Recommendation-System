# Movie Recommendation System

Python, Machine Learning (Content-Based Filtering)

A simple content-based movie recommender: pick a movie you like, and it
suggests similar movies using **TF-IDF** text vectorization and **cosine
similarity** — no user ratings required, just the movies' own genres and
plot descriptions.

A live web interface is built with **Streamlit**.

## How It Works (Plain-English Explanation)

1. Each movie has a **title**, **genres**, and a short **overview** (plot
   summary).
2. We combine the genres and overview into one block of text per movie —
   this text represents "what the movie is about."
3. We convert that text into numbers using **TF-IDF** (Term Frequency —
   Inverse Document Frequency). This gives more weight to words that are
   distinctive to a movie and less weight to common words that appear
   everywhere (like "a", "the", "story").
4. Every movie becomes a vector of numbers. We compare every pair of
   movies using **cosine similarity** — a measure of the angle between
   two vectors. A score close to **1** means two movies are very similar
   in content; a score close to **0** means they're unrelated.
5. When you pick a movie, the app looks up its similarity scores against
   every other movie and returns the highest-scoring ones.

This is called **content-based filtering** because it recommends based on
the *content* of items themselves, not on what other users liked (that
would be *collaborative filtering*, a different approach).

## Project Structure

```
movie-recommendation-system/
├── data/
│   └── movies.csv              # dataset: movie_id, title, genres, overview
├── model/
│   └── recommender_model.pkl   # trained TF-IDF vectorizer + similarity matrix
├── screenshots/
│   └── (add a screenshot of the running app here)
├── train_model.py              # builds and saves the recommendation model
├── recommender.py              # loads the model, returns recommendations
├── app.py                      # Streamlit web interface
├── requirements.txt
└── README.md
```

## File-by-File Explanation

**`data/movies.csv`**
The dataset. Each row is one movie with a title, genre tags, and a short
plot overview. This is the only "training data" the model needs — no
external ratings dataset required.

**`train_model.py`**
Loads `movies.csv`, combines genre + overview text for each movie, fits a
`TfidfVectorizer` from scikit-learn on that text, and computes a full
cosine-similarity matrix (every movie compared to every other movie).
Saves the vectorizer, similarity matrix, and dataframe together into
`model/recommender_model.pkl` using `pickle`, so this only needs to run
once (or whenever the dataset changes) — not every time the app starts.

**`recommender.py`**
The reusable "brain" of the project, with no UI code. Loads the saved
model file and exposes two functions:
- `get_all_titles()` — returns every movie title (for populating a dropdown)
- `get_recommendations(title, top_n=5)` — returns the top N most similar
  movies to the one given, using the precomputed similarity matrix

Keeping this separate from `app.py` means the same recommendation logic
could be reused in a command-line tool, a different UI, or tests, without
any changes.

**`app.py`**
The Streamlit web app. Lets the user pick a movie from a dropdown, choose
how many recommendations to show, and displays the results with genres
and similarity scores. This is what you'd demo live.

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (only needed once, or after editing movies.csv)
python train_model.py

# 3. Launch the web app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Command-line quick test (no Streamlit needed)

```bash
python recommender.py
```

This prints recommendations for one sample movie directly in the terminal
— useful for quickly checking the model works before launching the full app.

## Sample Output

```
Recommendations for 'Stellar Horizon':

  Orbit Break      (score: 0.283)  [Sci-Fi Action]
  Echo Chamber     (score: 0.275)  [Thriller Sci-Fi]
  Static Bloom     (score: 0.273)  [Sci-Fi Romance]
  The Long Static  (score: 0.266)  [Sci-Fi Drama]
  Static Horizon   (score: 0.251)  [Sci-Fi Thriller]
```

## Short Explanation for Viva / Interview

> "This is a content-based movie recommendation system. Instead of relying
> on what other users rated highly, it looks at each movie's own genres
> and plot description. I convert that text into numeric vectors using
> TF-IDF, which highlights words that are distinctive to a movie rather
> than common filler words. Then I use cosine similarity to measure how
> close two movies' vectors are — the smaller the angle between them, the
> more similar the movies are considered. When a user picks a movie, I
> look up its precomputed similarity scores against every other movie in
> the dataset and return the highest-scoring ones. I built the whole
> pipeline in Python using Pandas for data handling, scikit-learn for
> TF-IDF and cosine similarity, and Streamlit for the web interface so
> it's interactive and easy to demo."

**Likely follow-up questions and short answers:**

- *"Why TF-IDF instead of just counting words?"* — Plain word counts treat
  every word as equally important, so common words would dominate. TF-IDF
  down-weights words that appear in almost every movie and up-weights
  words that are distinctive, giving a more meaningful comparison.

- *"Why cosine similarity instead of Euclidean distance?"* — Cosine
  similarity measures the *direction* of the vectors, not their magnitude
  (length). Since TF-IDF vectors can vary a lot in length depending on
  text length, comparing direction gives a fairer similarity measure
  between a short and a long overview.

- *"What's a limitation of this approach?"* — It can only recommend
  movies similar in genre/plot wording to ones you already like — it
  won't discover something in a totally different genre that other
  similar users happened to enjoy (that's what collaborative filtering
  would add).

- *"How would you extend this?"* — Add a collaborative-filtering
  component using user ratings (a hybrid recommender), include more
  metadata (cast, director, keywords), or move from TF-IDF to sentence
  embeddings for semantic similarity instead of exact word overlap.

## Author
Adari Sindhu — [GitHub](https://github.com/sadari4-alt) | [LinkedIn](https://linkedin.com/in/sindhu-adari-96b244293)
