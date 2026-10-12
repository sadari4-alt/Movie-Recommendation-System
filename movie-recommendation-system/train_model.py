"""
train_model.py
---------------
Builds the content-based recommendation model.

How it works (in plain terms):
1. Load the movie dataset (title, genres, overview).
2. Combine each movie's genres + overview into one "content" string —
   this text represents what the movie is "about".
3. Convert that text into numbers using TF-IDF (Term Frequency -
   Inverse Document Frequency), which turns each movie into a vector
   where common, unhelpful words matter less and distinctive words
   matter more.
4. Compute the cosine similarity between every pair of movie vectors.
   Cosine similarity measures the angle between two vectors — a value
   close to 1 means two movies talk about very similar things (similar
   genres/themes), and a value close to 0 means they're unrelated.
5. Save the vectorizer, similarity matrix, and dataframe to disk with
   pickle so the app doesn't need to retrain every time it starts.

Run this once (or whenever movies.csv changes):
    python train_model.py
"""

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data/movies.csv"
MODEL_PATH = "model/recommender_model.pkl"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def build_content_column(df):
    """
    Combine genres and overview into a single text field per movie.
    Genres are repeated to give them a bit more weight than the overview
    text, since genre similarity is usually a stronger signal for
    "movies like this one" than plot wording.
    """
    df = df.copy()
    df["genres"] = df["genres"].fillna("")
    df["overview"] = df["overview"].fillna("")
    df["content"] = (df["genres"] + " " + df["genres"] + " " + df["overview"])
    return df


def build_model(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["content"])

    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return vectorizer, similarity_matrix


def main():
    print("Loading data...")
    df = load_data()
    df = build_content_column(df)

    print(f"Building TF-IDF model for {len(df)} movies...")
    vectorizer, similarity_matrix = build_model(df)

    print("Saving model to disk...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "dataframe": df,
                "vectorizer": vectorizer,
                "similarity_matrix": similarity_matrix,
            },
            f,
        )

    print(f"Done. Model saved to {MODEL_PATH}")
    print(f"Similarity matrix shape: {similarity_matrix.shape}")


if __name__ == "__main__":
    main()
