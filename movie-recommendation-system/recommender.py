"""
recommender.py
---------------
Loads the trained model (from train_model.py) and provides a simple
function to get movie recommendations for a given title.

This module has no UI code in it on purpose — it can be reused by
app.py (Streamlit), a command-line script, or tests, without changes.
"""

import pickle

MODEL_PATH = "model/recommender_model.pkl"

_cache = {}


def load_model(path=MODEL_PATH):
    """Loads the pickled model once and caches it in memory."""
    if "model" not in _cache:
        with open(path, "rb") as f:
            _cache["model"] = pickle.load(f)
    return _cache["model"]


def get_all_titles():
    """Returns the list of all movie titles available in the dataset."""
    model = load_model()
    return model["dataframe"]["title"].tolist()


def get_recommendations(title, top_n=5):
    """
    Returns the top_n movies most similar to the given title, based on
    cosine similarity of their TF-IDF content vectors.

    Returns a list of dicts: [{"title": ..., "genres": ..., "score": ...}, ...]
    Raises ValueError if the title isn't found in the dataset.
    """
    model = load_model()
    df = model["dataframe"]
    similarity_matrix = model["similarity_matrix"]

    matches = df.index[df["title"].str.lower() == title.lower()]
    if len(matches) == 0:
        raise ValueError(f"'{title}' not found in the dataset.")

    idx = matches[0]

    # Get similarity scores for this movie against every other movie
    scores = list(enumerate(similarity_matrix[idx]))

    # Sort by similarity score, descending, excluding the movie itself
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]

    recommendations = []
    for movie_idx, score in scores:
        row = df.iloc[movie_idx]
        recommendations.append(
            {
                "title": row["title"],
                "genres": row["genres"],
                "overview": row["overview"],
                "score": round(float(score), 3),
            }
        )

    return recommendations


if __name__ == "__main__":
    # Quick manual check when running this file directly
    sample_title = get_all_titles()[0]
    print(f"Recommendations for '{sample_title}':\n")
    for rec in get_recommendations(sample_title, top_n=5):
        print(f"  {rec['title']}  (score: {rec['score']})  [{rec['genres']}]")
