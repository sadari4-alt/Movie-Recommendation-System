"""
app.py
------
Streamlit web interface for the Movie Recommendation System.

Run with:
    streamlit run app.py
"""

import streamlit as st
from recommender import get_all_titles, get_recommendations

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write(
    "Pick a movie you like, and this app will suggest similar movies "
    "based on genre and plot content using **TF-IDF + cosine similarity**."
)

try:
    titles = sorted(get_all_titles())
except FileNotFoundError:
    st.error(
        "Model file not found. Please run `python train_model.py` first "
        "to generate `model/recommender_model.pkl`."
    )
    st.stop()

selected_title = st.selectbox("Choose a movie:", titles)
top_n = st.slider("Number of recommendations:", min_value=3, max_value=10, value=5)

if st.button("Get Recommendations"):
    recommendations = get_recommendations(selected_title, top_n=top_n)

    st.subheader(f"Because you liked *{selected_title}*:")

    for rec in recommendations:
        with st.container():
            st.markdown(f"### {rec['title']}")
            st.caption(f"Genres: {rec['genres']}  •  Similarity: {rec['score']}")
            st.write(rec["overview"])
            st.divider()

st.sidebar.header("About")
st.sidebar.write(
    "This is a **content-based** recommender: it doesn't use other users' "
    "ratings, only the text content (genres + overview) of each movie. "
    "Two movies are considered similar if their TF-IDF vectors point in "
    "a similar direction (high cosine similarity)."
)
