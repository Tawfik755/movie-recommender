import streamlit as st
from joblib import load
import pandas as pd

from src.data_loader import load_movielens
from src.collaborative import recommend_knn
from src.content_based import recommend_content
from src.hybrid import hybrid_recommend

st.title("🎬 Movie Recommendation System")

ratings, movies = load_movielens()

knn_model = load("models/knn_item.pkl")
content_model = load("models/content_model.pkl")

option = st.selectbox("Choose type of recommendation:", 
    ["Collaborative Filtering", "Content-Based", "Hybrid"])

if option == "Collaborative Filtering":
    user = st.number_input("Enter User ID", min_value=1)
    if st.button("Recommend"):
        results = recommend_knn(user, ratings, movies, knn_model)
        st.write(results)

elif option == "Content-Based":
    title = st.selectbox("Choose a movie:", movies["title"].values)
    if st.button("Recommend"):
        results = recommend_content(title, movies, content_model)
        st.write(results)

else:
    user = st.number_input("Enter User ID", min_value=1)
    if st.button("Recommend"):
        results = hybrid_recommend(user, ratings, movies, 
        knn_model, content_model)
        st.write(results)

