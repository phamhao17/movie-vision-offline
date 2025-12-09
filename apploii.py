import streamlit as st
import pickle
import numpy as np

# Load embeddings
with open("data/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]  # đây là torch tensor, nhưng chúng ta chỉ đọc, không cần torch
df = data["df"]

st.title("Movie Vision App")

# Nhập mô tả tìm phim
query = st.text_input("Enter movie description:")

if query:
    st.write("Searching for movies matching:", query)
    
    # Ví dụ dummy search: lấy 5 phim đầu (có thể thay bằng tính cosine similarity offline)
    top_movies = df.head(5)
    for i, row in top_movies.iterrows():
        st.write(f"{row['title']} - {row['tags']}")
