import streamlit as st
import pickle
import torch
import torch.nn.functional as F

# ------------------------------
# Load embeddings and movie data
# ------------------------------
with open("data/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]  # tensor (num_movies x 512)
df = data["df"]                  # dataframe with movie info

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Movie Vision App")
st.write("Find movies related to your description!")

query = st.text_input("Enter a description of the movie:")

# ------------------------------
# Search functionality
# ------------------------------
if query:
    st.write("Searching for movies related to:", query)

    # Load CLIP model and tokenizer
    from transformers import CLIPModel, CLIPTokenizer

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Tokenize query and get embedding
    with torch.no_grad():
        inputs = tokenizer(query, return_tensors="pt")
        query_embedding = model.get_text_features(**inputs)

    # Normalize embeddings
    query_embedding = F.normalize(query_embedding, dim=1)
    movie_embeddings = F.normalize(embeddings, dim=1)

    # Compute cosine similarity
    similarities = (query_embedding @ movie_embeddings.T).squeeze(0)

    # Get top 5 most similar movies
    topk = torch.topk(similarities, k=5)
    top_indices = topk.indices.tolist()
    top_scores = topk.values.tolist()

    st.write("Top 5 recommended movies:")
    for idx, score in zip(top_indices, top_scores):
        title = df.iloc[idx]["title"]
        tags = df.iloc[idx]["tags"]
        st.write(f"- **{title}** (Score: {score:.3f}) | Tags: {tags}")
