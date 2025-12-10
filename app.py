import streamlit as st
import json
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests

# Set the device for model computation (Using CPU as requested for general/free deployment)
device = "cpu"

# --- Data Loading ---

# Load movies.json (Ensure this file is in the 'data' directory on GitHub)
try:
    with open("data/movies.json", "r") as f:
        movies_data = json.load(f)
except FileNotFoundError:
    st.error("Error: 'data/movies.json' not found. Please ensure the file is committed and pushed to GitHub.")
    st.stop()

# Fix and normalize embeddings (ensure all embeddings are 512 dimensions)
for m in movies_data:
    emb = m["embedding"]
    if len(emb) < 512:
        # Pad with zeros if shorter
        emb += [0.0] * (512 - len(emb))
    elif len(emb) > 512:
        # Truncate if longer
        emb = emb[:512]
    m["embedding"] = emb

# Convert all embeddings into a PyTorch Tensor
embeddings = torch.tensor([m["embedding"] for m in movies_data])

# --- Model Loading ---

@st.cache_resource
def load_clip_model():
    """Loads the CLIP model and processor, and caches them."""
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor
    except Exception as e:
        st.error(f"Error loading CLIP model. Check if 'transformers' and 'accelerate' are installed. Details: {e}")
        st.stop()

model, processor = load_clip_model()

# --- Streamlit Application Logic ---

st.title("🎬 Movie Recommendation App based on Text")
st.write("Enter a description of the movie you are looking for.")

# 1. Get query from the user
text_query = st.text_input("Your Movie Description:", "A thrilling action movie about fighting robots in the future")

if text_query:
    # 2. Process the query
    with st
