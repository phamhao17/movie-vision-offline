# app.py
import streamlit as st
import pathlib # <- Line 1: Add the pathlib library import

# (Other Streamlit configuration or setup commands...)

# --- FILE ERROR FIX SECTION ---
# Line 2: Get the absolute directory path where the current script (app.py) is located
script_dir = pathlib.Path(__file__).parent 

# Line 3: Construct the full, absolute path to the JSON file
json_path = script_dir / "data" / "movies.json"
# ------------------------------

# Line 4 (The previous Line 18 in your Traceback): Use the new path variable
try:
    with open(json_path, "r") as f: # <- Use the 'json_path' variable instead of "data/movies.json"
        # ... your code to load JSON ...
        movies = json.load(f) # Assuming you are using the 'json' library
    
    st.success(f"Successfully loaded data from: {json_path}")
    
except Exception as e:
    st.error(f"An error occurred while loading the JSON file: {e}")
    # Handle the error or display a friendly message

# (The rest of your Streamlit code)
# ...import streamlit as st
import json
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests
from io import BytesIO

# ----------------------
# Cấu hình
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------
# Load movies.json
# ----------------------
with open("data/movies.json", "r") as f:
    movies_data = json.load(f)

# Fix embeddings to exactly 512 dims
for m in movies_data:
    emb = m["embedding"]
    if len(emb) < 512:
        emb = emb + [0.0]*(512 - len(emb))
    elif len(emb) > 512:
        emb = emb[:512]
    m["embedding"] = emb

titles = [m["title"] for m in movies_data]
tags = [m["tags"] for m in movies_data]
summaries = [m["summary"] for m in movies_data]
poster_urls = [m["poster_url"] for m in movies_data]
embeddings = torch.tensor([m["embedding"] for m in movies_data])

# ----------------------
# CLIP Model
# ----------------------
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ----------------------
# Music suggestion
# ----------------------
music_map = {
    "korean": ("K-Pop", "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"),
    "twist": ("Jazz", "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3"),
    "fbi": ("Electronic", "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3"),
    "con man": ("Hip-hop", "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3"),
    "time loop": ("Synthwave", "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3"),
    "aliens": ("Ambient", "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3"),
    "blindfold": ("Chillout", "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-7.mp3"),
    "class divide": ("Classical", "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3")
    # Thêm các thể loại khác nếu muốn
}

# ----------------------
# Streamlit UI
# ----------------------
st.title("🎬 Movie Vision - Image & Music Recommendation")
mode = st.radio("Choose mode:", ["Upload Image to find movies", "Text Description to find images"])

if mode == "Upload Image to find movies":
    uploaded_file = st.file_uploader("Upload a movie image", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # CLIP image embedding
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_emb = model.get_image_features(**inputs)

        # Cosine similarity
        image_emb_norm = F.normalize(image_emb, dim=-1)
        movie_embs_norm = F.normalize(embeddings, dim=-1).to(device)
        similarities = (image_emb_norm @ movie_embs_norm.T).squeeze(0)
        top5_idx = similarities.topk(min(5, len(titles))).indices.cpu().numpy()

        st.write("### Top 5 similar movies:")
        for idx in top5_idx:
            st.write(f"### {titles[idx]}")
            st.write(f"**Tags:** {tags[idx]}")
            st.write(f"**Summary:** {summaries[idx]}")
            if poster_urls[idx]:
                st.image(poster_urls[idx], width=200)
            # Music suggestion
            movie_tags = tags[idx].lower().split("|")
            suggested_genres = set()
            for t in movie_tags:
                if t.strip() in music_map:
                    suggested_genres.add(t.strip())
            if suggested_genres:
                st.write("🎵 Suggested music:")
                for g in suggested_genres:
                    genre_name, music_url = music_map[g]
                    st.write(f"- {genre_name}")
                    try:
                        audio_bytes = requests.get(music_url).content
                        st.audio(audio_bytes, format="audio/mp3")
                    except:
                        st.write("Audio not available.")
            st.write("---")

elif mode == "Text Description to find images":
    desc = st.text_area("Enter your image description:")
    if st.button("Generate placeholder images"):
        st.write("Note: This demo does not generate real images, just shows placeholders.")
        # Fake placeholder images (since no real API)
        for i in range(3):
            placeholder_url = f"https://via.placeholder.com/200x200.png?text=Image+{i+1}"
            st.image(placeholder_url, caption=f"Generated Image {i+1}")
