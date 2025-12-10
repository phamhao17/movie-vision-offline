import streamlit as st
import json
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests

# ----------------------
# Load movies.json
# ----------------------
with open("data/movies.json", "r") as f:
    movies_data = json.load(f)

# Fix embeddings to exactly 512 dims
for m in movies_data:
    emb = m["embedding"]
    if len(emb) < 512:
        emb += [0.0] * (512 - len(emb))
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
device = "cpu"  # HF Spaces miễn phí chỉ dùng CPU
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ----------------------
# Music suggestion map
# ----------------------
music_map = {
    "korean": ("K-Pop", "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"),
    "twist": ("Jazz", "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3"),
    "fbi": ("Electronic", "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3"),
    "con man": ("Hip-hop", "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3"),
    "time loop": ("Synthwave", "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3"),
    "aliens": ("Ambient", "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3"),
    "blindfold": ("Chillout", "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-7.mp3"),
    "class divide": ("Classical", "https://www.soundhelix.com/examples/mp3/Sou
