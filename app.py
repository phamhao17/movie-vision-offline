import streamlit as st
import json
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests

device = "cpu"  # HF Spaces CPU

# Load movies.json
with open("data/movies.json", "r") as f:
    movies_data = json.load(f)

# Fix embeddings
for m in movies_data:
    emb = m["embedding"]
    if len(emb) < 512:
        emb += [0.0] * (512 - len(emb))
    elif len(emb) > 512:
        emb = emb[:512]
    m["embedding"] = emb

embeddings = torch.tensor([m["embedding"] for m in movies_data])

# Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
