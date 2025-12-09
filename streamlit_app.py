import streamlit as st
import pickle
import torch

# Load embeddings
with open("data/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]
df = data["df"]
