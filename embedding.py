import pandas as pd
import pickle
import torch
from transformers import CLIPModel, CLIPTokenizer
import os


def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    return model, tokenizer


def generate_embeddings(movie_csv_path="data/movies.csv",
                        output_path="data/embeddings.pkl"):

    df = pd.read_csv(movie_csv_path)
    texts = (df["title"] + " " + df["tags"]).tolist()

    model, tokenizer = load_clip()

    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, return_tensors="pt")
        embeddings = model.get_text_features(**inputs)

    with open(output_path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "df": df}, f)

    print("Saved embeddings!")
