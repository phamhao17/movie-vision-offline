import pandas as pd
import pickle
import torch
from transformers import CLIPModel, CLIPTokenizer

def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    return model, tokenizer

def generate_embeddings(movie_csv_path="data/movies.csv",
                        output_path="data/embeddings.pkl"):

    # Read CSV and create list of text descriptions
    df = pd.read_csv(movie_csv_path)
    texts = (df["title"].fillna("").astype(str) + " " + df["tags"].fillna("").astype(str)).tolist()

    # Load model and tokenizer
    model, tokenizer = load_clip_model()

    # Generate embeddings
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, return_tensors="pt")
        embeddings = model.get_text_features(**inputs)

    # Save embeddings to file
    with open(output_path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "df": df}, f)

    print("Embeddings saved! Number of embeddings:", embeddings.shape)

# Call the function when running the script directly
if __name__ == "__main__":
    generate_embeddings(movie_csv_path="data/movies.csv",
                        output_path="data/embeddings.pkl")
