import pickle
import json
import numpy as np
import os

# Paths
pickle_path = "data/embeddings.pkl"
json_path = "data/movies.json"

# Load embeddings.pkl
with open(pickle_path, "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]
df = data["df"]

# Convert embeddings from tensor to list
if hasattr(embeddings, "detach"):
    embeddings = embeddings.detach().cpu().numpy()

embeddings_list = embeddings.tolist()

# Prepare JSON data
movies_data = {
    "titles": df["title"].fillna("").tolist(),
    "tags": df["tags"].fillna("").tolist(),
    "embeddings": embeddings_list
}

# Save lightweight JSON
os.makedirs("data", exist_ok=True)
with open(json_path, "w") as f:
    json.dump(movies_data, f)

print(f"Converted {pickle_path} → {json_path}, number of movies: {len(df)}")
