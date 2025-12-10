import json

movies_data = [
    {
        "title": "The Handmaiden",
        "tags": "korean|twist",
        "summary": "A Korean woman is hired as a handmaiden to a Japanese heiress, but is secretly involved in a con.",
        "poster_url": "https://via.placeholder.com/200x300.png?text=Handmaiden",
        "embedding": [0.1 + 0.01*i for i in range(512)]
    },
    {
        "title": "Catch Me If You Can",
        "tags": "fbi|con man",
        "summary": "A brilliant young con artist escapes the FBI while forging checks across the country.",
        "poster_url": "https://via.placeholder.com/200x300.png?text=Catch+Me+If+You+Can",
        "embedding": [0.2 + 0.01*i for i in range(512)]
    },
    {
        "title": "Edge of Tomorrow",
        "tags": "time loop|aliens",
        "summary": "A soldier relives the same day fighting aliens, gaining skill to save the world.",
        "poster_url": "https://via.placeholder.com/200x300.png?text=Edge+of+Tomorrow",
        "embedding": [0.3 + 0.01*i for i in range(512)]
    },
    {
        "title": "Bird Box",
        "tags": "blindfold",
        "summary": "A mysterious force drives society to suicide; a mother and children navigate a dangerous world blindfolded.",
        "poster_url": "https://via.placeholder.com/200x300.png?text=Bird+Box",
        "embedding": [0.4 + 0.01*i for i in range(512)]
    },
    {
        "title": "Parasite",
        "tags": "korean|class divide",
        "summary": "A poor family schemes to infiltrate a wealthy household, revealing social inequality.",
        "poster_url": "https://via.placeholder.com/200x300.png?text=Parasite",
        "embedding": [0.5 + 0.01*i for i in range(512)]
    },
    {
        "title": "Inception",
        "tags": "sci-fi|dream",
        "summary": "A thief enters dreams to steal secrets but gets caught in a complex heist.",
        "poster_url": "https://via.placeholder.com/200x300.png?text=Inception",
        "embedding": [0.6 + 0.01*i for i in range(512)]
    },
    {
        "title": "Interstellar",
        "tags": "space|sci-fi",
        "summary": "Astronauts travel through a wormhole to find a new home for humanity.",
        "poster_url": "https://via.placeholder.com/200x300.png?text=Interstellar",
        "embedding": [0.7 + 0.01*i for i in range(512)]
    },
    {
        "title": "La La Land",
        "tags": "musical|romance",
        "summary": "A jazz musician and an aspiring actress fall in love while pursuing their dreams.",
        "poster_url": "https://via.placeholder.com/200x300.png?text=La+La+Land",
        "embedding": [0.8 + 0.01*i for i in range(512)]
    },
    {
        "title": "The Matrix",
        "tags": "sci-fi|action",
        "summary": "A hacker discovers reality is a simulation and fights against the machines.",
        "poster_url": "https://via.placeholder.com/200x300.png?text=Matrix",
        "embedding": [0.9 + 0.01*i for i in range(512)]
    },
    {
        "title": "Frozen",
        "tags": "animation|family",
        "summary": "Two sisters navigate magical powers and family bonds in a frozen kingdom.",
        "poster_url": "https://via.placeholder.com/200x300.png?text=Frozen",
        "embedding": [1.0 + 0.01*i for i in range(512)]
    }
]

# Save to JSON file
with open("data/movies.json", "w") as f:
    json.dump(movies_data, f, indent=4)

print("Generated movies.json with 10 demo movies and 512-dim embeddings!")
