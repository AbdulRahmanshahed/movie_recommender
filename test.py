# recommend.py

import pickle
from surprise import Dataset

# Load trained model
with open("movie_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load movie data
data = Dataset.load_builtin("ml-100k")
trainset = data.build_full_trainset()

# Get movie ID mappings
movie_id_map = {v: k for k, v in trainset._raw2inner_id_items.items()}
movie_titles = {}
with open(r"C:\Users\DELL\.surprise_data\ml-100k\ml-100k\u.item", encoding="ISO-8859-1") as f:
    for line in f:
        parts = line.split("|")
        movie_titles[parts[0]] = parts[1]

print("🎬 Movie Recommender Ready! Type 'exit' to quit.\n")

while True:
    user_input = input("Enter User ID (1–943): ")

    if user_input.lower() == "exit":
        print("👋 Exiting recommender.")
        break

    try:
        user_id = int(user_input)
        # Get all movie IDs
        all_movie_ids = list(trainset._raw2inner_id_items.keys())

        # Predict ratings for unseen movies
        predictions = []
        for movie_id in all_movie_ids:
            pred = model.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Top 5 recommendations
        print("\n🎥 Top 5 Recommendations:")
        for movie_id, rating in predictions[:5]:
            print(f"{movie_titles[movie_id]} (Predicted Rating: {rating:.2f})")
        print("")

    except Exception as e:
        print(f"⚠️ Error: {e}\n")
