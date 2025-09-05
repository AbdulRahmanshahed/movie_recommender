# recommender.py

import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle

# 1. Load built-in MovieLens 100k dataset
data = Dataset.load_builtin("ml-100k")

# 2. Split train/test
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# 3. Train SVD model
model = SVD()
model.fit(trainset)

# 4. Evaluate model
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

# 5. Save model
with open("movie_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as movie_model.pkl")
