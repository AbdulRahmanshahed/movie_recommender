# 🎬 Movie Recommender System (Collaborative Filtering with Surprise)

This project is part of my AI/ML internship (Week 2). It builds a movie recommender system using the **MovieLens 100k dataset** and **Collaborative Filtering (SVD matrix factorization)** with the `scikit-surprise` library. The workflow: load dataset → split into train/test → train SVD model → evaluate with RMSE & MAE → save model (`movie_model.pkl`) → create script to recommend top 5 movies for any user ID. Tech stack: Python, scikit-surprise, pandas, numpy, matplotlib, pickle. Files: `recommender.py` (training script), `recommend.py` (interactive recommender), `movie_model.pkl` (saved trained model).  

### ⚡ How to Run
```bash
pip install scikit-surprise pandas numpy matplotlib
python recommender.py     # trains and saves model
python recommend.py       # enter a user ID to get recommendations
```
### 📊 Example Usage
Enter User ID (1–943): 50
🎥 Top 5 Recommendations:
Star Wars (Predicted Rating: 4.89)
The Godfather (Predicted Rating: 4.76)
Raiders of the Lost Ark (Predicted Rating: 4.71)
Schindler's List (Predicted Rating: 4.68)
Braveheart (Predicted Rating: 4.62)
### 📈 Results
Model: SVD (Singular Value Decomposition)

Evaluation: RMSE ≈ 0.93, MAE ≈ 0.74 (on MovieLens 100k test split)

Provides top-5 personalized movie recommendations for each user
Enter User ID (1–943): exit
👋 Exiting recommender.
