# NETFLIX MOVIE RECOMMENDATION SYSTEM
# Using Singular Value Decomposition (SVD)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

RATINGS_FILE = os.path.join(
    DATA_DIR, "combined_data_1.txt"
)
MOVIES_FILE = os.path.join(
    DATA_DIR, "movie_titles.txt"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Netflix SVD project initialized.")
print("Project folder:", PROJECT_DIR)
print("Output folder:", OUTPUT_DIR)

print("Ratings file exists:", os.path.isfile(RATINGS_FILE))
print("Movie titles file exists:", os.path.isfile(MOVIES_FILE))



# STEP 1: LOAD MOVIE TITLES


# Load the original Netflix Prize movie metadata
movie_records = []

with open(MOVIES_FILE, "r", encoding="latin-1") as file:
    for line in file:
        line = line.rstrip("\r\n")
        parts = line.split(",", 2)

        if len(parts) != 3:
            continue

        try:
            movie_id = int(parts[0])
            year = parts[1].strip()
            title = parts[2].strip()

            movie_records.append({
                "movie_id": movie_id,
                "year": year,
                "title": title
            })
        except ValueError:
            continue

movies = pd.DataFrame(movie_records)

print("\nNetflix movie metadata loaded:")
print(movies.head(10).to_string(index=False))
print("Total movie titles:", len(movies))




# Inspect the Netflix ratings file
print("\n--- First 10 lines of ratings file ---")

with open(RATINGS_FILE, "r", encoding="utf-8-sig") as file:
    for i, line in enumerate(file):
        if i >= 10:
            break
        print(repr(line.rstrip("\n")))




# Check all dataset files
print("\n--- Files in data folder ---")

for filename in os.listdir(DATA_DIR):
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.isfile(filepath):
        print(f"{filename} - {os.path.getsize(filepath):,} bytes")



# Inspect movie IDs in both datasets
print("\n--- Checking dataset compatibility ---")

# Get movie IDs from the metadata
metadata_ids = set(movies["movie_id"].dropna().astype(int))

# Read the first 20 movie headers from the ratings file
ratings_movie_ids = []

with open(RATINGS_FILE, "r", encoding="utf-8-sig") as file:
    for line in file:
        line = line.strip()

        if line.endswith(":") and line[:-1].isdigit():
            ratings_movie_ids.append(int(line[:-1]))

            if len(ratings_movie_ids) >= 20:
                break

print("First 20 movie IDs in ratings file:")
print(ratings_movie_ids)

print("\nMatching IDs in movie metadata:")
print([mid for mid in ratings_movie_ids if mid in metadata_ids])

print("\nTotal movie IDs in metadata:", len(metadata_ids))



import random

# Load a sample from across the entire Netflix ratings file
def load_ratings_sample(file_path, sample_size=100_000, seed=42):
    """
    Reservoir sampling keeps a representative sample of ratings
    while reading the complete file with limited memory.
    """
    rng = random.Random(seed)
    sample = []
    movie_id = None
    total_ratings = 0
    skipped_lines = 0

    print("\nReading ratings file. This may take a few minutes...")

    with open(file_path, "r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            # Netflix movie header, e.g. 123:
            if line.endswith(":") and line[:-1].isdigit():
                movie_id = int(line[:-1])
                continue

            # Rating row: user_id,rating,date
            parts = line.split(",")

            if movie_id is None or len(parts) < 2:
                skipped_lines += 1
                continue

            try:
                user_id = int(parts[0])
                rating = float(parts[1])

                if not 1 <= rating <= 5:
                    skipped_lines += 1
                    continue

                total_ratings += 1
                row = (user_id, movie_id, rating)

                if len(sample) < sample_size:
                    sample.append(row)
                else:
                    j = rng.randrange(total_ratings)
                    if j < sample_size:
                        sample[j] = row

            except (ValueError, IndexError):
                skipped_lines += 1

    ratings = pd.DataFrame(
        sample, columns=["user_id", "movie_id", "rating"]
    )

    print("Total ratings scanned:", f"{total_ratings:,}")
    print("Sampled ratings:", f"{len(ratings):,}")
    print("Skipped malformed lines:", f"{skipped_lines:,}")

    return ratings


# Run the parser
ratings = load_ratings_sample(RATINGS_FILE, sample_size=100_000)

print("\nFirst 10 sampled ratings:")
print(ratings.head(10))

print("\nUnique users:", ratings["user_id"].nunique())
print("Unique movies:", ratings["movie_id"].nunique())

# Save the sample for reuse
sample_path = os.path.join(OUTPUT_DIR, "ratings_sample.csv")
ratings.to_csv(sample_path, index=False)
print("\nSample saved to:", sample_path)




# ==========================================
# STEP 8: EXPLORATORY DATA ANALYSIS
# ==========================================

print("\n========== EXPLORATORY DATA ANALYSIS ==========")

# 1. Basic statistics
print("\nRating statistics:")
print(ratings["rating"].describe())

print("\nRating counts:")
print(ratings["rating"].value_counts().sort_index())

print("\nAverage rating:", round(ratings["rating"].mean(), 2))

# 2. Plot rating distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=ratings, x="rating")
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rating_distribution.png"))
plt.close()

# 3. Most frequently rated movies
popular_movies = ratings["movie_id"].value_counts().head(10)

plt.figure(figsize=(10, 5))
popular_movies.sort_values().plot(kind="barh")
plt.title("Top 10 Most Frequently Rated Movie IDs")
plt.xlabel("Number of Ratings")
plt.ylabel("Movie ID")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "popular_movies.png"))
plt.close()

# 4. Distribution of ratings per user
user_rating_counts = ratings.groupby("user_id").size()

plt.figure(figsize=(8, 5))
plt.hist(user_rating_counts, bins=30)
plt.title("Distribution of Ratings per User")
plt.xlabel("Number of Ratings in Sample")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ratings_per_user.png"))
plt.close()

print("\nEDA completed.")
print("Charts saved in:", OUTPUT_DIR)
print(" - rating_distribution.png")
print(" - popular_movies.png")
print(" - ratings_per_user.png")



# ==========================================
# STEP 9: TRAIN AND EVALUATE SVD MODEL
# ==========================================

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

print("\n========== SVD MODEL TRAINING ==========")

# Prepare data for Surprise
reader = Reader(rating_scale=(1, 5))

surprise_data = Dataset.load_from_df(
    ratings[["user_id", "movie_id", "rating"]],
    reader
)

# Split into training and testing data
trainset, testset = train_test_split(
    surprise_data,
    test_size=0.20,
    random_state=42
)

print("Training ratings:", trainset.n_ratings)
print("Testing ratings:", len(testset))
print("Training users:", trainset.n_users)
print("Training movies:", trainset.n_items)

# Initialize the SVD model
svd_model = SVD(
    n_factors=50,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

# Train
print("\nTraining SVD model. Please wait...")
svd_model.fit(trainset)

print("Training completed.")

# Evaluate on held-out test data
predictions = svd_model.test(testset)

rmse = accuracy.rmse(predictions, verbose=True)
mae = accuracy.mae(predictions, verbose=True)

print("\n========== MODEL PERFORMANCE ==========")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")

# Save metrics
metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.txt")

with open(metrics_path, "w", encoding="utf-8") as file:
    file.write("Netflix Movie Recommendation - SVD\n")
    file.write(f"Ratings sampled: {len(ratings)}\n")
    file.write(f"Training ratings: {trainset.n_ratings}\n")
    file.write(f"Testing ratings: {len(testset)}\n")
    file.write(f"RMSE: {rmse:.4f}\n")
    file.write(f"MAE: {mae:.4f}\n")

print("\nMetrics saved to:", metrics_path)




# ==========================================
# STEP 10: PERSONALIZED RECOMMENDATIONS
# ==========================================

print("\n========== MOVIE RECOMMENDATIONS ==========")

# Select a user with the most ratings in the training set
user_counts = {
    trainset.to_raw_uid(inner_uid): len(items)
    for inner_uid, items in trainset.ur.items()
}

target_user = max(user_counts, key=user_counts.get)

print("Selected user ID:", target_user)
print("Ratings in training set:", user_counts[target_user])

# Movies rated by this user in the sample
rated_movie_ids = set(
    ratings.loc[
        ratings["user_id"] == int(target_user),
        "movie_id"
    ].astype(int)
)

# Recommend only movies represented in the training data
candidate_movie_ids = {
    int(trainset.to_raw_iid(inner_iid))
    for inner_iid in range(trainset.n_items)
}

unrated_movie_ids = candidate_movie_ids - rated_movie_ids

# Predict ratings for unrated movies
recommendations = []

for movie_id in unrated_movie_ids:
    prediction = svd_model.predict(target_user, movie_id)

    recommendations.append({
        "movie_id": movie_id,
        "predicted_rating": prediction.est
    })

recommendations_df = pd.DataFrame(recommendations)

recommendations_df = recommendations_df.sort_values(
    "predicted_rating",
    ascending=False
).head(10)

# Attach metadata provisionally.
# Verify that movie IDs and titles belong to the same dataset.
recommendations_df = recommendations_df.merge(
    movies[["movie_id", "title"]],
    on="movie_id",
    how="left"
)

recommendations_df = recommendations_df[
    ["movie_id", "title", "predicted_rating"]
]

print("\nTop 10 recommendations:")
print(recommendations_df.to_string(index=False))

recommendations_df.to_csv(
    os.path.join(OUTPUT_DIR, "top_10_recommendations.csv"),
    index=False
)

print("\nRecommendations saved to output/top_10_recommendations.csv")



# ==========================================
# STEP 11: IMPROVED RECOMMENDATIONS
# ==========================================

print("\n========== IMPROVED RECOMMENDATIONS ==========")

# Build user statistics from the training data only
train_rows = [
    (
        trainset.to_raw_uid(uid),
        trainset.to_raw_iid(iid),
        rating
    )
    for uid, items in trainset.ur.items()
    for iid, rating in items
]

train_ratings_df = pd.DataFrame(
    train_rows,
    columns=["user_id", "movie_id", "rating"]
)

user_stats = train_ratings_df.groupby("user_id").agg(
    rating_count=("rating", "count"),
    average_rating=("rating", "mean")
)

# Prefer users with at least 10 ratings and average rating >= 3.5
eligible_users = user_stats[
    (user_stats["rating_count"] >= 10) &
    (user_stats["average_rating"] >= 3.5)
]

if eligible_users.empty:
    eligible_users = user_stats[user_stats["rating_count"] >= 10]

if eligible_users.empty:
    raise ValueError("No users have enough training ratings.")

# Pick the most active eligible user
target_user = eligible_users.sort_values(
    ["rating_count", "average_rating"],
    ascending=False
).index[0]

print("Selected user:", target_user)
print("Ratings in training:", int(
    user_stats.loc[target_user, "rating_count"]
))
print("User average rating:", round(
    user_stats.loc[target_user, "average_rating"], 2
))

# Exclude movies already rated by this user in training
rated_ids = set(
    train_ratings_df.loc[
        train_ratings_df["user_id"] == target_user,
        "movie_id"
    ].astype(int)
)

known_movie_ids = {
    int(trainset.to_raw_iid(iid))
    for iid in range(trainset.n_items)
}

candidates = known_movie_ids - rated_ids

new_recommendations = []

for movie_id in candidates:
    estimated_rating = svd_model.predict(
        target_user, movie_id
    ).est

    if estimated_rating >= 3.5:
        new_recommendations.append({
            "movie_id": movie_id,
            "predicted_rating": round(estimated_rating, 3)
        })

improved_df = pd.DataFrame(new_recommendations)

if not improved_df.empty:
    improved_df = improved_df.sort_values(
        "predicted_rating", ascending=False
    ).head(10)

    improved_df = improved_df.merge(
        movies[["movie_id", "title"]],
        on="movie_id",
        how="left"
    )

    improved_df = improved_df[
        ["movie_id", "title", "predicted_rating"]
    ]

    print("\nRecommended movies:")
    print(improved_df.to_string(index=False))

    improved_df.to_csv(
        os.path.join(
            OUTPUT_DIR, "improved_recommendations.csv"
        ),
        index=False
    )

    print("\nSaved improved_recommendations.csv")
else:
    print(
        "No candidates reached 3.5. "
        "Try a lower threshold or improve the model."
    )




print("\n--- Sample movie metadata for validation ---")

print("Available columns:", movies.columns.tolist())

sample_movie_ids = movies["movie_id"].head(10).tolist()

print(
    movies.loc[
        movies["movie_id"].isin(sample_movie_ids),
        ["movie_id", "year", "title"]
    ].to_string(index=False)
)
