# Collaborative Filtering Experiment

import pandas as pd
import random
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Generate pretend user IDs
num_users = 100
user_ids = [f"user_{i}" for i in range(1, num_users + 1)]

# Generate pretend book IDs
num_books = 1000
book_ids = [f"book_{i}" for i in range(1, num_books + 1)]

# Generate pretend ratings
user_item_ratings = []
for user_id in user_ids:
    for book_id in book_ids:
        rating = random.randint(1, 5)  # Pretend ratings between 1 and 5
        user_item_ratings.append([user_id, book_id, rating])

# Create a DataFrame
df = pd.DataFrame(user_item_ratings, columns=["userID", "itemID", "rating"])

# Save the pretend data to a CSV file
df.to_csv("synthetic_ratings.csv", index=False)

# Load the pretend data into Surprise dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df, reader)

# Split the dataset into a training set and a test set
trainset, testset = train_test_split(data, test_size=0.2)

# Create a collaborative filtering model 
model = SVD()

# Train the model on the training set
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model
rmse = accuracy.rmse(predictions)

print("Collaborative Filtering Results:")
print(f"RMSE: {rmse}")

