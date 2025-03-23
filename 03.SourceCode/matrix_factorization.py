import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read data from CSV file
data = pd.read_csv(
    "C:\\Users\\LT64\\Desktop\\DoAnTotNghiep\\02.Dataset\\rating.csv", encoding="utf-8"
)

# Get necessary columns
ratings = data[
    ["ProductID", "CustomerID", "Rating"]
].copy()  # Create a copy to avoid warnings

# Convert ProductID and CustomerID to integer indices
user_ids = ratings["CustomerID"].unique()
product_ids = ratings["ProductID"].unique()

user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
product_to_index = {product_id: idx for idx, product_id in enumerate(product_ids)}

ratings["user_idx"] = ratings["CustomerID"].map(user_to_index)
ratings["product_idx"] = ratings["ProductID"].map(product_to_index)

# Split data into training and test sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Initialize parameters
n_users = len(user_ids)
n_products = len(product_ids)
n_factors = 20  # Number of latent factors
learning_rate = 0.005  # Reduced learning rate to prevent overflow
n_epochs = 50  # Number of iterations
reg = 0.01  # Regularization parameter

# Initialize latent factor matrices for users and products
# Use smaller initial values to prevent overflow
P = np.random.normal(0, 0.01, (n_users, n_factors))  # User matrix
Q = np.random.normal(0, 0.01, (n_products, n_factors))  # Product matrix


# Function to calculate RMSE
def rmse(predictions, actual):
    # Filter out any NaN values
    valid_indices = ~np.isnan(predictions)
    if not np.any(valid_indices):
        return float("nan")
    return np.sqrt(np.mean((predictions[valid_indices] - actual[valid_indices]) ** 2))


# Function to clip values to prevent overflow
def clip_value(value, min_val=-5.0, max_val=5.0):
    return max(min_val, min(max_val, value))


# Train the model with SGD
train_rmse_history = []
test_rmse_history = []

for epoch in range(n_epochs):
    # Train on training set
    for _, row in train_data.iterrows():
        u = int(row["user_idx"])
        i = int(row["product_idx"])
        r_ui = row["Rating"]

        # Predict rating
        prediction = np.dot(P[u, :], Q[i, :])
        # Clip prediction to prevent extreme values
        prediction = clip_value(prediction)
        error = r_ui - prediction

        # Update P and Q using SGD with gradient clipping
        for f in range(n_factors):
            p_update = learning_rate * (error * Q[i, f] - reg * P[u, f])
            q_update = learning_rate * (error * P[u, f] - reg * Q[i, f])

            # Clip updates to prevent overflow
            p_update = clip_value(p_update, -0.5, 0.5)
            q_update = clip_value(q_update, -0.5, 0.5)

            P[u, f] += p_update
            Q[i, f] += q_update

    # Calculate RMSE on training set
    train_predictions = np.array(
        [
            clip_value(
                np.dot(P[int(row["user_idx"]), :], Q[int(row["product_idx"]), :])
            )
            for _, row in train_data.iterrows()
        ]
    )
    train_rmse = rmse(train_predictions, train_data["Rating"].values)
    train_rmse_history.append(train_rmse)

    # Calculate RMSE on test set
    test_predictions = np.array(
        [
            clip_value(
                np.dot(P[int(row["user_idx"]), :], Q[int(row["product_idx"]), :])
            )
            for _, row in test_data.iterrows()
        ]
    )
    test_rmse = rmse(test_predictions, test_data["Rating"].values)
    test_rmse_history.append(test_rmse)

    print(
        f"Epoch {epoch + 1}/{n_epochs} - Train RMSE: {train_rmse:.4f} - Test RMSE: {test_rmse:.4f}"
    )


# Function to predict rating for a user-product pair
def predict_rating(user_id, product_id):
    if user_id not in user_to_index or product_id not in product_to_index:
        return None  # If user or product is not in the data
    u = user_to_index[user_id]
    i = product_to_index[product_id]
    # Clip prediction to avoid extreme values
    return clip_value(np.dot(P[u, :], Q[i, :]))


# Example prediction
user_id_example = 18387707  # An example CustomerID
product_id_example = 192733741  # An example ProductID
predicted_rating = predict_rating(user_id_example, product_id_example)
if predicted_rating is not None:
    print(
        f"Predicted rating for User {user_id_example} and Product {product_id_example}: {predicted_rating:.2f}"
    )
else:
    print(
        f"Unable to predict for User {user_id_example} and Product {product_id_example} (not in training data)"
    )

# Plot RMSE graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_epochs + 1), train_rmse_history, label="Train RMSE")
plt.plot(range(1, n_epochs + 1), test_rmse_history, label="Test RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Matrix Factorization Learning Curve")
plt.legend()
plt.grid(True)
plt.show()
