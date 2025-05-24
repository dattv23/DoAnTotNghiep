import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def evaluate_mf_model(
    train_df, test_df, n_factors=20, n_epochs=50, learning_rate=0.005, reg=0.01
):
    # Step 1: Map user and product IDs to indices
    all_users = pd.concat([train_df["userId"], test_df["userId"]]).unique()
    all_products = pd.concat([train_df["productId"], test_df["productId"]]).unique()

    user_to_index = {user_id: idx for idx, user_id in enumerate(all_users)}
    product_to_index = {product_id: idx for idx, product_id in enumerate(all_products)}

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["user_idx"] = train_df["userId"].map(user_to_index)
    train_df["product_idx"] = train_df["productId"].map(product_to_index)
    test_df["user_idx"] = test_df["userId"].map(user_to_index)
    test_df["product_idx"] = test_df["productId"].map(product_to_index)

    n_users = len(user_to_index)
    n_products = len(product_to_index)

    # Step 2: Initialize latent factor matrices
    P = np.random.normal(0, 0.01, (n_users, n_factors))
    Q = np.random.normal(0, 0.01, (n_products, n_factors))

    def clip_value(value, min_val=-5.0, max_val=5.0):
        return max(min_val, min(max_val, value))

    def rmse(predictions, actual):
        valid = ~np.isnan(predictions)
        return np.sqrt(np.mean((predictions[valid] - actual[valid]) ** 2))

    train_rmse_history = []
    test_rmse_history = []

    # Step 3: Train the model
    for epoch in range(n_epochs):
        for _, row in train_df.iterrows():
            u = int(row["user_idx"])
            i = int(row["product_idx"])
            rating = row["rating"]

            pred = clip_value(np.dot(P[u], Q[i]))
            error = rating - pred

            for f in range(n_factors):
                p_grad = learning_rate * (error * Q[i, f] - reg * P[u, f])
                q_grad = learning_rate * (error * P[u, f] - reg * Q[i, f])
                P[u, f] += clip_value(p_grad, -0.5, 0.5)
                Q[i, f] += clip_value(q_grad, -0.5, 0.5)

        # Evaluate on training set
        train_preds = np.array(
            [
                clip_value(np.dot(P[int(row["user_idx"])], Q[int(row["product_idx"])]))
                for _, row in train_df.iterrows()
            ]
        )
        train_rmse = rmse(train_preds, train_df["rating"].values)
        train_rmse_history.append(train_rmse)

        # Evaluate on test set
        test_preds = np.array(
            [
                clip_value(np.dot(P[int(row["user_idx"])], Q[int(row["product_idx"])]))
                for _, row in test_df.iterrows()
            ]
        )
        test_rmse = rmse(test_preds, test_df["rating"].values)
        test_rmse_history.append(test_rmse)

        print(
            f"Epoch {epoch+1}/{n_epochs} - Train RMSE: {train_rmse:.4f} - Test RMSE: {test_rmse:.4f}"
        )

    # Step 4: Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs + 1), train_rmse_history, label="Train RMSE")
    plt.plot(range(1, n_epochs + 1), test_rmse_history, label="Test RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Matrix Factorization Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    return P, Q, user_to_index, product_to_index, train_rmse_history, test_rmse_history


def recommend_top_products(
    user_id,
    P,
    Q,
    user_to_index,
    product_to_index,
    index_to_product,
    rated_product_ids=None,
    top_n=10,
):
    def clip(v, lo=-5.0, hi=5.0):
        return max(lo, min(hi, v))

    if user_id not in user_to_index:
        return []

    u_idx = user_to_index[user_id]
    user_vector = P[u_idx]

    scores = {}
    for p_idx, product_vector in enumerate(Q):
        product_id = index_to_product[p_idx]
        if rated_product_ids and product_id in rated_product_ids:
            continue  # Skip items the user already rated
        score = clip(np.dot(user_vector, product_vector))
        scores[product_id] = score

    # Sort products by predicted rating (descending)
    top_products = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_products
