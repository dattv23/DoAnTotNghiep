import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu
train_data = pd.read_csv(
    "C:\\Users\\LT64\\Desktop\\DoAnTotNghiep\\02.Dataset\\train_data.csv",
    encoding="utf-8",
)
test_data = pd.read_csv(
    "C:\\Users\\LT64\\Desktop\\DoAnTotNghiep\\02.Dataset\\test_data.csv",
    encoding="utf-8",
)

# Lấy cột cần thiết
train_ratings = train_data[["ProductID", "CustomerID", "Rating"]].copy()
test_ratings = test_data[["ProductID", "CustomerID", "Rating"]].copy()

# Kết hợp dữ liệu để tạo ánh xạ
combined_ratings = pd.concat([train_ratings, test_ratings])
user_ids = combined_ratings["CustomerID"].unique()
product_ids = combined_ratings["ProductID"].unique()

user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
product_to_index = {product_id: idx for idx, product_id in enumerate(product_ids)}

train_ratings["user_idx"] = train_ratings["CustomerID"].map(user_to_index)
train_ratings["product_idx"] = train_ratings["ProductID"].map(product_to_index)
test_ratings["user_idx"] = test_ratings["CustomerID"].map(user_to_index)
test_ratings["product_idx"] = test_ratings["ProductID"].map(product_to_index)

# Khởi tạo tham số
n_users = len(user_ids)
n_products = len(product_ids)
n_factors = 5
learning_rate = 0.001  # Giảm learning rate
n_epochs = 50
reg = 0.5  # Tăng regularization
dropout_rate = 0.2  # Thêm dropout 20%

# Khởi tạo ma trận yếu tố ẩn
P = np.random.normal(0, 0.01, (n_users, n_factors))
Q = np.random.normal(0, 0.01, (n_products, n_factors))


# Hàm tính RMSE
def rmse(predictions, actual):
    valid_indices = ~np.isnan(predictions)
    if not np.any(valid_indices):
        return float("nan")
    return np.sqrt(np.mean((predictions[valid_indices] - actual[valid_indices]) ** 2))


# Hàm clip giá trị
def clip_value(value, min_val=-5.0, max_val=5.0):
    return max(min_val, min(max_val, value))


# Huấn luyện với SGD, Dropout và Early Stopping
train_rmse_history = []
test_rmse_history = []
best_test_rmse = float("inf")
patience = 3  # Giảm patience
min_delta = 0.001  # Ngưỡng cải thiện tối thiểu
patience_counter = 0
best_P = P.copy()
best_Q = Q.copy()

for epoch in range(n_epochs):
    # Giảm dần learning rate
    current_lr = learning_rate * (0.95**epoch)

    # Huấn luyện trên tập train
    for _, row in train_ratings.iterrows():
        u = int(row["user_idx"])
        i = int(row["product_idx"])
        r_ui = row["Rating"]

        # Áp dụng dropout ngẫu nhiên trên yếu tố ẩn
        dropout_mask_P = np.random.binomial(1, 1 - dropout_rate, n_factors).astype(
            float
        ) / (1 - dropout_rate)
        dropout_mask_Q = np.random.binomial(1, 1 - dropout_rate, n_factors).astype(
            float
        ) / (1 - dropout_rate)

        # Dự đoán rating
        prediction = np.dot(P[u, :] * dropout_mask_P, Q[i, :] * dropout_mask_Q)
        prediction = clip_value(prediction)
        error = r_ui - prediction

        # Cập nhật P và Q
        for f in range(n_factors):
            p_update = current_lr * (
                error * Q[i, f] * dropout_mask_Q[f] - reg * P[u, f]
            )
            q_update = current_lr * (
                error * P[u, f] * dropout_mask_P[f] - reg * Q[i, f]
            )
            p_update = clip_value(p_update, -0.5, 0.5)
            q_update = clip_value(q_update, -0.5, 0.5)
            P[u, f] += p_update
            Q[i, f] += q_update

    # Tính RMSE trên tập train
    train_predictions = np.array(
        [
            clip_value(
                np.dot(P[int(row["user_idx"]), :], Q[int(row["product_idx"]), :])
            )
            for _, row in train_ratings.iterrows()
        ]
    )
    train_rmse_value = rmse(train_predictions, train_ratings["Rating"].values)
    train_rmse_history.append(train_rmse_value)

    # Tính RMSE trên tập test
    test_predictions = np.array(
        [
            clip_value(
                np.dot(P[int(row["user_idx"]), :], Q[int(row["product_idx"]), :])
            )
            for _, row in test_ratings.iterrows()
        ]
    )
    test_rmse_value = rmse(test_predictions, test_ratings["Rating"].values)
    test_rmse_history.append(test_rmse_value)

    print(
        f"Epoch {epoch + 1}/{n_epochs} - Train RMSE: {train_rmse_value:.4f} - Test RMSE: {test_rmse_value:.4f} - LR: {current_lr:.6f}"
    )

    # Early Stopping
    if test_rmse_value < best_test_rmse - min_delta:
        best_test_rmse = test_rmse_value
        best_P = P.copy()
        best_Q = Q.copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            P = best_P
            Q = best_Q
            break


# Hàm dự đoán rating
def predict_rating(user_id, product_id):
    if user_id not in user_to_index or product_id not in product_to_index:
        return None
    u = user_to_index[user_id]
    i = product_to_index[product_id]
    return clip_value(np.dot(P[u, :], Q[i, :]))


# Ví dụ dự đoán
user_id_example = 18387707
product_id_example = 192733741
predicted_rating = predict_rating(user_id_example, product_id_example)
if predicted_rating is not None:
    print(
        f"Predicted rating for User {user_id_example} and Product {product_id_example}: {predicted_rating:.2f}"
    )
else:
    print(
        f"Unable to predict for User {user_id_example} and Product {product_id_example}"
    )

# Vẽ biểu đồ RMSE
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_rmse_history) + 1), train_rmse_history, label="Train RMSE")
plt.plot(range(1, len(test_rmse_history) + 1), test_rmse_history, label="Test RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Matrix Factorization Learning Curve with Enhanced Regularization")
plt.legend()
plt.grid(True)
plt.savefig("C:\\Users\\LT64\\Desktop\\DoAnTotNghiep\\03.SourceCode\\rmse_curve.png")
plt.show()


# Hàm gợi ý sản phẩm
def recommend_products(user_id, N=5, exclude_rated=True):
    if user_id not in user_to_index:
        return None
    u = user_to_index[user_id]
    user_vector = P[u, :]
    predicted_ratings = {}
    rated_products = set()
    if exclude_rated:
        user_ratings = train_ratings[train_ratings["CustomerID"] == user_id]
        rated_products = set(user_ratings["ProductID"].values)
    for product_id, i in product_to_index.items():
        if exclude_rated and product_id in rated_products:
            continue
        product_vector = Q[i, :]
        predicted_rating = clip_value(np.dot(user_vector, product_vector))
        predicted_ratings[product_id] = predicted_rating
    sorted_products = sorted(
        predicted_ratings.items(), key=lambda x: x[1], reverse=True
    )
    return sorted_products[:N]


# Ví dụ gợi ý
top_recommendations = recommend_products(user_id_example, N=5)
if top_recommendations:
    print(f"\nTop 5 product recommendations for User {user_id_example}:")
    for i, (product_id, rating) in enumerate(top_recommendations, 1):
        print(f"{i}. Product ID: {product_id}, Predicted Rating: {rating:.2f}")
else:
    print(f"Unable to generate recommendations for User {user_id_example}")
