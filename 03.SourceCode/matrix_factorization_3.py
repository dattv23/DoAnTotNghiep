import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# [Phần đọc dữ liệu và xử lý ban đầu giữ nguyên]
data = pd.read_csv(
    "D:\\DoAnTotNghiep\\02.Dataset\\reviews.csv",
    encoding="utf-8",
)

unique_products = data["ProductID"].unique()
unique_customers = data["CustomerID"].unique()

product_to_idx = {pid: i for i, pid in enumerate(unique_products)}
customer_to_idx = {cid: i for i, cid in enumerate(unique_customers)}

idx_to_product = {i: pid for pid, i in product_to_idx.items()}
idx_to_customer = {i: cid for cid, i in customer_to_idx.items()}

data["product_idx"] = data["ProductID"].map(product_to_idx)
data["customer_idx"] = data["CustomerID"].map(customer_to_idx)

print(f"Số lượng sản phẩm: {len(unique_products)}")
print(f"Số lượng khách hàng: {len(unique_customers)}")

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print(f"Số lượng đánh giá trong tập huấn luyện: {len(train_data)}")
print(f"Số lượng đánh giá trong tập kiểm tra: {len(test_data)}")

n_users = len(unique_customers)
n_items = len(unique_products)
ratings_matrix = np.zeros((n_users, n_items))

for _, row in train_data.iterrows():
    ratings_matrix[row["customer_idx"], row["product_idx"]] = row["Rating"]


# Class Matrix Factorization với cập nhật để đo RMSE trên cả train và test
class MatrixFactorization:
    def __init__(
        self,
        ratings,
        test_data,
        n_factors=10,
        learning_rate=0.01,
        regularization=0.01,
        iterations=100,
    ):
        self.ratings = ratings
        self.test_data = test_data  # Thêm tập test vào để đánh giá
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations

        self.user_factors = np.random.normal(
            scale=0.1, size=(self.n_users, self.n_factors)
        )
        self.item_factors = np.random.normal(
            scale=0.1, size=(self.n_items, self.n_factors)
        )

        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        self.global_bias = np.mean(ratings[np.nonzero(ratings)])

    def predict(self, u, i):
        prediction = self.global_bias + self.user_biases[u] + self.item_biases[i]
        prediction += self.user_factors[u].dot(self.item_factors[i])
        return prediction

    def train(self):
        self.train_rmse_history = []  # Lưu RMSE trên tập train
        self.test_rmse_history = []  # Lưu RMSE trên tập test
        mask = self.ratings > 0

        for iteration in range(self.iterations):
            # Tính dự đoán
            predicted_ratings = (
                self.global_bias
                + self.user_biases[:, np.newaxis]
                + self.item_biases[np.newaxis, :]
                + self.user_factors.dot(self.item_factors.T)
            )

            # RMSE trên tập train
            error_train = mask * (self.ratings - predicted_ratings)
            train_rmse = np.sqrt((error_train**2).sum() / mask.sum())
            self.train_rmse_history.append(train_rmse)

            # RMSE trên tập test
            test_predictions = []
            test_actuals = []
            for _, row in self.test_data.iterrows():
                u = row["customer_idx"]
                i = row["product_idx"]
                test_predictions.append(self.predict(u, i))
                test_actuals.append(row["Rating"])
            test_rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
            self.test_rmse_history.append(test_rmse)

            # Cập nhật tham số qua gradient descent
            for u in range(self.n_users):
                for i in range(self.n_items):
                    if self.ratings[u, i] > 0:
                        err_ui = self.ratings[u, i] - self.predict(u, i)
                        self.user_biases[u] += self.learning_rate * (
                            err_ui - self.regularization * self.user_biases[u]
                        )
                        self.item_biases[i] += self.learning_rate * (
                            err_ui - self.regularization * self.item_biases[i]
                        )
                        temp_user_factors = self.user_factors[u].copy()
                        self.user_factors[u] += self.learning_rate * (
                            err_ui * self.item_factors[i]
                            - self.regularization * self.user_factors[u]
                        )
                        self.item_factors[i] += self.learning_rate * (
                            err_ui * temp_user_factors
                            - self.regularization * self.item_factors[i]
                        )

        return self.train_rmse_history, self.test_rmse_history

    def evaluate(self, test_data):
        predictions = []
        actuals = []
        for _, row in test_data.iterrows():
            u = row["customer_idx"]
            i = row["product_idx"]
            predictions.append(self.predict(u, i))
            actuals.append(row["Rating"])
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        return rmse


# Huấn luyện mô hình
model = MatrixFactorization(
    ratings_matrix,
    test_data=test_data,  # Truyền test_data vào
    n_factors=3,
    learning_rate=0.01,
    regularization=0.02,
    iterations=50,
)
train_rmse_history, test_rmse_history = model.train()

# Đánh giá cuối cùng
final_rmse = model.evaluate(test_data)
print(f"RMSE cuối cùng trên tập kiểm tra: {final_rmse:.4f}")

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(train_rmse_history, label="Train RMSE", color="blue")
plt.plot(test_rmse_history, label="Test RMSE", color="orange")
plt.title("RMSE trên tập Train và Test qua các Iteration")
plt.xlabel("Iterations")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.legend()
plt.grid(True)
plt.show()
