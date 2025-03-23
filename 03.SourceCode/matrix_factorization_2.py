import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Read data from CSV file
data = pd.read_csv(
    "C:\\Users\\LT64\\Desktop\\DoAnTotNghiep\\02.Dataset\\rating.csv",
    encoding="utf-8",
)

# Ánh xạ ID của sản phẩm và khách hàng sang chỉ số liên tục
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

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print(f"Số lượng đánh giá trong tập huấn luyện: {len(train_data)}")
print(f"Số lượng đánh giá trong tập kiểm tra: {len(test_data)}")

# Xây dựng ma trận đánh giá từ dữ liệu huấn luyện
n_users = len(unique_customers)
n_items = len(unique_products)
ratings_matrix = np.zeros((n_users, n_items))

for _, row in train_data.iterrows():
    ratings_matrix[row["customer_idx"], row["product_idx"]] = row["Rating"]


# Triển khai Matrix Factorization
class MatrixFactorization:
    def __init__(
        self,
        ratings,
        n_factors=10,
        learning_rate=0.01,
        regularization=0.01,
        iterations=100,
    ):
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations

        # Khởi tạo ngẫu nhiên các ma trận đặc trưng của người dùng và sản phẩm
        self.user_factors = np.random.normal(
            scale=0.1, size=(self.n_users, self.n_factors)
        )
        self.item_factors = np.random.normal(
            scale=0.1, size=(self.n_items, self.n_factors)
        )

        # Khởi tạo biases
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        self.global_bias = np.mean(ratings[np.nonzero(ratings)])

    def predict(self, u, i):
        prediction = self.global_bias + self.user_biases[u] + self.item_biases[i]
        prediction += self.user_factors[u].dot(self.item_factors[i])
        return prediction

    def train(self):
        # Lưu lịch sử MSE để theo dõi quá trình huấn luyện
        self.training_errors = []

        # Tạo mask cho các giá trị khác 0 trong ma trận đánh giá
        mask = self.ratings > 0

        for _ in range(self.iterations):
            # Tính toán dự đoán cho tất cả các cặp user-item
            predicted_ratings = (
                self.global_bias
                + self.user_biases[:, np.newaxis]
                + self.item_biases[np.newaxis, :]
                + self.user_factors.dot(self.item_factors.T)
            )

            # Tính lỗi chỉ trên các đánh giá đã biết
            error = mask * (self.ratings - predicted_ratings)

            # Tính MSE
            mse = (error**2).sum() / mask.sum()
            self.training_errors.append(mse)

            # Cập nhật các tham số qua gradient descent
            for u in range(self.n_users):
                for i in range(self.n_items):
                    if self.ratings[u, i] > 0:
                        err_ui = self.ratings[u, i] - self.predict(u, i)

                        # Cập nhật biases
                        self.user_biases[u] += self.learning_rate * (
                            err_ui - self.regularization * self.user_biases[u]
                        )
                        self.item_biases[i] += self.learning_rate * (
                            err_ui - self.regularization * self.item_biases[i]
                        )

                        # Lưu trữ biến tạm của đặc trưng để cập nhật đồng thời
                        temp_user_factors = self.user_factors[u].copy()

                        # Cập nhật đặc trưng của người dùng và sản phẩm
                        self.user_factors[u] += self.learning_rate * (
                            err_ui * self.item_factors[i]
                            - self.regularization * self.user_factors[u]
                        )
                        self.item_factors[i] += self.learning_rate * (
                            err_ui * temp_user_factors
                            - self.regularization * self.item_factors[i]
                        )

        return self.training_errors

    def get_full_predictions(self):
        """Dự đoán tất cả các đánh giá."""
        return (
            self.global_bias
            + self.user_biases[:, np.newaxis]
            + self.item_biases[np.newaxis, :]
            + self.user_factors.dot(self.item_factors.T)
        )

    def evaluate(self, test_data):
        """Đánh giá mô hình trên dữ liệu kiểm tra."""
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
    ratings_matrix, n_factors=3, learning_rate=0.01, regularization=0.02, iterations=50
)
training_errors = model.train()

# Đánh giá mô hình
rmse = model.evaluate(test_data)
print(f"RMSE trên tập kiểm tra: {rmse:.4f}")

# Vẽ đồ thị quá trình huấn luyện
plt.figure(figsize=(10, 5))
plt.plot(training_errors)
plt.title("Training Error vs. Iterations")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.show()

# Hiển thị một số dự đoán
# full_predictions = model.get_full_predictions()

# print("\nMột số dự đoán đánh giá:")
# print("CustomerID, ProductID, Đánh giá thực tế, Dự đoán")

# for _, row in test_data.iterrows():
#     u = row["customer_idx"]
#     i = row["product_idx"]
#     actual = row["Rating"]
#     predicted = model.predict(u, i)
#     cust_id = idx_to_customer[u]
#     prod_id = idx_to_product[i]
#     print(f"{cust_id}, {prod_id}, {actual}, {predicted:.2f}")


# Đề xuất sản phẩm cho người dùng
def recommend_products_for_user(user_id, n_recommendations=5):
    """Đề xuất n sản phẩm cho người dùng dựa trên dự đoán đánh giá."""
    if user_id not in customer_to_idx:
        print(f"Người dùng có ID {user_id} không tồn tại trong dữ liệu.")
        return

    u = customer_to_idx[user_id]

    # Lấy các sản phẩm mà người dùng chưa đánh giá
    user_ratings = ratings_matrix[u]
    unrated_items = np.where(user_ratings == 0)[0]

    # Dự đoán đánh giá cho các sản phẩm chưa đánh giá
    predicted_ratings = [(i, model.predict(u, i)) for i in unrated_items]

    # Sắp xếp theo đánh giá dự đoán giảm dần
    predicted_ratings.sort(key=lambda x: x[1], reverse=True)

    # Trả về top n sản phẩm được đề xuất
    top_recommendations = predicted_ratings[:n_recommendations]

    print(f"\nĐề xuất top {n_recommendations} sản phẩm cho người dùng {user_id}:")
    print("ProductID, Đánh giá dự đoán")

    for i, rating in top_recommendations:
        prod_id = idx_to_product[i]
        print(f"{prod_id}, {rating:.2f}")


# Thử đề xuất sản phẩm cho một số người dùng
for user_id in [745, 2191, 2204]:
    recommend_products_for_user(user_id, n_recommendations=3)
