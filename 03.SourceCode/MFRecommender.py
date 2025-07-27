import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


class MFRecommender:
    def __init__(self, n_factors=20, n_epochs=50, lr=0.005, reg=0.01):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.P = None
        self.Q = None
        self.user_to_index = {}
        self.product_to_index = {}
        self.index_to_user = {}
        self.index_to_product = {}

    def _clip(self, value, min_val=-5.0, max_val=5.0):
        return max(min_val, min(max_val, value))

    def fit(self, train_df, test_df):
        all_users = pd.concat([train_df["userId"], test_df["userId"]]).unique()
        all_products = pd.concat([train_df["productId"], test_df["productId"]]).unique()

        self.user_to_index = {uid: idx for idx, uid in enumerate(all_users)}
        self.product_to_index = {pid: idx for idx, pid in enumerate(all_products)}
        self.index_to_user = {idx: uid for uid, idx in self.user_to_index.items()}
        self.index_to_product = {idx: pid for pid, idx in self.product_to_index.items()}

        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["user_idx"] = train_df["userId"].map(self.user_to_index)
        train_df["product_idx"] = train_df["productId"].map(self.product_to_index)
        test_df["user_idx"] = test_df["userId"].map(self.user_to_index)
        test_df["product_idx"] = test_df["productId"].map(self.product_to_index)

        n_users = len(self.user_to_index)
        n_items = len(self.product_to_index)
        self.P = np.random.normal(0, 0.01, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.01, (n_items, self.n_factors))

        train_rmse, test_rmse = [], []

        for epoch in range(self.n_epochs):
            for _, row in train_df.iterrows():
                u, i = int(row["user_idx"]), int(row["product_idx"])
                r_ui = row["rating"]

                pred = self._clip(np.dot(self.P[u], self.Q[i]))
                err = r_ui - pred

                for f in range(self.n_factors):
                    grad_p = self.lr * (err * self.Q[i][f] - self.reg * self.P[u][f])
                    grad_q = self.lr * (err * self.P[u][f] - self.reg * self.Q[i][f])
                    self.P[u][f] += self._clip(grad_p, -0.5, 0.5)
                    self.Q[i][f] += self._clip(grad_q, -0.5, 0.5)

            train_rmse.append(self.evaluate(train_df))
            test_rmse.append(self.evaluate(test_df))

            print(
                f"Epoch {epoch+1}/{self.n_epochs} - Train RMSE: {train_rmse[-1]:.4f} - Test RMSE: {test_rmse[-1]:.4f}"
            )

        # Learning curve
        plt.plot(range(1, self.n_epochs + 1), train_rmse, label="Train RMSE")
        plt.plot(range(1, self.n_epochs + 1), test_rmse, label="Test RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate(self, df):
        preds = [
            self._clip(
                np.dot(self.P[int(row["user_idx"])], self.Q[int(row["product_idx"])])
            )
            for _, row in df.iterrows()
        ]
        actual = df["rating"].values
        return np.sqrt(np.mean((np.array(preds) - actual) ** 2))

    def recommend_for_user(self, user_id, top_k=10, exclude_seen=True, ratings_df=None):
        if user_id not in self.user_to_index:
            return []

        user_idx = self.user_to_index[user_id]
        seen_items = (
            set(ratings_df[ratings_df["userId"] == user_id]["productId"].values)
            if exclude_seen
            else set()
        )
        scores = []

        for pid, idx in self.product_to_index.items():
            if pid in seen_items:
                continue
            pred_rating = self._clip(np.dot(self.P[user_idx], self.Q[idx]))
            scores.append((pid, pred_rating))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in scores[:top_k]]

    def save_model(self, file_path="mf_model.pkl"):
        with open(file_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load_model(self, file_path="mf_model.pkl"):
        with open(file_path, "rb") as f:
            self.__dict__ = pickle.load(f)
