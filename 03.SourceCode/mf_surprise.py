"""
Recommendation Engine using Surprise library
Supports SVD, NMF, and KNN algorithms for collaborative filtering
"""

import pandas as pd
import numpy as np
import argparse
import json
import sys
from surprise import Dataset, Reader, SVD, NMF, KNNBasic
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
import pickle
import os
import logging
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("recommendation_engine.log"),
    ],
)


class RecommendationEngine:
    def __init__(self):
        self.model = None
        self.trainset = None
        self.testset = None
        self.data = None

    def load_data(self, file_path=None):
        """Load ratings data from CSV file"""
        try:
            if file_path is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(script_dir, "ratings.csv")

            df = pd.read_csv(file_path)
            reader = Reader(rating_scale=(1, 5))
            self.data = Dataset.load_from_df(
                df[["user_id", "product_id", "rating"]], reader
            )
            return True
        except Exception as e:
            print(json.dumps({"error": f"Error loading data: {str(e)}"}))
            return False

    def train_model(self, algorithm="SVD", parameters=None):
        try:
            """Train the recommendation model"""
            logging.info(f"Starting training with algorithm: {algorithm}")
            logging.info(f"Parameters: {parameters}")

            if self.data is None:
                print("No data loaded")
                return False

            # Split data
            self.trainset, self.testset = train_test_split(self.data, test_size=0.2)

            # Initialize algorithm
            if algorithm == "SVD":
                params = parameters or {
                    "n_factors": 100,
                    "n_epochs": 20,
                    "lr_all": 0.005,
                    "reg_all": 0.02,
                }
                self.model = SVD(**params)
            elif algorithm == "NMF":
                params = parameters or {"n_factors": 50, "n_epochs": 50}
                self.model = NMF(**params)
            elif algorithm == "KNNBasic":
                params = parameters or {
                    "k": 40,
                    "sim_options": {"name": "cosine", "user_based": False},
                }
                self.model = KNNBasic(**params)
            else:
                print(f"Unsupported algorithm: {algorithm}")
                return False

            # Train model
            self.model.fit(self.trainset)

            # Evaluate
            predictions = self.model.test(self.testset)
            rmse = accuracy.rmse(predictions, verbose=False)
            mae = accuracy.mae(predictions, verbose=False)

            # Save model
            model_path = f"model_{algorithm.lower()}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)

            logging.info("Training completed successfully")
            return {
                "algorithm": algorithm,
                "rmse": rmse,
                "mae": mae,
                "model_path": model_path,
            }
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def load_model(self, model_path="model_svd.pkl"):
        """Load pre-trained model"""
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_user_recommendations(
        self, user_id, top_k=10, exclude_purchased=True, file_path=None
    ):
        """Get recommendations for a specific user"""
        if self.model is None:
            print("No model loaded")
            return []

        # Get all product IDs
        if file_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, "ratings.csv")
        df = pd.read_csv(file_path)
        all_products = df["product_id"].unique()

        # Get products user has already rated
        user_products = set(df[df["user_id"] == user_id]["product_id"].values)

        # Get predictions for unrated products
        predictions = []
        for product_id in all_products:
            if exclude_purchased and product_id in user_products:
                continue

            pred = self.model.predict(user_id, product_id)
            predictions.append((product_id, pred.est))

        # Sort by estimated rating and return top K
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [prod_id for prod_id, _ in predictions[:top_k]]

    def get_similar_products(self, product_id, top_k=5):
        """Get products similar to given product"""
        if self.model is None or not hasattr(self.model, "qi"):
            print("Model doesn't support item similarity")
            return []

        try:
            # Get product inner id
            product_inner_id = self.trainset.to_inner_iid(product_id)

            # Calculate similarities
            similarities = []
            for other_inner_id in range(self.trainset.n_items):
                if other_inner_id != product_inner_id:
                    similarity = np.dot(
                        self.model.qi[product_inner_id], self.model.qi[other_inner_id]
                    )
                    other_product_id = self.trainset.to_raw_iid(other_inner_id)
                    similarities.append((other_product_id, similarity))

            # Sort and return top K
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [prod_id for prod_id, _ in similarities[:top_k]]

        except Exception as e:
            print(f"Error getting similar products: {e}")
            return []


def main():
    import traceback

    try:
        parser = argparse.ArgumentParser(description="Recommendation Engine")
        parser.add_argument(
            "--mode", default="recommend", choices=["recommend", "train", "similar"]
        )
        parser.add_argument("--user_id", type=str, help="User ID for recommendations")
        parser.add_argument("--product_id", type=str, help="Product ID for similarity")
        parser.add_argument("--top_k", type=int, default=10)
        parser.add_argument(
            "--algorithm", default="SVD", choices=["SVD", "NMF", "KNNBasic"]
        )
        parser.add_argument(
            "--parameters", type=str, help="Algorithm parameters as JSON"
        )
        parser.add_argument("--exclude_purchased", type=str, default="true")

        args = parser.parse_args()
        engine = RecommendationEngine()

        if args.mode == "train":
            if not engine.load_data():
                raise RuntimeError("Failed to load data for training.")
            parameters = json.loads(args.parameters) if args.parameters else None
            result = engine.train_model(args.algorithm, parameters)
            if not result:
                raise RuntimeError("Model training failed.")
            print(json.dumps(result))

        elif args.mode == "recommend":
            if not engine.load_data():
                raise RuntimeError("Failed to load data for recommendation.")
            if not engine.load_model():
                raise RuntimeError("Failed to load model.")
            exclude = args.exclude_purchased.lower() == "true"
            recommendations = engine.get_user_recommendations(
                args.user_id, args.top_k, exclude
            )
            print(json.dumps(recommendations))

        elif args.mode == "similar":
            if not engine.load_data():
                raise RuntimeError("Failed to load data for similarity.")
            if not engine.load_model():
                raise RuntimeError("Failed to load model.")
            similar_products = engine.get_similar_products(args.product_id, args.top_k)
            print(json.dumps(similar_products))

    except Exception as e:
        print(json.dumps({"error": str(e), "trace": traceback.format_exc()}))
        sys.exit(1)


if __name__ == "__main__":
    main()
