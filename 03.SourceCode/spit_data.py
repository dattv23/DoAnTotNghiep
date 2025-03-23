import pandas as pd

# Load the data from a CSV file
file_path = (
    "D:\\DoAnTotNghiep\\02.Dataset\\reviews.csv"  # Change this to your actual file path
)
data = pd.read_csv(file_path)

# Group the data by CustomerID and apply the train-test split
train_data = []
test_data = []

for customer_id, group in data.groupby("CustomerID"):
    # Shuffle the ratings of each user and split into train and test
    shuffled_group = group.sample(frac=1, random_state=42)  # Shuffle the group
    split_index = int(len(shuffled_group) * 0.8)  # 80% for training, 20% for testing
    train_data.append(shuffled_group[:split_index])
    test_data.append(shuffled_group[split_index:])

# Combine the train and test data into separate DataFrames
train_data = pd.concat(train_data).reset_index(drop=True)
test_data = pd.concat(test_data).reset_index(drop=True)

# Save the train and test data to CSV files
train_file_path = (
    "D:\\DoAnTotNghiep\\02.Dataset\\train_data.csv"  # Change this to your desired path
)
test_file_path = (
    "D:\\DoAnTotNghiep\\02.Dataset\\test_data.csv"  # Change this to your desired path
)

train_data.to_csv(train_file_path, index=False)
test_data.to_csv(test_file_path, index=False)

print(f"Training data saved to {train_file_path}")
print(f"Testing data saved to {test_file_path}")
