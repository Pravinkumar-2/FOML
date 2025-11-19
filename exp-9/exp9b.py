#FOMLEXP9.b
import numpy as np
import pandas as pd
from math import sqrt

# Step 1: Load dataset
data = pd.read_csv('/content/KNNAlgorithmDataset.csv')   # <-- change path if needed
print("Original Data:")
print(data.head(5))

# Step 2: Use all columns except the first (assuming first is an ID column)
# Separate features and target
X = data.iloc[:, 2:-1].values # Features (exclude id, diagnosis, and last unnamed column)
y = data.iloc[:, 1].values # Target (diagnosis column)

print("\nFeatures (X):")
print(X[:5])
print("\nTarget (y):")
print(y[:5])

# Step 3: Combine features and target for shuffling
combined_data = np.c_[X, y]

# Shuffle rows
shuffle_index = np.random.permutation(combined_data.shape[0])
shuffled_data = combined_data[shuffle_index]

print("\nAfter Shuffling (first 5 rows of combined data):")
print(shuffled_data[:5])

# Step 4: Split into train and test (70%-30%)
train_size = int(shuffled_data.shape[0] * 0.7)
train_data = shuffled_data[:train_size]
test_data = shuffled_data[train_size:]

train_X = train_data[:, :-1] # Features for training
train_y = train_data[:, -1] # Target for training

test_X = test_data[:, :-1] # Features for testing
test_y = test_data[:, -1] # Target for testing

print('\nTrain_X Shape:', train_X.shape)
print('Train_y Shape:', train_y.shape)
print('Test_X Shape:', test_X.shape)
print('Test_y Shape:', test_y.shape)


# -----------------------------------------------------------
# Step 5: KNN Helper Functions
# -----------------------------------------------------------

def euclidean_distance(x_test_row, x_train_row):
    distance = 0
    # Iterate through features (excluding the target column)
    for i in range(len(x_test_row)):
        # Ensure values are numeric before subtraction
        distance += (float(x_test_row[i]) - float(x_train_row[i])) ** 2
    return sqrt(distance)


def get_neighbors(x_test_row, x_train_data, num_neighbors):
    distances = []
    for train_row in x_train_data:
        dist = euclidean_distance(x_test_row, train_row)
        distances.append((train_row, dist))

    # Sort by distance
    distances.sort(key=lambda x: x[1])

    neighbors = [item[0] for item in distances[:num_neighbors]]
    return neighbors


def prediction(x_test_row, x_train_data, num_neighbors):
    neighbors = get_neighbors(x_test_row, x_train_data, num_neighbors)
    classes = [neighbor[-1] for neighbor in neighbors] # class label = last column of neighbor row
    predicted = max(classes, key=classes.count)  # most common class
    return predicted


def accuracy(y_true, y_pred):
    num_correct = sum(y_true[i] == y_pred[i] for i in range(len(y_true)))
    return num_correct / len(y_true)


# -----------------------------------------------------------
# Step 6: Run predictions
# -----------------------------------------------------------

y_pred = []
for test_row in test_data:
    y_pred.append(prediction(test_row[:-1], train_data, 5)) # Pass only features for distance calculation


# Step 7: Evaluate
acc = accuracy(test_y, y_pred)
print("\nPredicted values (first 10):", y_pred[:10])
print("True values (first 10):", list(test_y[:10]))
print(f"\nModel Accuracy: {acc * 100:.2f}%")
