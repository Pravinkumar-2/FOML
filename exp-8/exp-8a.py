#FOML EXP8.a
# Step 1 - Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Step 2 - Create dataset
df = pd.DataFrame()
df['X1'] = [1,2,3,4,5,6,6,7,9,9]
df['X2'] = [5,3,6,8,1,9,5,8,9,2]
df['label'] = [1,1,0,1,0,1,0,1,0,0]

# Convert label 0 → -1 for AdaBoost math
df['label'] = df['label'].replace({0: -1})

# Plot data
sns.scatterplot(x='X1', y='X2', hue='label', data=df)
plt.title("Data Distribution")
plt.show()

# Step 3 - Initialize equal weights
df['weights'] = 1 / df.shape[0]

# Step 4 - Train first weak learner
x = df[['X1', 'X2']].values
y = df['label'].values

dt1 = DecisionTreeClassifier(max_depth=1, random_state=42)
dt1.fit(x, y)

plot_decision_regions(x, y, clf=dt1, legend=2)
plt.title("Decision Tree 1 Decision Boundary")
plt.show()

# Step 5 - Predict and calculate weighted error
df['y_pred'] = dt1.predict(x)
error1 = np.sum(df['weights'] * (df['label'] != df['y_pred']))
print("Error 1:", error1)

# Step 6 - Calculate model weight (alpha)
def calculate_model_weight(error):
    # Avoid divide-by-zero or log(∞)
    error = np.clip(error, 1e-10, 1 - 1e-10)
    return 0.5 * np.log((1 - error) / error)

alpha1 = calculate_model_weight(error1)
print("Alpha 1:", alpha1)

# Step 7 - Update and normalize weights
def update_row_weights(row, alpha):
    if row['label'] == row['y_pred']:
        return row['weights'] * np.exp(-alpha)
    else:
        return row['weights'] * np.exp(alpha)

df['updated_weights'] = df.apply(update_row_weights, axis=1, alpha=alpha1)
df['normalized_weights'] = df['updated_weights'] / df['updated_weights'].sum()

# Step 8 - Cumulative weights for sampling
df['cumsum_upper'] = np.cumsum(df['normalized_weights'])
df['cumsum_lower'] = df['cumsum_upper'] - df['normalized_weights']

# Step 9 - Function to create new dataset
def create_new_dataset(df):
    indices = []
    for _ in range(df.shape[0]):
        a = np.random.random()
        for index, row in df.iterrows():
            if row['cumsum_upper'] > a and a > row['cumsum_lower']:
                indices.append(index)
                break
    return indices

index_values = create_new_dataset(df)
second_df = df.iloc[index_values, [0, 1, 2, 3]].copy()

# Step 10 - Train second weak learner
x2 = second_df[['X1', 'X2']].values
y2 = second_df['label'].values
dt2 = DecisionTreeClassifier(max_depth=1, random_state=42)
dt2.fit(x2, y2)

plot_decision_regions(x2, y2, clf=dt2, legend=2)
plt.title("Decision Tree 2 Decision Boundary")
plt.show()

second_df['y_pred'] = dt2.predict(x2)
error2 = np.sum(second_df['weights'] * (second_df['label'] != second_df['y_pred']))
alpha2 = calculate_model_weight(error2)
print("Alpha 2:", alpha2)

# Step 11 - Train third weak learner (simulate new sampling)
third_df = second_df.sample(frac=1, replace=True, weights=second_df['weights'], random_state=42)
x3 = third_df[['X1', 'X2']].values
y3 = third_df['label'].values

dt3 = DecisionTreeClassifier(max_depth=1, random_state=42)
dt3.fit(x3, y3)

plot_decision_regions(x3, y3, clf=dt3, legend=2)
plt.title("Decision Tree 3 Decision Boundary")
plt.show()

third_df['y_pred'] = dt3.predict(x3)
error3 = np.sum(third_df['weights'] * (third_df['label'] != third_df['y_pred']))
alpha3 = calculate_model_weight(error3)
print("Alpha 3:", alpha3)

print("\nModel Weights (Alpha):", alpha1, alpha2, alpha3)

# Step 12 - Final Predictions using Ensemble
def adaboost_predict(query):
    pred1 = dt1.predict(query)[0]
    pred2 = dt2.predict(query)[0]
    pred3 = dt3.predict(query)[0]
    final_score = alpha1 * pred1 + alpha2 * pred2 + alpha3 * pred3
    return np.sign(final_score)

# Test Query 1
query1 = np.array([1, 5]).reshape(1, 2)
print("\nPrediction for [1,5]:", adaboost_predict(query1))

# Test Query 2
query2 = np.array([9, 9]).reshape(1, 2)
print("Prediction for [9,9]:", adaboost_predict(query2))
