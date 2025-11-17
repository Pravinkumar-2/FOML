#Fomlexo04
import numpy as np

# input data: 4 examples, 2 features each
X = np.array([[0,0],
              [0,1],
              [1,1],
              [1,0]])
# desired outputs (for XOR: 0,1,0,1) – adjust if your mapping differs
y = np.array([[0],
              [1],
              [0],
              [1]])

# hyper-parameters
hidden_neurons = 2
lr = 0.1
epochs = 10000

# activation & derivative
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1.0 - x)

# weights & biases initialization
np.random.seed(1)
W1 = np.random.randn(2, hidden_neurons)  # weights from input to hidden
b1 = np.zeros((1, hidden_neurons))
W2 = np.random.randn(hidden_neurons, 1)  # weights from hidden to output
b2 = np.zeros((1, 1))

# training loop
for epoch in range(epochs):
    # --- forward pass ---
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # compute error
    error = a2 - y
    loss = np.mean(np.square(error))

    # --- backpropagation ---
    d2 = error * sigmoid_deriv(a2)  # output layer delta
    dW2 = np.dot(a1.T, d2)
    db2 = np.sum(d2, axis=0, keepdims=True)

    d1 = np.dot(d2, W2.T) * sigmoid_deriv(a1)  # hidden layer delta
    dW1 = np.dot(X.T, d1)
    db1 = np.sum(d1, axis=0, keepdims=True)

    # update weights & biases
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    # (optional) print loss every some epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, loss = {loss:.6f}")

# after training, test
print("\nFinal outputs:")
for x_input in X:
    hidden = sigmoid(np.dot(x_input, W1) + b1)
    output = sigmoid(np.dot(hidden, W2) + b2)
    print(x_input, output)
