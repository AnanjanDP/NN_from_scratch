import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def binary_cross_entropy(y, y_hat):
    eps = 1e-8
    return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
def generate_data(n=500):
    np.random.seed(42)
    X = np.random.randn(n, 2)
    y = (X[:, 0] * X[:, 1] > 0).astype(int).reshape(-1, 1)
    return X, y
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, lr):
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros((1, 1))
#this is forward propogation
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2
#this is backpropogation
    def backward(self, X, y):
        m = y.shape[0]
        dZ2 = self.A2 - y
        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.mean(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = (X.T @ dZ1) / m
        db1 = np.mean(dZ1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
#this is the training loop
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            y_hat = self.forward(X)
            loss = binary_cross_entropy(y, y_hat)
            self.backward(X, y)

            if epoch % 100 == 0:
                preds = (y_hat > 0.5).astype(int)
                acc = np.mean(preds == y)
                print(f"Epoch: {epoch} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

X, y = generate_data()
model = NeuralNetwork(input_dim=2, hidden_dim=16, lr=0.05)
model.train(X, y, epochs=2000)
