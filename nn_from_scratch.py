# x_train = x_train.reshape((num_examples, 10000))
# y_train = keras.utils.to_categorical(y_train, num_classes)

# TODOs
# activation function for the last layer could be softmax
# activation function for all other layers could be relu
# tune the hyperparameters in a better way

# activation function
def sigmoid(x, deriv=False):
    if deriv:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


class Layer:

    def __init__(self, num_input, num_output):
        self.weights = np.random.randn(num_input, num_output)
        self.biases = np.zeros((1, num_output))
        self.cache = None
        self.deltas = None

    def activation(self, inputs):
        self.cache = sigmoid(np.dot(inputs, self.weights) + self.biases)
        return self.cache


class NeuralNetwork:

    def __init__(self, x_train, y_train, dimentions, epochs, learning_rate):
        # normalize x_train?
        self.x_train = x_train
        self.y_train = y_train
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.layers = []
        for i in range(len(dimentions) - 1):
            self.layers.append(Layer(dimentions[i], dimentions[i + 1]))

    def cross_entropy(self, pred):
        return (pred - self.y_train) / self.y_train.shape[0]

    def error(self, pred):
        n = self.y_train.shape[0]
        logp = - np.log(pred[np.arange(n), self.y_train.argmax(axis=1)])
        loss = np.sum(logp) / n
        return loss

    def forward(self, x):
        cache = self.layers[0].activation(x)
        for layer in self.layers[1:]:
            cache = layer.activation(cache)
        return cache

    def backprop(self, x):
        loss = self.error(self.layers[-1].cache)
        print(loss)

        self.layers[-1].deltas = self.cross_entropy(self.layers[-1].cache)
        z_delta = np.dot(self.layers[-1].deltas, self.layers[-1].weights.T)

        for layer in self.layers[-2:0:-1]:
            layer.deltas = z_delta * sigmoid(layer.cache, deriv=True)
            z_delta = np.dot(layer.deltas, layer.weights.T)

        self.layers[0].deltas = z_delta * sigmoid(self.layers[0].cache, deriv=True)

        # update
        self.layers[0].weights -= self.learning_rate * np.dot(x.T, self.layers[0].deltas)
        self.layers[0].biases -= self.learning_rate * np.sum(self.layers[0].deltas, axis=0)
        cache = self.layers[0].cache

        for layer in self.layers[1:]:
            layer.weights -= self.learning_rate * np.dot(cache.T, layer.deltas)
            layer.biases -= self.learning_rate * np.sum(layer.deltas, axis=0)
            cache = layer.cache

    def predict(self, x):
        return np.apply_along_axis(np.argmax, 1, self.forward(x))

    def fit(self):
        for i in range(1, self.epochs + 1):
            print("Epoch: {}".format(i))
            self.forward(self.x_train)
            self.backprop(self.x_train)




model = NeuralNetwork(x_train, y_train, [100 ** 2, 1000, 500, 200, 31], 200, 0.05)
model.fit()

print(model.predict(x_train))
print(y_raw.astype(int))


# accuracy

# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_raw.astype(int), model.predict(x_train)))
# print(accuracy_score(y_valid_raw.astype(int), model.predict(x_valid)))
