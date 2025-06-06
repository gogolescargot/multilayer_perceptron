# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ggalon <ggalon@student.42lyon.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/04/16 15:26:53 by ggalon            #+#    #+#              #
#    Updated: 2025/06/06 14:31:05 by ggalon           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pickle as pkl
import signal

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class multilayer_perceptron:
    def __init__(
        self,
        hidden_layers=[16, 16],
        learning_rate=0.01,
        dropout_rate=0.1,
        epoch=100,
        activation="sigmoid",
    ):
        if len(hidden_layers) < 2:
            raise ValueError(
                "Neural network must contain at least two hidden layers"
            )
        if float(dropout_rate) < 0 or float(dropout_rate) > 1:
            raise ValueError("Dropout rate must be between 0 and 1")

        self.input_size = 30
        self.output_size = 2
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        self.weights = []
        self.biases = []

        self.losses_train = []
        self.accuracies_train = []
        self.losses_valid = []
        self.accuracies_valid = []

        self.epoch = epoch

        layer_sizes = (
            [self.input_size] + self.hidden_layers + [self.output_size]
        )

        if activation == "relu":
            self.activation_function = relu
            self.activation_derivative = relu_derivative
        elif activation == "sigmoid":
            self.activation_function = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            raise ValueError(
                "Unsupported activation function. Choose 'relu' or 'sigmoid'"
            )

        for i in range(len(layer_sizes) - 1):
            if activation == "relu":
                self.weights.append(
                    np.random.randn(layer_sizes[i], layer_sizes[i + 1])
                    * np.sqrt(2 / layer_sizes[i])
                )
            elif activation == "sigmoid":
                self.weights.append(
                    np.random.randn(layer_sizes[i], layer_sizes[i + 1])
                    * np.sqrt(1 / layer_sizes[i])
                )
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

        self.m_weights = []
        self.m_biases = []
        self.v_weights = []
        self.v_biases = []

        for i in range(len(layer_sizes) - 1):
            self.m_weights.append(
                np.zeros((layer_sizes[i], layer_sizes[i + 1]))
            )
            self.m_biases.append(np.zeros((1, layer_sizes[i + 1])))
            self.v_weights.append(
                np.zeros((layer_sizes[i], layer_sizes[i + 1]))
            )
            self.v_biases.append(np.zeros((1, layer_sizes[i + 1])))

    def feedforward(self, X, training=True):
        activations = [X]
        layer = X
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(layer, weight) + bias
            layer = self.activation_function(z)

            if training and i < len(self.weights) - 1:
                dropout_mask = (
                    np.random.rand(*layer.shape) > self.dropout_rate
                ).astype(float)
                layer *= dropout_mask
                layer /= 1 - self.dropout_rate

            activations.append(layer)
        return activations

    def one_hot_encoding(self, y):
        y_onehot = np.zeros((y.size, self.output_size))
        y_onehot[np.arange(y.size), y.flatten()] = 1
        return y_onehot

    def backpropagation(self, X, y):
        activations = self.feedforward(X)
        y_onehot = self.one_hot_encoding(y)

        delta = (activations[-1] - y_onehot) * self.activation_derivative(
            activations[-1]
        )
        nabla_w = [np.dot(activations[-2].T, delta)]
        nabla_b = [np.sum(delta, axis=0, keepdims=True)]

        for layer in range(2, len(self.weights) + 1):
            sp = self.activation_derivative(activations[-layer])
            delta = np.dot(delta, self.weights[-layer + 1].T) * sp
            nabla_w.insert(0, np.dot(activations[-layer - 1].T, delta))
            nabla_b.insert(0, np.sum(delta, axis=0, keepdims=True))

        return nabla_w, nabla_b

    def early_stopping(self, patience=7, decimals=5):
        if len(self.losses_valid) >= patience:
            last_losses = [
                round(loss, decimals) for loss in self.losses_valid[-patience:]
            ]
            for i in range(1, len(last_losses)):
                if last_losses[i] < last_losses[i - 1]:
                    return False
            return True
        return False

    def gradient_descent(self, X_train, y_train, X_valid, y_valid):
        for epoch in range(self.epoch):
            nabla_w, nabla_b = self.backpropagation(X_train, y_train)
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * nabla_w[i]
                self.biases[i] -= self.learning_rate * nabla_b[i]

            output_train = self.feedforward(X_train)[-1]
            self.losses_train.append(
                binary_cross_entropy(output_train, y_train)
            )
            self.accuracies_train.append(
                accuracy(choice(output_train), y_train)
            )

            output_valid = self.feedforward(X_valid)[-1]
            self.losses_valid.append(
                binary_cross_entropy(output_valid, y_valid)
            )
            self.accuracies_valid.append(
                accuracy(choice(output_valid), y_valid)
            )

            if self.early_stopping():
                print(f"Early stopping at epoch {epoch + 1}")
                self.save_model()
                return epoch + 1

            print(
                f"epoch {epoch + 1:>6}/{self.epoch} - loss: {binary_cross_entropy(output_train, y_train):.4f} - val_loss: {binary_cross_entropy(output_valid, y_valid):.4f}"
            )

        self.save_model()
        return self.epoch

    def gradient_descent_adam(self, X_train, y_train, X_valid, y_valid):
        t = 0
        beta1 = 0.9
        beta2 = 0.999
        for epoch in range(self.epoch):
            t += 1
            nabla_w, nabla_b = self.backpropagation(X_train, y_train)
            for i in range(len(self.weights)):
                self.m_weights[i] = (
                    beta1 * self.m_weights[i] + (1 - beta1) * nabla_w[i]
                )
                self.m_biases[i] = (
                    beta1 * self.m_biases[i] + (1 - beta1) * nabla_b[i]
                )

                self.v_weights[i] = beta2 * self.v_weights[i] + (1 - beta2) * (
                    nabla_w[i] ** 2
                )
                self.v_biases[i] = beta2 * self.v_biases[i] + (1 - beta2) * (
                    nabla_b[i] ** 2
                )

                m_hat_w = self.m_weights[i] / (1 - beta1**t)
                v_hat_w = self.v_weights[i] / (1 - beta2**t)
                m_hat_b = self.m_biases[i] / (1 - beta1**t)
                v_hat_b = self.v_biases[i] / (1 - beta2**t)

                self.weights[i] -= (
                    self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + 1e-7)
                )
                self.biases[i] -= (
                    self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + 1e-7)
                )

            output_train = self.feedforward(X_train, training=True)[-1]
            self.losses_train.append(
                binary_cross_entropy(output_train, y_train)
            )
            self.accuracies_train.append(
                accuracy(choice(output_train), y_train)
            )

            output_valid = self.feedforward(X_valid, training=False)[-1]
            self.losses_valid.append(
                binary_cross_entropy(output_valid, y_valid)
            )
            self.accuracies_valid.append(
                accuracy(choice(output_valid), y_valid)
            )

            if self.early_stopping():
                print(f"Early stopping at epoch {epoch + 1}")
                self.save_model()
                return epoch + 1

            print(
                f"epoch {epoch + 1:>6}/{self.epoch} - loss: {binary_cross_entropy(output_train, y_train):.4f} - val_loss: {binary_cross_entropy(output_valid, y_valid):.4f}"
            )

        self.save_model()
        return self.epoch

    def save_model(self, path="model/model.pkl"):
        data = {"weights": self.weights, "biases": self.biases}
        with open("model/model.pkl", "wb") as f_w:
            pkl.dump(data, f_w)

    def load_model(self, path="model/model.pkl"):
        with open(path, "rb") as f_r:
            data = pkl.load(f_r)
        self.weights = data["weights"]
        self.biases = data["biases"]

    def display_graph(self, epoch):
        plt.figure()
        plt.plot(range(0, epoch), self.losses_train, label="Train loss")
        plt.plot(range(0, epoch), self.losses_valid, label="Validation loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(
            range(0, epoch), self.accuracies_train, label="Train accuracy"
        )
        plt.plot(
            range(0, epoch), self.accuracies_valid, label="Validation accuracy"
        )
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()


def choice(input):
    return np.argmax(softmax(input), axis=1, keepdims=True)


def binary_cross_entropy(output, expected):
    n_classes = output.shape[1]
    expected_onehot = np.zeros((expected.size, n_classes))
    expected_onehot[np.arange(expected.size), expected.flatten()] = 1
    output = np.clip(output, 1e-7, 1 - 1e-7)
    return -np.mean(
        expected_onehot * np.log(output)
        + (1 - expected_onehot) * np.log(1 - output)
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def accuracy(predictions, expected):
    return np.mean(predictions == expected)


def data_parse(path):
    X = pd.read_csv(path, header=None, usecols=range(2, 32))
    y = (
        pd.read_csv(path, header=None, usecols=[1])
        .replace({"B": 0, "M": 1})
        .to_numpy()
    )

    return X, y


@click.command
@click.option("--layers", default="16 16", help="Hidden layer structure")
@click.option("--epoch", default="1000", help="Number of epochs")
@click.option("--learning_rate", default="0.01", help="Learning rate")
@click.option("--dropout_rate", default="0", help="Dropout rate")
@click.option("--seed", default="0")
@click.option(
    "--input_train",
    default="data/data_training.csv",
    help="Input data training file",
)
@click.option(
    "--input_valid",
    default="data/data_validation.csv",
    help="Input data validation file",
)
@click.option("--output", default="model/model.pkl", help="Output model file")
@click.option(
    "--activation",
    default="sigmoid",
    help="Activation function (relu or sigmoid)",
)
def train(
    layers,
    epoch,
    learning_rate,
    dropout_rate,
    seed,
    input_train,
    input_valid,
    output,
    activation,
):
    try:
        signal.signal(
            signal.SIGINT,
            lambda *_: (
                print("\033[2DMLP: CTRL+C sent by user."),
                exit(1),
            ),
        )
        X_train, y_train = data_parse(input_train)
        X_valid, y_valid = data_parse(input_valid)

        layers = [int(lyr) for lyr in layers.split()]
        np.random.seed(int(seed))

        mlp = multilayer_perceptron(
            hidden_layers=layers,
            learning_rate=float(learning_rate),
            epoch=int(epoch),
            dropout_rate=float(dropout_rate),
            activation=activation,
        )
        epoch = mlp.gradient_descent_adam(X_train, y_train, X_valid, y_valid)
        mlp.display_graph(epoch)

        print(f"Model trained and saved: {output}")
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    train()
