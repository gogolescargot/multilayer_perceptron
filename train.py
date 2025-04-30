# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ggalon <ggalon@student.42lyon.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/04/16 15:26:53 by ggalon            #+#    #+#              #
#    Updated: 2025/04/30 17:04:52 by ggalon           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle as pkl
import click

class multilayer_perceptron:
	def __init__(self, hidden_layers=[16, 16], learning_rate=0.01, epoch=100):
		if len(hidden_layers) < 2:
			raise ValueError("Neural network must contain at least two hidden layers")
		self.input_size = 30
		self.output_size = 2
		self.hidden_layers = hidden_layers
		self.learning_rate = learning_rate
		
		self.weights = []
		self.biases = []
		
		self.losses_train = []
		self.accuracies_train = []
		self.losses_valid = []
		self.accuracies_valid = []

		self.epoch = epoch
		
		layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
		
		for i in range(len(layer_sizes) - 1):
			self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))
			self.biases.append(np.zeros((1, layer_sizes[i + 1])))

	def feedforward(self, X):
		activations = [X]
		layer = X
		for weight, bias in zip(self.weights, self.biases):
			z = np.dot(layer, weight) + bias
			layer = sigmoid(z)
			activations.append(layer)
		
		return activations
	
	def one_hot_encoding(self, y):
		y_onehot = np.zeros((y.size, self.output_size))
		y_onehot[np.arange(y.size), y.flatten()] = 1
		return y_onehot
	
	def backpropagation(self, X, y):
		activations = self.feedforward(X)
		y_onehot = self.one_hot_encoding(y)

		delta = (activations[-1] - y_onehot) * activations[-1] * (1 - activations[-1])
		nabla_w = [np.dot(activations[-2].T, delta)]
		nabla_b = [np.sum(delta, axis=0, keepdims=True)]

		for layer in range(2, len(self.weights) + 1):
			sp = activations[-layer] * (1 - activations[-layer])
			delta = np.dot(delta, self.weights[-layer + 1].T) * sp
			nabla_w.insert(0, np.dot(activations[-layer - 1].T, delta))
			nabla_b.insert(0, np.sum(delta, axis=0, keepdims=True))

		return nabla_w, nabla_b
	
	def gradient_descent(self, X_train, y_train, X_valid, y_valid):
		for epoch in range(self.epoch):
			nabla_w, nabla_b = self.backpropagation(X_train, y_train)
			for i in range(len(self.weights)):
				self.weights[i] -= self.learning_rate * nabla_w[i]
				self.biases[i] -= self.learning_rate * nabla_b[i]

			output_train = self.feedforward(X_train)[-1]
			self.losses_train.append(binary_cross_entropy(output_train, y_train))
			self.accuracies_train.append(accuracy(choice(output_train), y_train))

			output_valid = self.feedforward(X_valid)[-1]
			self.losses_valid.append(binary_cross_entropy(output_valid, y_valid))
			self.accuracies_valid.append(accuracy(choice(output_valid), y_valid))

			print(f"epoch {epoch + 1:>6}/{self.epoch} - loss: {binary_cross_entropy(output_train, y_train):.4f} - val_loss: {binary_cross_entropy(output_valid, y_valid):.4f}")
		self.save_model()

	def save_model(self, path="model/model.pkl"):
		data = {"weights": self.weights, "biases": self.biases}
		with open("model/model.pkl", "wb") as f_w:
			pkl.dump(data, f_w)

	def load_model(self, path="model/model.pkl"):
		with open(path, "rb") as f_r:
			data = pkl.load(f_r)
		self.weights = data["weights"]
		self.biases = data["biases"]

	def display_graph(self):
		plt.figure()
		plt.plot(range(0, self.epoch), self.losses_train, label="Train loss")
		plt.plot(range(0, self.epoch), self.losses_valid, label="Validation loss")
		plt.title("Loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.legend()
		plt.show()

		plt.figure()
		plt.plot(range(0, self.epoch), self.accuracies_train, label="Train accuracy")
		plt.plot(range(0, self.epoch), self.accuracies_valid, label="Validation accuracy")
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
	return -np.mean(expected_onehot * np.log(output) + (1 - expected_onehot) * np.log(1 - output))

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def softmax(x):
	e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
	return e_x / np.sum(e_x, axis=1, keepdims=True)

def accuracy(predictions, expected):
	return np.mean(predictions == expected)

def data_parse(path):
	X = pd.read_csv(path, header=None, usecols=range(2, 32))
	y = pd.read_csv(path, header=None, usecols=[1]).replace({'B': 0, 'M': 1}).to_numpy()

	return X, y

@click.command
@click.option('--layers', default="16 16", help="Hidden layer structure")
@click.option('--epoch', default="100", help="Number of epochs")
@click.option('--learning_rate', default="0.01")
@click.option('--seed', default="0")
def train(layers, epoch, learning_rate, seed):
	X_train, y_train = data_parse("data/data_training.csv")
	X_valid, y_valid = data_parse("data/data_validation.csv")

	layers = [int(l) for l in layers.split()]
	np.random.seed(int(seed))

	mlp = multilayer_perceptron(hidden_layers=layers, learning_rate=float(learning_rate), epoch=int(epoch))
	mlp.gradient_descent(X_train, y_train, X_valid, y_valid)
	mlp.display_graph()

	print(f"Model trained and saved: model/model.pkl")


if __name__ == '__main__':
	train()