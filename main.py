# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ggalon <ggalon@student.42lyon.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/04/16 15:26:53 by ggalon            #+#    #+#              #
#    Updated: 2025/04/22 22:16:35 by ggalon           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

class multilayer_perceptron:
	def __init__(self, hidden_layers=[16, 16], learning_rate=0.01):
		self.input_size = 30
		self.output_size = 2
		self.hidden_layers = hidden_layers
		self.learning_rate = learning_rate
		
		self.weights = []
		self.biases = []
		
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
	
	def predict(self, input):
		return np.argmax(softmax(input), axis=1, keepdims=True)
	
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
	
	def gradient_descent(self, X, y, epochs):
		for epoch in range(epochs):
			nabla_w, nabla_b = self.backpropagation(X, y)
			for i in range(len(self.weights)):
				self.weights[i] -= self.learning_rate * nabla_w[i]
				self.biases[i] -= self.learning_rate * nabla_b[i]

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def softmax(x):
	e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
	return e_x / np.sum(e_x, axis=1, keepdims=True)

def cost(input, expected):
	return np.mean((input - expected) ** 2)

def standardization(values):
	return (values - values.mean()) / values.std()

def accuracy(predictions, expected):
    return np.mean(predictions == expected)

def main():
	values = pd.read_csv("data.csv", header=None, usecols=range(2, 32))
	validation = pd.read_csv("data.csv", header=None, usecols=[1]).replace({'B': 0, 'M': 1}).to_numpy()
	values_norm = standardization(values).to_numpy()

	mlp = multilayer_perceptron()
	mlp.gradient_descent(values_norm, validation, 1000)
	activations = mlp.feedforward(values_norm)
	predictions = mlp.predict(activations[-1])
	print(f"Accuracy: {accuracy(predictions, validation) * 100:.2f} %")

if __name__ == '__main__':
	main()