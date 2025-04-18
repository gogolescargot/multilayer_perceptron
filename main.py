# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ggalon <ggalon@student.42lyon.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/04/16 15:26:53 by ggalon            #+#    #+#              #
#    Updated: 2025/04/18 17:02:43 by ggalon           ###   ########.fr        #
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

	def gradient_descent(self):
		pass

	def backpropagation(self):
		pass

	def feedforward(self, input):
		layer = input
		for weight, bias in zip(self.weights, self.biases):
			layer = sigmoid(np.dot(layer, weight) + bias)
		return layer
	
	def predict(self, input):
		return np.argmax(softmax(input), axis=1, keepdims=True)

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
	validation = pd.read_csv("data.csv", header=None, usecols=[1]).replace({'B': 0, 'M': 1})
	values_norm = standardization(values)

	mlp = multilayer_perceptron()
	output = mlp.feedforward(values_norm.iloc[:, :])
	predict = mlp.predict(output)
	# print(predict)
	print(f"Accuracy: {accuracy(predict, validation) * 100:.2f} %")
	# print(output)
	# print(cost(output, [1, 0]))

if __name__ == '__main__':
	main()