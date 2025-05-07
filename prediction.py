# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    prediction.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ggalon <ggalon@student.42lyon.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/04/23 15:34:23 by ggalon            #+#    #+#              #
#    Updated: 2025/05/07 14:30:17 by ggalon           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from train import multilayer_perceptron, binary_cross_entropy, accuracy, choice, data_parse

import click
import numpy as np

def precision_recall_f1(predictions, expected):
	tp = np.sum((predictions == 1) & (expected == 1))
	fp = np.sum((predictions == 1) & (expected == 0))
	fn = np.sum((predictions == 0) & (expected == 1))
	precision = tp / (tp + fp + 1e-7)
	recall = tp / (tp + fn + 1e-7)
	f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
	return precision, recall, f1

@click.command()
@click.option('--model', default="model/model.pkl", help="Path of the model file")
@click.option('--validation', default="data/data_validation.csv", help="Path of the CSV validation data file")
def prediction(model, validation):
	X, y = data_parse(validation)
	mlp = multilayer_perceptron()
	mlp.load_model(model)
	output = mlp.feedforward(X)[-1]
	predictions = choice(output)
	precision, recall, f1 = precision_recall_f1(predictions, y)
	print(f"Precision: {precision * 100:2f} %")
	print(f"Recall: {recall * 100:2f} %")
	print(f"F1 Score: {f1 * 100:2f} %")
	print(f"Error: {binary_cross_entropy(output, y) * 100:2f} %")
	print(f"Accuracy: {accuracy(choice(output), y) * 100:2f} %")

if __name__ == '__main__':
	prediction()