# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    prediction.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ggalon <ggalon@student.42lyon.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/04/23 15:34:23 by ggalon            #+#    #+#              #
#    Updated: 2025/04/23 17:01:35 by ggalon           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from train import multilayer_perceptron, binary_cross_entropy, data_parse

import pandas as pd
import click

@click.command()
@click.option('--model', default="model/model.pkl", help="Path of the model file")
@click.option('--validation', default="data/data_validation.csv", help="Path of the CSV validation data file")
def prediction(model, validation):
	X, y = data_parse(validation)
	mlp = multilayer_perceptron()
	mlp.load_model(model)
	output = mlp.feedforward(X)[-1]
	print(f"Error: {binary_cross_entropy(output, y) * 100:2f} %")

if __name__ == '__main__':
	prediction()