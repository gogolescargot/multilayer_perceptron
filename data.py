# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    data.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ggalon <ggalon@student.42lyon.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/04/23 15:34:41 by ggalon            #+#    #+#              #
#    Updated: 2025/04/23 17:01:54 by ggalon           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import click

def standardization(dataframe):
	df = dataframe.copy()
	df.iloc[:, 2:] = (df.iloc[:, 2:] - df.iloc[:, 2:].mean()) / df.iloc[:, 2:].std()
	return df

@click.command()
@click.option('--path', default="data/data.csv", help="Path of the CSV data file")
@click.option('--output_training', default="data/data_training.csv", help="Output of the CSV training data file")
@click.option('--output_validation', default="data/data_validation.csv", help="Output of the CSV valdiation data file")
@click.option('--percentage', default=80, help="Percentage of training data")
def data_split(path, output_training, output_validation, percentage):
	if percentage < 1 or percentage > 99:
		raise ValueError("Percentage must be between 1 and 99 percent")
	
	df = pd.read_csv(path, header=None)
	df = standardization(df)
	cut = int(len(df) / 100 * percentage)

	df_train = df[:cut]
	df_test = df[cut:]

	df_train.to_csv(output_training, index=False, header=None)
	df_test.to_csv(output_validation, index=False, header=None)

	print(f"Data processed and saved: {output_training}, {output_validation}")

if __name__ == '__main__':
	data_split()