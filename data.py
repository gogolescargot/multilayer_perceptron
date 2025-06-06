# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    data.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ggalon <ggalon@student.42lyon.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/04/23 15:34:41 by ggalon            #+#    #+#              #
#    Updated: 2025/06/06 14:26:41 by ggalon           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import click
import pandas as pd


def standardization(dataframe):
    df = dataframe.copy()
    df.iloc[:, 2:] = (df.iloc[:, 2:] - df.iloc[:, 2:].mean()) / df.iloc[
        :, 2:
    ].std()
    return df


@click.command()
@click.option(
    "--path", default="data/data.csv", help="Path of the CSV data file"
)
@click.option(
    "--output_training",
    default="data/data_training.csv",
    help="Output of the CSV training data file",
)
@click.option(
    "--output_validation",
    default="data/data_validation.csv",
    help="Output of the CSV validation data file",
)
@click.option("--percentage", default=80, help="Percentage of training data")
def data_split(path, output_training, output_validation, percentage):
    try:
        if percentage < 1 or percentage > 99:
            raise ValueError("Percentage must be between 1 and 99 percent")

        df = pd.read_csv(path, header=None)
        df = standardization(df)
        cut = int(len(df) / 100 * percentage)

        df_train = df[:cut]
        df_test = df[cut:]

        df_train.to_csv(output_training, index=False, header=None)
        df_test.to_csv(output_validation, index=False, header=None)

        print(
            f"Data processed and saved: {output_training}, {output_validation}"
        )
    except FileNotFoundError:
        print(f"Error: The file at path '{path}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file at path '{path}' is empty or invalid.")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    data_split()
