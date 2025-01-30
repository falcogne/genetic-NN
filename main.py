import os  # for ENV_VARIABLES variables set by Makefile
import tensorflow as tf
import argparse

from helper_functions import ENV_VARIABLES, set_up_1d_data, set_up_2d_data, create_starter_population, run_for_time, print_best_networks
from evolution_training import EvolutionStructure


BATCH_SIZE = 32
# Won't use this by default, will take from makefile (which is why this needs to be string)
USE_2D_DATA = "True"  
# USE_2D_DATA = "False"
EPOCHS = 1
OPT_STRING = "adam"

POPULATION_SIZE = 4

# allow it to use more epochs as it aproaches the end of training, networks should be better by then
# TODO: don't use this yet but maybe should
EPOCHS_OVER_TIME = [1, 1, 1, 2, 2, 3, 3, 5, 5, 10]


ENV_VARIABLES['OPT_STRING'] = OPT_STRING
ENV_VARIABLES['BATCH_SIZE'] = BATCH_SIZE
ENV_VARIABLES['USE_2D_DATA'] = os.environ.get('USE_2D_DATA', USE_2D_DATA) == "True"
ENV_VARIABLES['EPOCHS'] = EPOCHS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process time parameters.")
    parser.add_argument("--hour", type=int, default=0, help="Hour value (0-23)")
    parser.add_argument("--minute", type=int, default=1, help="Minute value (0-59)")
    parser.add_argument("--second", type=int, default=0, help="Second value (0-59)")

    args = parser.parse_args()

    hour = args.hour
    minute = args.minute
    second = args.second

    print(f"will run for {hour:02} hr {minute:02} min and {second:02} sec")


    if ENV_VARIABLES['USE_2D_DATA']:
        X_train, X_val, y_train, y_val = set_up_2d_data()
        # if len(tf.config.list_physical_devices('GPU')) < 1:
        #     raise ValueError("Tensorflow did not find GPU, but ran in 2D mode")
    else:
        X_train, X_val, y_train, y_val = set_up_1d_data()

    population = create_starter_population(POPULATION_SIZE)

    evolution = EvolutionStructure(
        population,
        X_train,
        y_train,
        X_val,
        y_val,
        optimizer_str=ENV_VARIABLES['OPT_STRING'],
        loss_str=ENV_VARIABLES['LOSS_STR']
    )

    run_for_time(evolution, hour, minute, second)

    print_best_networks(evolution)