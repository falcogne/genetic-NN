import os  # for ENV_VARIABLES variables set by Makefile

from helper_functions import ENV_VARIABLES, set_up_1d_data, set_up_2d_data, create_starter_population, run_for_time, print_best_networks
from evolution_training import EvolutionStructure


BATCH_SIZE = 32
USE_2D_DATA = "True"  # needs to be string because passed in from Makefile as string so same format here
EPOCHS = 1

HOURS   = 0
MINUTES = 1
SECONDS = 0
POPULATION_SIZE = 3



ENV_VARIABLES['BATCH_SIZE'] = BATCH_SIZE
ENV_VARIABLES['USE_2D_DATA'] = os.environ.get('USE_2D_DATA', USE_2D_DATA) == "True"
ENV_VARIABLES['EPOCHS'] = EPOCHS

if __name__ == "__main__":
    if ENV_VARIABLES['USE_2D_DATA']:
        X_train, X_val, y_train, y_val = set_up_2d_data()
    else:
        X_train, X_val, y_train, y_val = set_up_1d_data()

    print(ENV_VARIABLES)
    population = create_starter_population(POPULATION_SIZE)

    evolution = EvolutionStructure(
        population,
        X_train,
        y_train,
        X_val,
        y_val,
        optimizer_str='adam',
        loss_str=ENV_VARIABLES['LOSS_STR']
    )

    run_for_time(evolution, HOURS, MINUTES, SECONDS)

    print_best_networks(evolution)