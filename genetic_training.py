from malleable_network import GeneticNetwork
import random
import tensorflow as tf
# import tensorflow_addons as tfa

def create_starter_population_entry(network):
    return {
        'network': network,
        'training_reps': 0,
        'rank': None,
        'fitness': None,
    }

class EvolutionStructure():
    def __init__(self, population:list, X_train, y_train, X_test, y_test, optimizer_str='adam', loss_str='categorical_crossentropy'):
        self.population = population
        self.average_fitnesses = []
        self.population_size = len(self.population)

        self.optimizer_str = optimizer_str
        self.loss_str = loss_str

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        for d in self.population:
            d['network'].compile(
                optimizer=self.optimizer_str,
                loss=self.loss_str,
                metrics=[
                    'acc',
                    # tfa.metrics.F1Score(num_classes=OUTPUT_SIZE, average='weighted'),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC()
                ]
            )


    def calculate_fitness(self):
        avg_fitness = 0
        for d in self.population:
            network, reps_done, rank, fitness = d['network'], d['training_reps'], d['rank'], d['fitness']
            if fitness is not None:
                avg_fitness += fitness
                continue  # already calculated fitness, don't do it again

            loss, accuracy, precision, recall, auc = network.evaluate(self.X_test, self.y_test)
            fitness = -loss
            d['fitness'] = fitness

            avg_fitness += fitness
        
        self.average_fitnesses.append(avg_fitness/len(self.population))

    def train_population(self, epochs=1):
        for d in self.population:
            network, epochs_done, rank, fitness = d['network'], d['training_reps'], d['rank'], d['fitness']
            epochs_to_do = epochs - epochs_done
            if epochs_to_do <= 0:
                continue  # already trained don't need to do more

            network.fit(
                self.X_train,
                self.y_train,
                epochs=epochs,
            )

            d['training_reps'] = epochs_done + epochs_to_do

    def kill_population(self, kill_proportion=0.5):
        to_keep = int(self.population_size * (1-kill_proportion))
        print(f"deleted {len(self.population) - to_keep} elements from population")
        self.population = self.population[:to_keep]  # don't need the random removal
        return
        num_kept = 0
        remaining_pop = []
        for d in self.population:
            network, reps_done, rank, fitness = d['network'], d['training_reps'], d['rank'], d['fitness']
            if not isinstance(rank, int):
                raise TypeError(f"rank {rank} is not an integer, run `rank_population()` before this function")
            delete_prob = rank / self.population_size

            if random.random() > delete_prob:  # if it is not being deleted
                num_kept += 1
                remaining_pop.append(d)

            if num_kept >= to_keep:
                break
        
        print(f"deleted {len(self.population) - len(remaining_pop)} elements from population")
        self.population = remaining_pop

    def replace_population(self):
        """
        make the population back to its normal size by copying from the front of the population.
        the idea is that you would have already sorted the population by rank so this takes from the best ones first.
        """
        to_add = self.population_size - len(self.population)
        
        for i in range(to_add):
            network_to_copy = self.population[i]['network'].copy()  # deep copy of GeneticNetwork object
            try:
                network_to_copy.mutate()
            except AttributeError:
                print(f"WARNING: network of type {type(network_to_copy)} cannot mutate")

            print(f"BUILDING WITH ORIG INPUT SHAPE OF {network_to_copy.orig_input_shape}")
            network_to_copy.build((None,) + network_to_copy.orig_input_shape)

            network_to_copy.compile(
                optimizer=self.optimizer_str,
                loss=self.loss_str,
                metrics=[
                    'acc',
                    # tfa.metrics.F1Score(num_classes=OUTPUT_SIZE, average='weighted'),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC()
                ]
            )
            self.population.append(create_starter_population_entry(network_to_copy))
        
        print(f"population is back to {len(self.population)} networks")

    def rank_population(self):
        self.population.sort(key=lambda x:x['fitness'], reverse=True)
        for i, d in enumerate(self.population):
            print(d)
            d['rank'] = i

    def iterate_population(self, train_epochs=20):
        """
        you should use this to run an iteration of the evolutionary algorithm, it will ensure things run in order
        """
        print("population before:")
        for d in self.population:
            d['network'].print_structure()
        self.train_population(epochs=train_epochs)
        self.calculate_fitness()
        self.rank_population()
        self.kill_population()
        self.replace_population()
        print("population after:")
        for d in self.population:
            d['network'].print_structure()
