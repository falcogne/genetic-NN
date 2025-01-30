import random
import tensorflow as tf
# import tensorflow_addons as tfa
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def create_starter_population_entry(network, model=None):
    return {
        'network': network,
        'model': network.to_keras_model() if model is None else model,
        'compiled': False,
        'training_reps': 0,
        'rank': None,
        'fitness': None,
        'stats': [],
    }

class EvolutionStructure():
    def __init__(self, population:list, X_train, y_train, X_test, y_test, optimizer_str='adam', loss_str='categorical_crossentropy', num_mutations=1):
        self.population = population
        self.all_stats = []
        self.best_stats = []
        self.average_stats = []
        self.population_size = len(self.population)

        self.optimizer_str = optimizer_str
        self.loss_str = loss_str

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.num_mutations = num_mutations

        for d in self.population:
            d['model'].compile(
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
            d['compiled'] = True


    def calculate_fitness(self):
        stat_list = []
        for d in self.population:
            model, epochs_done, rank, fitness, stats = d['model'], d['training_reps'], d['rank'], d['fitness'], d['stats']
            if fitness is not None:
                stat_list.append(stats)
                continue  # already calculated fitness, don't do it again

            # loss, accuracy, precision, recall, auc
            stats = model.evaluate(self.X_test, self.y_test)
            
            loss = stats[0]
            fitness = -loss
            d['fitness'] = fitness
            d['stats'] = stats

            stat_list.append(stats)
        
        self.all_stats.append(stat_list)
        self.best_stats.append(min(stat_list, key=lambda x:x[0]))  # DON'T FORGET THE MIN HERE if you change the stat


    def train_model(self, d, X_train, y_train, epochs):
        model, epochs_done = d['model'], d['training_reps']
        epochs_to_do = epochs - epochs_done
        if epochs_to_do <= 0:
            return d  # already trained

        model.fit(
            X_train,
            y_train,
            epochs=epochs_to_do,
            verbose=1,
        )

        d['training_reps'] = epochs_done + epochs_to_do
        return d


    def train_population(self, epochs=1):
        with ThreadPoolExecutor() as executor:
        # with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self.train_model, d, self.X_train, self.y_train, epochs)
                for d in self.population
            ]

            # Collect results and update population
            self.population = [future.result() for future in futures]


    def kill_population(self, kill_proportion=0.5):
        to_keep = int(self.population_size * (1-kill_proportion))
        self.population = self.population[:to_keep]  # don't need the random removal
        return
        num_kept = 0
        remaining_pop = []
        for d in self.population:
            model, reps_done, rank, fitness = d['model'], d['training_reps'], d['rank'], d['fitness']
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
        start_population_length = len(self.population)
        to_add = self.population_size - start_population_length
        
        for i in range(to_add):
            network_to_copy = self.population[i % start_population_length]['network'].copy()
            try:
                for _ in range(self.num_mutations):
                    network_to_copy.mutate()
            except AttributeError as e:
                print("#"*100)
                print(f"\n\nWARNING: network of type {type(network_to_copy)} cannot mutate")
                print(e)
                print("\n")
                print("#"*100)

            # network_to_copy.build((None,) + network_to_copy.orig_input_shape)
            model = network_to_copy.to_keras_model()

            model.compile(
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
            population_entry = create_starter_population_entry(network_to_copy, model)
            population_entry['compiled'] = True
            self.population.append(population_entry)

        
    def rank_population(self):
        self.population.sort(key=lambda x:x['fitness'], reverse=True)
        for i, d in enumerate(self.population):
            # print(d)
            d['rank'] = i

    def iterate_population(self, train_epochs=20):
        """
        you should use this to run an iteration of the evolutionary algorithm, it will ensure things run in order
        """
        print()
        print()
        print('*'*100)
        for i, d in enumerate(self.population):
            network, epochs_done, rank, fitness, stats = d['network'], d['training_reps'], d['rank'], d['fitness'], d['stats']
            print()
            print(f"<{i}>"*20)
            print(f"network rank {rank} (done with {epochs_done} epochs)")
            print(f"network fitness {fitness}")
            print(str(network))

        print("\n" + "-"*50)
        print("\nTRAINING POPULATION ...")
        self.train_population(epochs=train_epochs)

        print("\n" + "-"*50)
        print("\nEVALUATING POPULATION ...")
        self.calculate_fitness()

        print("\n rank, remove, and replace")
        self.rank_population()
        self.kill_population()
        self.replace_population()

        print("\n*done with iteration*\n")
        # print("population after:")
        # for d in self.population:
        #     d['network'].print_structure()
