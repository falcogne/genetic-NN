import time
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evolution_training import EvolutionStructure, create_starter_population_entry
from genetic_network import GeneticNetwork1D, GeneticNetwork2D

import os

# global dict to share variables with main.py
ENV_VARIABLES = {}


def preprocess(sample, image_size, num_classes):
    image = tf.image.resize(sample['image'], image_size) / 255.0  # Normalize
    category = tf.one_hot(sample['label'], depth=num_classes)
    return image, category


def set_up_2d_data():
    batch_size = int(ENV_VARIABLES['BATCH_SIZE'])
    LOSS_STR='categorical_crossentropy'
    ACTIVATION_STR='softmax'

    dataset, info = tfds.load('cifar10', split=['train', 'test'], with_info=True, as_supervised=False)
    OUTPUT_SIZE = 10
    INPUT_SHAPE = (32, 32, 3)
    DATASET_NAME = 'CIFAR-10'

    # dataset, info = tfds.load('cifar100', split=['train', 'test'], with_info=True, as_supervised=False)
    # OUTPUT_SIZE = 100
    # INPUT_SHAPE = (32, 32)
    # DATASET_NAME = 'CIFAR-100'

    # tfds checksum does not pass for this one for some reason
    # dataset, info = tfds.load('caltech101', split=['train', 'test'], with_info=True, as_supervised=False)
    # # OUTPUT_SIZE = 101
    # # INPUT_SHAPE = (128, 128)
    # # DATASET_NAME = 'caltech 101'

    # dataset, info = tfds.load('imagenette', split=['train', 'validation'], with_info=True, as_supervised=False)
    # OUTPUT_SIZE = 10
    # INPUT_SHAPE = (160, 160)
    # DATASET_NAME = 'tiny imagenet'

    # all_imgnet, info = tfds.load('imagenet2012', split='train', shuffle_files=True, as_supervised=False)
    # dataset = all_imgnet.take(5_000)
    # # INPUT_SHAPE = (128, 128)
    # OUTPUT_SIZE=None
    # DATASET_NAME = 'imagenet'

    train_dataset = dataset[0]
    test_dataset = dataset[1]

    # Convert the train dataset to NumPy arrays
    train_images = []
    train_labels = []

    for sample in tfds.as_numpy(train_dataset.map(lambda sample: preprocess(sample, INPUT_SHAPE[:2], OUTPUT_SIZE))):
        train_images.append(sample[0])
        train_labels.append(sample[1])

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # Split the data using train_test_split
    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.13, random_state=42)

    # Create TensorFlow datasets
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Process the test dataset
    test_data = test_dataset.map(lambda sample: preprocess(sample, INPUT_SHAPE[:2], OUTPUT_SIZE)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Print dataset sizes
    print(f"Number of training batches:   {len(train_data)}")
    print(f"Number of validation batches: {len(val_data)}")

    ENV_VARIABLES['OUTPUT_SIZE'] = OUTPUT_SIZE
    ENV_VARIABLES['INPUT_SHAPE'] = INPUT_SHAPE
    ENV_VARIABLES['LOSS_STR'] = LOSS_STR
    ENV_VARIABLES['ACTIVATION_STR'] = ACTIVATION_STR
    ENV_VARIABLES['DATASET_NAME'] = DATASET_NAME
    ENV_VARIABLES['USE_2D_DATA'] = True
    return X_train, X_val, y_train, y_val


def set_up_1d_data():
    LOSS_STR='binary_crossentropy'
    ACTIVATION_STR='sigmoid'

    # Load the data
    DATASET_NAME = "titanic-survivors"
    data = pd.read_csv('titanic-model/train.csv')

    # Fill missing values (this is just an example; you may use other imputation methods)
    data['Age'].fillna(data['Age'].median(), inplace=True)
    # data['Fare'].fillna(data['Fare'].median(), inplace=True)
    # data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    # One-hot encode categorical variables
    categorical_features = ['Sex', 'Embarked']
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(data[categorical_features])

    # Normalize numerical features
    numerical_features = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(data[numerical_features])

    # Combine features
    X = np.concatenate([encoded_features, normalized_features], axis=1)
    y = data['Survived'].values

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    INPUT_SHAPE = X_train.shape[1:]
    OUTPUT_SIZE = 1

    ENV_VARIABLES['OUTPUT_SIZE'] = OUTPUT_SIZE
    ENV_VARIABLES['INPUT_SHAPE'] = INPUT_SHAPE
    ENV_VARIABLES['LOSS_STR'] = LOSS_STR
    ENV_VARIABLES['ACTIVATION_STR'] = ACTIVATION_STR
    ENV_VARIABLES['DATASET_NAME'] = DATASET_NAME
    ENV_VARIABLES['USE_2D_DATA'] = False
    return X_train, X_val, y_train, y_val


def create_starter_population(n:int):
    input_shape = ENV_VARIABLES['INPUT_SHAPE']
    output_size = ENV_VARIABLES['OUTPUT_SIZE']
    activation_str = ENV_VARIABLES['ACTIVATION_STR']

    population = [
        (
        create_starter_population_entry(
            GeneticNetwork2D(input_shape=input_shape, output_features=output_size, output_activation_str=activation_str,)
        ) if ENV_VARIABLES['USE_2D_DATA'] else
        create_starter_population_entry(
            GeneticNetwork1D(input_shape=input_shape, output_features=output_size, output_activation_str=activation_str,)
        )
        ) for _ in range(n)
    ]
    return population


def run_for_time(evolution, hours, minutes, seconds):
    seconds_to_run = hours*60*60+minutes*60+seconds
    epochs = ENV_VARIABLES['EPOCHS']

    start_time = time.time()
    i = 0

    ran_for = time.time() - start_time
    while ran_for < seconds_to_run:
        print(f"\n\n--ITERATION {i} | {ran_for:.2f} seconds in out of {seconds_to_run} --\n\n")
        evolution.iterate_population(train_epochs=epochs)
        i+=1
        ran_for = time.time() - start_time

    print(f"\n   <All done with iterating after {ran_for:.2f} seconds>")


def print_best_networks(evolution):
    print("&"*100)
    print()
    print("Printing results of best networks (will train if untrained)")
    print()
    print("&"*100)
    epochs = ENV_VARIABLES['EPOCHS']

    evolution.train_population(epochs=epochs)
    evolution.calculate_fitness()
    evolution.rank_population()
    for d in evolution.population:
        network, epochs_done, rank, fitness, stats = d['network'], d['training_reps'], d['rank'], d['fitness'], d['stats']
        print()
        print("-*"*80 + "-")
        print(f"\nNetwork Ranked #{rank+1}:")
        print(f"With a fitness of {fitness} after {epochs_done} epochs...")
        print(network)


    print("loss, accuracy, precision, recall, auc")
    for ele in evolution.all_stats:
        print(ele)
    print("\n best fitness stats over iterations:")
    print("loss, accuracy, precision, recall, auc")
    for ele in evolution.best_stats:
        print(ele)