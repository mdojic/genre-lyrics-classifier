import json
import time
import math

from collections import defaultdict

import app_data
import src.clf.classify as clf

import src.utils.preprocessing as preprocessing
import src.utils.dataset_utils as dataset_utils

# String to print in the console to separate different outputs
CONSOLE_SEPARATOR = "\n\t***\n"


def run():

    start_time = time.time()
    print("App started")
    print(CONSOLE_SEPARATOR)

    # Check if preprocessing should be done for lyrics
    preprocessing_needed = app_data.PREPROCESS_LYRICS
    if preprocessing_needed:
        preprocess_lyrics()

    # Split dataset into training and test set if needed
    split_dataset_needed = app_data.SPLIT_DATASET
    if split_dataset_needed:
        split_dataset()

    # Load training and test sets
    print("Load lyrics...")
    training_set = dataset_utils.load_training_set()
    test_set     = dataset_utils.load_test_set()

    print("Loaded.")
    print(CONSOLE_SEPARATOR)

    genres = app_data.get_genres()

    # Execute classification
    print("Classify lyrics...")
    clf.classify_and_test_lyrics(training_set, test_set, genres)
    print("Classified.")
    print(CONSOLE_SEPARATOR)

    end_time = time.time()
    running_time = end_time - start_time
    print("Finished main in " + str(running_time) + " seconds.")


def preprocess_lyrics():
    print("Preprocess lyrics...")
    preprocess_start_time = time.time()

    preprocessing.preprocess_lyrics()

    preprocess_end_time = time.time()
    elapsed = preprocess_end_time - preprocess_start_time
    print("Preprocessing finished. Time took: " + str(elapsed))
    print(CONSOLE_SEPARATOR)


def split_dataset():

    print("Split dataset into training and test set...")

    # Load lyrics grouped by genre
    genre_lyrics_map = load_lyrics()

    # Initialize empty maps (dicts) for training and test lyrics-per-genre
    train_lyrics_map = defaultdict(list)
    test_lyrics_map  = defaultdict(list)

    for genre, genre_lyrics in genre_lyrics_map.items():

        # Determine the boundaries of training and test data for current genre lyrics
        size = len(genre_lyrics)
        train_size = math.ceil(size * 0.75)
        test_start = train_size + 1

        # Split current genre's lyrics into training and test data
        genre_lyrics_train = genre_lyrics[:train_size]
        genre_lyrics_test  = genre_lyrics[test_start:]

        # Save training and test lyrics for this genre into lyrics-per-genre maps
        train_lyrics_map[genre] = genre_lyrics_train
        test_lyrics_map[genre]  = genre_lyrics_test

    # Save the split training and test set to project files for later usage
    dataset_utils.save_training_set(train_lyrics_map)
    dataset_utils.save_test_set(test_lyrics_map)

    print("Split finished.")
    print(CONSOLE_SEPARATOR)

    pass


def load_lyrics():

    # Determine whether to load a small portion of data for testing,
    # or to use the full dataset
    lyrics_file_path = app_data.get_preprocessed_lyrics_path()
    with open(lyrics_file_path, 'r') as lyrics_file:
        genre_lyrics_map = json.load(lyrics_file)

    return genre_lyrics_map


if __name__ == "__main__":
    run()
