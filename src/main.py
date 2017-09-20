import json
import time
import math

from collections import defaultdict

import app_data
from src.utils.dataset_utils import DatasetUtils
from src.clf.classify import Classify
from src.utils.preprocessing import Preprocessing


class Main(object):

    def run(self):
        start_time = time.time()
        print("App started")

        # Check if preprocessing should be done for lyrics
        preprocessing_needed = app_data.PREPROCESS_LYRICS

        if preprocessing_needed:
            preprocess_start_time = time.time()
            Preprocessing.preprocess_lyrics()
            preprocess_end_time = time.time()
            elapsed = preprocess_end_time - preprocess_start_time
            print("Preprocessing time: " + str(elapsed))

        split_dataset_needed = app_data.SPLIT_DATASET
        if split_dataset_needed:
            genre_lyrics_map = self._load_lyrics()

            train_lyrics_map = defaultdict(list)
            test_lyrics_map  = defaultdict(list)

            for genre, genre_lyrics in genre_lyrics_map.items():
                size = len(genre_lyrics)
                train_size = math.ceil(size*0.75)
                test_start = train_size + 1

                genre_lyrics_train = genre_lyrics[:train_size]
                genre_lyrics_test  = genre_lyrics[test_start:]

                train_lyrics_map[genre] = genre_lyrics_train
                test_lyrics_map[genre]  = genre_lyrics_test

            print("Train lyrics map: ")
            print(train_lyrics_map)

            print("Test lyrics map: ")
            print(test_lyrics_map)

            DatasetUtils.save_training_set(train_lyrics_map)
            DatasetUtils.save_test_set(test_lyrics_map)

        training_set = DatasetUtils.load_training_set()
        test_set = DatasetUtils.load_test_set()


        print("Loaded lyrics")

        Classify.classify_lyrics(training_set, test_set)

        # self.classify_lyrics_bag_of_words(genre_lyrics_map)

        # self.classify_lyrics_pos(genre_lyrics_map)

        # self.classify_lyrics_mixed_features(genre_lyrics_map)

        # Classify.classify_lyrics_all_features(genre_lyrics_map)

        end_time = time.time()
        running_time = end_time - start_time
        print("Finished main in " + str(running_time) + " seconds")


    def _load_lyrics(self):

        if app_data.DEBUG_MODE:
            lyrics_file_path = app_data.PREPROCESSED_LYRICS_FILE_PATH_TEST
        else:
            lyrics_file_path = app_data.PREPROCESSED_LYRICS_FILE_PATH

        with open(lyrics_file_path, 'r') as lyrics_file:
            genre_lyrics_map = json.load(lyrics_file)

        return genre_lyrics_map



if __name__ == "__main__":
    main = Main()
    main.run()
