import src.app_data as app_data
import json
import time

import src.app_data as app_data
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


        genre_lyrics_map = self._load_lyrics()

        print("Loaded lyrics")

        # self.classify_lyrics_bag_of_words(genre_lyrics_map)

        # self.classify_lyrics_pos(genre_lyrics_map)

        # self.classify_lyrics_mixed_features(genre_lyrics_map)

        Classify.classify_lyrics_all_features(genre_lyrics_map)

        end_time = time.time()
        running_time = end_time - start_time
        print("Finished main in " + str(running_time) + " seconds")


    def _load_lyrics(self):

        if app_data.DEBUG_MODE:
            lyrics_file_path = app_data.PREPROCESSED_LYRICS_FILE_PATH_TEST
        else:
            lyrics_file_path = app_data.PREPROCESSED_LYRICS_FILE_PATH

        with open (lyrics_file_path, 'r') as lyrics_file:
            genre_lyrics_map = json.load(lyrics_file)

        return genre_lyrics_map



if __name__ == "__main__":
    main = Main()
    main.run()
