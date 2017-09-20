import app_data
import json


class DatasetUtils(object):

    @staticmethod
    def save_training_set(data):

        training_file_path = app_data.TRAINING_DATASET_LYRICS_FILE_PATH

        with open(training_file_path, 'w') as output_file:
            json.dump(data, output_file)


    @staticmethod
    def save_test_set(data):

        test_file_path = app_data.TEST_DATASET_LYRICS_FILE_PATH

        with open(test_file_path, 'w') as output_file:
            json.dump(data, output_file)


    @staticmethod
    def load_training_set():

        training_file_path = app_data.TRAINING_DATASET_LYRICS_FILE_PATH

        with open(training_file_path, 'r') as training_file:
            data = json.load(training_file)

        return data


    @staticmethod
    def load_test_set():

        test_file_path = app_data.TEST_DATASET_LYRICS_FILE_PATH

        with open(test_file_path, 'r') as test_file:
            data = json.load(test_file)

        return data

