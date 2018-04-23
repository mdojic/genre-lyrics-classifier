import app_data
import json


def save_training_set(data):

    training_file_path = app_data.get_training_set_file_path()

    with open(training_file_path, 'w') as output_file:
        json.dump(data, output_file)


def save_test_set(data):

    test_file_path = app_data.get_test_set_file_path()

    with open(test_file_path, 'w') as output_file:
        json.dump(data, output_file)


def load_training_set():

    training_file_path = app_data.get_training_set_file_path()

    with open(training_file_path, 'r') as training_file:
        data = json.load(training_file)

    return data


def load_test_set():

    test_file_path = app_data.get_test_set_file_path()

    with open(test_file_path, 'r') as test_file:
        data = json.load(test_file)

    return data



