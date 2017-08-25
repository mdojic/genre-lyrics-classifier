import sys
import src.app_data as app_data
import json
import time
import re
import numpy as np

from src.preprocessing import Preprocessing
from src.lyrics_reader import  LyricsReader
from src.lemma_vectorizer import LemmaVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from nltk.stem import WordNetLemmatizer
from itertools import islice
import nltk

class Main(object):

    def run(self):

        print("App started")

        nltk.download()

        # Check if lyrics need to be read and filtered so only english lyrics are left
        filtering_needed = app_data.FILTER_LYRICS

        # If filtering wasn't done before, do it now and save results on disk
        if filtering_needed:
            genre_lyrics_map = Preprocessing.filter_lyrics()
            self._save_filtered_lyrics(genre_lyrics_map)

        # If filtering was done before, load filtered lyrics from disk
        else:
            genre_lyrics_map = self._load_filtered_lyrics()


        print("Got filtered lyrics")


        # Create vectorizer for extractig bag-of-words representations for lyrics
        # vectorizer = TfidfVectorizer(stop_words='english', max_df=0.4) # 0.45
        # vectorizer = TfidfVectorizer() # 0.448
        # vectorizer = CountVectorizer() # 0.4775, 0.4783
        vectorizer = LemmaVectorizer()

        # Create an array containing all lyrics, and an array with genres of the respective lyrics
        all_lyrics = []
        all_lyrics_genres = []
        for genre in genre_lyrics_map.keys():

            genre_lyrics = genre_lyrics_map[genre]
            print("Lyrics count for genre " + genre + ": " + str(len(genre_lyrics)))

            for song_lyrics in genre_lyrics:
                all_lyrics.append(song_lyrics)
                all_lyrics_genres.append(genre)


        print("Total lyrics: " + str(len(all_lyrics)))

        if app_data.PREPROCESS_LYRICS:
            print("Remove newlines...")
            all_lyrics = Preprocessing.remove_newlines(all_lyrics)
            print("Removed")

        lyrics_train, lyrics_test, genres_train, genres_test = train_test_split(all_lyrics, all_lyrics_genres, test_size=0.33)


        # genres_count = {}
        # for genre in genres_train:
        #     genres_count[genre] = genres_count[genre] + 1 if genres_count.get(genre) != None else 1
        #
        # genres_count_test = {}
        # for genre in genres_test:
        #     genres_count_test[genre] = genres_count_test[genre] + 1 if genres_count_test.get(genre) != None else 1

        print("Train lyrics and train genres count: " + str(len(lyrics_train)) + ", " + str(len(genres_train)))
        print("Test lyrics and test genres count: " + str(len(lyrics_test)) + ", " + str(len(genres_test)))

        print("Fit vectorizer...")

        # First teach the vectorizer the vocabulary of all available lyrics
        vectorizer.fit(all_lyrics)

        print("Vectorizer fit finished")
        print("Teach classifier...")

        partial_fit_classifiers = []
        full_fit_classifiers = []

        # partial_fit_classifiers.append(GaussianNB())          #
        partial_fit_classifiers.append(MultinomialNB())     # 0.455

        # full_fit_classifiers.append(SVC())
        # full_fit_classifiers.append(LinearSVC())

        if len(partial_fit_classifiers) != 0:
            for clf in partial_fit_classifiers:
                self.teach_classifier_in_batches(clf, vectorizer, lyrics_train, genres_train, app_data.LYRICS_GENRES)

        if len(full_fit_classifiers) != 0:
            for clf in full_fit_classifiers:
                self.teach_classifier(clf, vectorizer, lyrics_train, genres_train)

        print("Teaching finished")
        print("Test score on test data")

        if len(partial_fit_classifiers) != 0:
            for clf in partial_fit_classifiers:
                self.test_classifier_in_batches(clf, vectorizer, lyrics_test, genres_test)

        if len(full_fit_classifiers) != 0:
            for clf in full_fit_classifiers:
                score = self.test_classifier(clf, vectorizer, lyrics_test, genres_test)


        classifiers = partial_fit_classifiers + full_fit_classifiers
        for clf in classifiers:
            self.print_top10(vectorizer, clf, app_data.LYRICS_GENRES)


        print("Pickling classifier...")
        joblib.dump(clf, app_data.PICKLE_FILE_PATH)
        print("Pickled!")

        print("Finished main")



    def teach_classifier_in_batches(self, classifier, vectorizer, dataset, classes, available_classes):
        """ Go through the training set, vectorize a batch of data, and teach it to the classifier
            Repeat while there is more data to teach"""

        batch_num = 0
        dataset_batches = self._get_batches(dataset)
        classes_batches = self._get_batches(classes)
        for dataset_batch, classes_batch in zip(dataset_batches, classes_batches):
            batch_num = batch_num + 1
            print("Teaching batch number " + str(batch_num))

            dataset_batch_vect = vectorizer.transform(dataset_batch)
            classifier.partial_fit(dataset_batch_vect.toarray(), classes_batch, available_classes)


    def teach_classifier(self, classifier, vectorizer, dataset, classes ):

        dataset_vect = vectorizer.transform(dataset)
        classifier.fit(dataset_vect, classes)


    def test_classifier_in_batches(self, classifier, vectorizer, dataset, classes):

        dataset_batches = self._get_batches(dataset)
        classes_batches = self._get_batches(classes)

        batch_num = 0
        score_sum = 0
        for dataset_batch, classes_batch in zip(dataset_batches, classes_batches):
            dataset_batch_vect = vectorizer.transform(dataset_batch)
            score = classifier.score(dataset_batch_vect.toarray(), classes_batch)
            score_sum += score

            batch_num = batch_num + 1
            print("Score for batch number " + str(batch_num) + ": " + str(score))

        average_score = score_sum / batch_num
        print("Average score: " + str(average_score))

        return average_score


    def test_classifier(self, classifier, vectorizer, dataset, classes):
        dataset_vect = vectorizer.transform(dataset)
        score = classifier.score(dataset_vect, classes)
        print("Score: ", score)


    # Get generator which returns elements of an iterable object in batches
    def _get_batches(self, l):

        batch_size = app_data.BATCH_SIZE
        list_size = len(l)

        for i in range(0, list_size, batch_size):
            yield l[i:i + batch_size]


    def _save_filtered_lyrics(self, genre_lyrics_map):

        if app_data.DEBUG_MODE:
            output_file_path = app_data.FILTERED_LYRICS_FILE_PATH_TEST
        else:
            output_file_path = app_data.FILTERED_LYRICS_FILE_PATH

        with open(output_file_path, 'w') as output_file:
            json.dump(genre_lyrics_map, output_file)


    def _load_filtered_lyrics(self):

        current_time = self.current_milli_time()

        if app_data.DEBUG_MODE:
            filtered_lyrics_file_path = app_data.FILTERED_LYRICS_FILE_PATH_TEST
        else:
            filtered_lyrics_file_path = app_data.FILTERED_LYRICS_FILE_PATH

        with open (filtered_lyrics_file_path, 'r') as filtered_lyrics_file:
            genre_lyrics_map = json.load(filtered_lyrics_file)

        return genre_lyrics_map


    def current_milli_time(self):
        return int(round(time.time() * 1000))


    def print_top10(self, vectorizer, clf, class_labels):
        """Prints features with the highest coefficient values, per class"""

        if not hasattr(clf, 'coef_'):
            print("Can't find top features for classifier: " )
            print(clf)
            return

        print("Top features for classifier: ")
        print(clf)

        feature_names = vectorizer.get_feature_names()
        for i, class_label in enumerate(class_labels):
            top10 = np.argsort(clf.coef_[i])[-10:]
            print("%s: %s" % (class_label, " ".join(feature_names[j] for j in top10)))


if __name__ == "__main__":
    main = Main()
    main.run()
