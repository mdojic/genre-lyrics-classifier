import sys
import src.app_data as app_data
import json
import time
import re
import time
import numpy as np

from src.preprocessing import Preprocessing
from src.lyrics_features_extractor import LyricsFeaturesExtractor
from src.lyrics_reader import  LyricsReader
from src.lemma_vectorizer import LemmaVectorizer
from src.lemma_tokenizer import LemmaTokenizer
from src.custom_transformers import LyricsFeatureSelector

from random import shuffle

from sklearn import preprocessing as prep
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from nltk.stem import WordNetLemmatizer
from itertools import islice
import nltk

class Main(object):

    def run(self):

        print("App started")

        # Check if preprocessing should be done for lyrics
        preprocessing_needed = app_data.PREPROCESS_LYRICS

        if preprocessing_needed:
            start_time = time.time()
            Preprocessing.preprocess_lyrics()
            end_time = time.time()
            elapsed = end_time - start_time
            print("Preprocessing time: " + str(elapsed))


        genre_lyrics_map = self._load_lyrics()

        print("Loaded lyrics")

        # self.classify_lyrics_bag_of_words(genre_lyrics_map)

        # self.classify_lyrics_pos(genre_lyrics_map)

        # self.classify_lyrics_mixed_features(genre_lyrics_map)

        self.classify_lyrics_all_features(genre_lyrics_map)

        # print("Pickling classifier...")
        # joblib.dump(clf, app_data.PICKLE_FILE_PATH)
        # print("Pickled!")

        print("Finished main")


    def classify_lyrics_all_features(self, genre_lyrics_map):

        all_lyrics        = []
        all_lyrics_genres = []
        for genre in genre_lyrics_map.keys():

            current_genre_lyrics_count = 0

            genre_lyrics = genre_lyrics_map[genre]
            print("Lyrics count for genre " + genre + ": " + str(len(genre_lyrics)))

            for song_lyrics in genre_lyrics:
                # if current_genre_lyrics_count > 2000:
                #     break

                all_lyrics.append(song_lyrics)
                all_lyrics_genres.append(genre)

        lyrics_train, lyrics_test, genres_train, genres_test = train_test_split(all_lyrics, all_lyrics_genres, test_size=0.33)

        lyrics_features_dict_train = Preprocessing.lyrics_array_to_features_dict(lyrics_train)
        lyrics_features_dict_test  = Preprocessing.lyrics_array_to_features_dict(lyrics_test)

        lyrics_features_extractor = LyricsFeaturesExtractor()

        transformers_union = FeatureUnion(transformer_list=[

                # ('verse_count', Pipeline([
                #                     ('selector', LyricsFeatureSelector(key='verse_count')),
                #                     ('transformer', TfidfVectorizer()),
                #                 ])
                # ),
                #
                # ('stanza_count', Pipeline([
                #                     ('selector', LyricsFeatureSelector(key='stanza_count')),
                #                     ('transformer', TfidfVectorizer())
                #                 ])
                # ),
                #
                # ('avg_verse_length', Pipeline([
                #                         ('selector', LyricsFeatureSelector(key='avg_verse_length')),
                #                         ('transformer', TfidfVectorizer())
                #                     ])
                # ),

                ('pos_tags_map', Pipeline([
                                    ('selector', LyricsFeatureSelector(key='pos_tags_map')),
                                    ('vectorizer', DictVectorizer())
                                    #('transformer', TfidfTransformer())
                                 ])
                ),

                ('lyrics_bow', Pipeline([
                                    ('selector', LyricsFeatureSelector(key='lyrics')),
                                    ('vectorizer', CountVectorizer())
                                    #('transformer', TfidfTransformer())
                                ])
                )

            ], transformer_weights={
                'verse_count'      : 0.5,
                'stanza_count'     : 0.5,
                'avg_verse_length' : 0.8,
                'pos_tags_map'     : 1.0,
                'lyrics_bow'       : 0.3
            }

        )

        pipeline = Pipeline([
            ('extractor', lyrics_features_extractor),
            ('transformes', transformers_union),
            ('clf', SVC(kernel='linear'))
        ])

        pipeline.fit(lyrics_train, genres_train)
        score = pipeline.score(lyrics_test, genres_test)
        print("score = " + str(score))


    def classify_lyrics_mixed_features(self, genre_lyrics_map):

        vectorizer = DictVectorizer()

        all_lyrics_features = []
        all_lyrics_genres   = []

        for genre in genre_lyrics_map.keys():

            genre_lyrics = genre_lyrics_map[genre]

            for song_lyrics in genre_lyrics:

                features = song_lyrics["features"]

                song_features_map = {}

                for feature_name, feature_value in features.items():

                    # pos_tags_map is a dictionary - merge it with song_features_map dictionary
                    if feature_name == "pos_tags_map":
                        song_features_map.update(feature_value)

                    # All other features are numeric, add their name and value as a new key-value pair to the song_features_map
                    else:
                        song_features_map[feature_name] = feature_value

                print("Features: " + str(song_features_map))
                all_lyrics_features.append(song_features_map)
                all_lyrics_genres.append(genre)


        features_train, features_test, genres_train, genres_test = train_test_split(all_lyrics_features, all_lyrics_genres, test_size=0.33)

        vectorizer.fit(all_lyrics_features)

        classifiers_to_use = self.get_classifiers()
        partial_fit_classifiers = classifiers_to_use["partial"]
        full_fit_classifiers = classifiers_to_use["full"]

        self.teach_classifiers(partial_fit_classifiers, full_fit_classifiers, vectorizer, features_train, genres_train, app_data.LYRICS_GENRES)

        self.test_classifiers(partial_fit_classifiers, full_fit_classifiers, vectorizer, features_test, genres_test)

        self.print_top_features(partial_fit_classifiers + full_fit_classifiers, vectorizer, app_data.LYRICS_GENRES)

        self.print_classification_report(full_fit_classifiers[0], vectorizer, features_test, genres_test)


    def classify_lyrics_pos(self, genre_lyrics_map):

        vectorizer = DictVectorizer()

        all_lyrics_pos_tags = []
        all_lyrics_genres   = []

        for genre in genre_lyrics_map.keys():

            genre_lyrics = genre_lyrics_map[genre]

            for song_lyrics in genre_lyrics:

                pos_tags_map = song_lyrics["features"]["pos_tags_map"]

                all_lyrics_pos_tags.append(pos_tags_map)
                all_lyrics_genres.append(genre)

        pos_train, pos_test, genres_train, genres_test = train_test_split(all_lyrics_pos_tags, all_lyrics_genres, test_size=0.33)

        vectorizer.fit(all_lyrics_pos_tags)
        vect = vectorizer.transform((all_lyrics_pos_tags))
        print("vect = " + str(vect))

        classifiers_to_use      = self.get_classifiers()
        partial_fit_classifiers = classifiers_to_use["partial"]
        full_fit_classifiers    = classifiers_to_use["full"]

        self.teach_classifiers(partial_fit_classifiers, full_fit_classifiers, vectorizer, pos_train, genres_train, app_data.LYRICS_GENRES)

        self.test_classifiers(partial_fit_classifiers, full_fit_classifiers, vectorizer, pos_test, genres_test)

        self.print_top_features(partial_fit_classifiers + full_fit_classifiers, vectorizer, app_data.LYRICS_GENRES)



    def classify_lyrics_bag_of_words(self, genre_lyrics_map):

        # Create vectorizer for extractig bag-of-words representations for lyrics
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.4) # 0.45
        # vectorizer = TfidfVectorizer() # 0.448
        # vectorizer = CountVectorizer(stop_words='english', max_df=0.4, max_features=10000) # 0.4775, 0.4783
        # vectorizer = CountVectorizer(stop_words='english', token_pattern=r"(?u)\b[a-zA-Z0-9_'][a-zA-Z0-9_']+\b", max_df=0.1)
        # vectorizer = LemmaVectorizer()
        # vectorizer = CountVectorizer(stop_words='english', tokenizer=LemmaTokenizer(), token_pattern=r"(?u)\b[a-zA-Z0-9_'][a-zA-Z0-9_']+\b", max_features=10000)


        # Create an array containing all lyrics, and an array with genres of the respective lyrics
        all_lyrics = []
        all_lyrics_genres = []
        for genre in genre_lyrics_map.keys():

            current_genre_lyrics_count = 0

            genre_lyrics = genre_lyrics_map[genre]
            print("Lyrics count for genre " + genre + ": " + str(len(genre_lyrics)))

            for song_lyrics in genre_lyrics:

                # if current_genre_lyrics_count > 2000:
                #     break

                all_lyrics.append(song_lyrics["lyrics"])
                all_lyrics_genres.append(genre)
                current_genre_lyrics_count += 1

        print("Total lyrics: " + str(len(all_lyrics)))

        lyrics_train, lyrics_test, genres_train, genres_test = train_test_split(all_lyrics, all_lyrics_genres, test_size=0.33)

        print("Train lyrics and train genres count: " + str(len(lyrics_train)) + ", " + str(len(genres_train)))
        print("Test lyrics and test genres count: " + str(len(lyrics_test)) + ", " + str(len(genres_test)))

        print("Fit vectorizer...")

        # First teach the vectorizer the vocabulary of all available lyrics
        vectorizer.fit(all_lyrics)

        print("Vectorizer fit finished")
        print("Vocabulary_ size: " + str(len(vectorizer.vocabulary_)))
        print("Save vocabulary...")
        file = open('../res/vocab.txt', 'w')
        for word in vectorizer.vocabulary_:
            file.write(word + " | ")


        classifiers_to_use      = self.get_classifiers()
        partial_fit_classifiers = classifiers_to_use["partial"]
        full_fit_classifiers    = classifiers_to_use["full"]

        self.teach_classifiers(partial_fit_classifiers, full_fit_classifiers, vectorizer, lyrics_train, genres_train, app_data.LYRICS_GENRES)

        self.test_classifiers(partial_fit_classifiers, full_fit_classifiers, vectorizer, lyrics_test, genres_test)

        self.print_top_features(partial_fit_classifiers + full_fit_classifiers, vectorizer, app_data.LYRICS_GENRES)

        self.print_classification_report(full_fit_classifiers[0], vectorizer, lyrics_test, genres_test)

        self.print_confusion_matrix(full_fit_classifiers[0], vectorizer, lyrics_test, genres_test)

        self.predict_sample_lyrics_genre(app_data.SB_SAMPLE, full_fit_classifiers[0], vectorizer)




    def predict_sample_lyrics_genre(self, lyrics, classifier, vectorizer):

        vect = vectorizer.transform([lyrics])
        pred = classifier.predict(vect)
        dec  = classifier.decision_function(vect)

        print("Prediction: " + str(pred))
        print("Decision function: " + str(dec))


    def teach_classifiers(self, partial_fit_classifiers, full_fit_classifiers, vectorizer, data, classes, available_classes):

        print("\n\n---\tTeach partial fit classifiers... \n")

        if len(partial_fit_classifiers) != 0:
            for clf in partial_fit_classifiers:
                self.teach_classifier_in_batches(clf, vectorizer, data, classes, available_classes)

        print("\n\n---\tTeach full fit classifiers... \n")

        if len(full_fit_classifiers) != 0:
            for clf in full_fit_classifiers:
                self.teach_classifier(clf, vectorizer, data, classes)


    def test_classifiers(self, partial_fit_classifiers, full_fit_classifiers, vectorizer, test_data, test_classes):

        print("\n\n***\tTest partial classifiers: \n\n")

        if len(partial_fit_classifiers) != 0:
            for clf in partial_fit_classifiers:
                self.test_classifier_in_batches(clf, vectorizer, test_data, test_classes)

        print("\n\n***\tTest full fit classifiers: \n\n")

        if len(full_fit_classifiers) != 0:
            for clf in full_fit_classifiers:
                self.test_classifier(clf, vectorizer, test_data, test_classes)


    def print_top_features(self, classifiers, vectorizer, available_classes, n_features=50):

        print("\n\nTop features: \n\n")
        for clf in classifiers:
            self.print_classifier_top_features(vectorizer, clf, available_classes, n_features)
            print("\n\n")


    def print_classification_report(self, classifier, vectorizer, data, true_classes):

        data = vectorizer.transform(data)
        pred_classes = classifier.predict(data)
        report = classification_report(true_classes, pred_classes)

        print("\n")
        print("-" * 64)
        print("Classification report: ")
        print(report)
        print("-" * 64)
        print("\n")


    def print_confusion_matrix(self, classifier, vectorizer, data, true_classes):

        data = vectorizer.transform(data)
        pred_classes = classifier.predict(data)
        matrix = confusion_matrix(true_classes, pred_classes, labels=["black", "death", "doom", "thrash"])

        print("\n")
        print("-" * 64)
        print("Confusion matrix: " )
        print(matrix)
        print("-" * 64)
        print("\n")

    def get_classifiers(self):

        partial_fit_classifiers = []
        full_fit_classifiers    = []

        # partial_fit_classifiers.append(GaussianNB())          #
        partial_fit_classifiers.append(MultinomialNB())  # 0.455

        # full_fit_classifiers.append(SVC())
        full_fit_classifiers.append(LinearSVC())
        # full_fit_classifiers.append(LogisticRegression(class_weight='balanced'))

        classifiers = {
            "partial" : partial_fit_classifiers,
            "full"    : full_fit_classifiers
        }

        return classifiers


    def teach_classifier_in_batches(self, classifier, vectorizer, dataset, classes, available_classes):
        """ Go through the training set, vectorize a batch of data, and teach it to the classifier
            Repeat while there is more data to teach"""

        batch_num = 0
        dataset_batches = self._get_batches(dataset)
        classes_batches = self._get_batches(classes)
        for dataset_batch, classes_batch in zip(dataset_batches, classes_batches):
            batch_num = batch_num + 1
            print("Teaching batch number " + str(batch_num) + " with size " + str(len(dataset_batch)))

            dataset_batch_vect = vectorizer.transform(dataset_batch)
            # dataset_batch_vect = prep.scale(dataset_batch_vect, with_mean=False)
            classifier.partial_fit(dataset_batch_vect.toarray(), classes_batch, available_classes)


    def teach_classifier(self, classifier, vectorizer, dataset, classes ):

        dataset_vect = vectorizer.transform(dataset)
        # dataset_vect = prep.scale(dataset_vect, with_mean=False)
        classifier.fit(dataset_vect, classes)


    def test_classifier_in_batches(self, classifier, vectorizer, dataset, classes):

        print("*\tTesting classifier: " + str(classifier))

        dataset_batches = self._get_batches(dataset)
        classes_batches = self._get_batches(classes)

        batch_num = 0
        score_sum = 0
        for dataset_batch, classes_batch in zip(dataset_batches, classes_batches):
            dataset_batch_vect = vectorizer.transform(dataset_batch)
            # dataset_batch_vect = prep.scale(dataset_batch_vect, with_mean=False)
            score = classifier.score(dataset_batch_vect.toarray(), classes_batch)
            score_sum += score

            batch_num = batch_num + 1
            print("Score for batch number " + str(batch_num) + ": " + str(score))

        average_score = score_sum / batch_num
        print("Average score: " + str(average_score))

        return average_score


    def test_classifier(self, classifier, vectorizer, dataset, classes):
        print("*\tTesting classifier: " + str(classifier))
        dataset_vect = vectorizer.transform(dataset)
        # dataset_vect = prep.scale(dataset_vect, with_mean=False)
        score = classifier.score(dataset_vect, classes)
        print("Score: ", score)


    # Get generator which returns elements of an iterable object in batches
    def _get_batches(self, l):

        batch_size = app_data.BATCH_SIZE
        list_size = len(l)

        for i in range(0, list_size, batch_size):
            yield l[i:i + batch_size]


    def print_classifier_top_features(self, vectorizer, clf, class_labels, n_features=10):
        """Prints features with the highest coefficient values, per class"""

        if not hasattr(clf, 'coef_'):
            print("Can't find top features for classifier: " )
            print(clf)
            return

        print("Top features for classifier: ")
        print(clf)

        feature_names = vectorizer.get_feature_names()
        for i, class_label in enumerate(class_labels):
            top = np.argsort(clf.coef_[i])[-n_features:]
            print("%s: %s" % (class_label, " ".join(feature_names[j] for j in top)))


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
