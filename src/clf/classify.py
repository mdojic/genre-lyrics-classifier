import joblib
import numpy as np
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import app_data as app_data
import src.clf.custom_transformers as ct
import src.utils.preprocessing as preprocessing
import src.utils.drawing_utils as dutils

from src.clf.custom_transformers import BasicLyricsFeaturesExtractor
from src.clf.custom_transformers import LyricsFeatureSelector



def predict_lyrics_genre(lyrics, metal_only=True):
    """
    Predicts genre of the given lyrics.

    :param lyrics: The lyrics for which to predict genre
    :param metal_only: Whether classification should be done regarding metal subgenres, or bigger genres
    :return str: The predicted genre for the gi5e6uy7tr ven lyrics
    """

    lyrics_dict = preprocessing.get_lyrics_dict([lyrics])

    pickled_clf_path = app_data.PICKLE_FILE_PATH_METAL if metal_only else app_data.PICKLE_FILE_PATH_ALL
    clf = joblib.load(pickled_clf_path)
    genre = clf.predict([lyrics_dict])
    return genre


def classify_and_test_lyrics(training_set, test_set, genres):

    lyrics_train, lyrics_test, genres_train, genres_test = [], [], [], []

    # Go through all available genres
    for genre in genres:

        current_genre_lyrics_count = 0

        # Get training and test lyrics for current genre
        genre_lyrics_train = training_set[genre]
        genre_lyrics_test  = test_set[genre]
        genre_lyrics = genre_lyrics_train + genre_lyrics_test
        print("Lyrics count for genre " + genre + ": " + str(len(genre_lyrics)))
        print("Train lyrics count for genre {}: {}".format(genre, len(genre_lyrics_train)))
        print("Test lyrics count for genre {}: {}".format(genre, len(genre_lyrics_test)))
        print("\n")

        # Whether to limit lyrics count per genre or not
        limit_lyrics_count = app_data.LIMIT_LYRICS_COUNT

        # Append current genre's training lyrics to training set, and the genre to training classes
        for song_lyrics in genre_lyrics_train:
            if limit_lyrics_count and current_genre_lyrics_count > 1500:
                break

            lyrics_train.append(song_lyrics)
            genres_train.append(genre)

            current_genre_lyrics_count += 1

        current_genre_lyrics_count = 0

        # Append current genre's test lyrics to test set, and the genre to test classes
        for song_lyrics in genre_lyrics_test:
            if current_genre_lyrics_count > 1500:
                break

            lyrics_test.append(song_lyrics)
            genres_test.append(genre)

            current_genre_lyrics_count += 1

    # Train classifier if needed; otherwise load pickled classifier
    if app_data.TRAIN_CLASSIFIER:

        # Get transformers for extracting features from lyrics
        transformers_union = _get_lyrics_vectorizers()

        # Build a pipeline which extracts features from lyrics, and then classifies them
        pipeline = Pipeline([
            ('transformers', transformers_union),
            ('clf', LinearSVC())
        ])

        # gs_clf = GridSearchCV(pipeline, parameters, n_jobs=1)
        # Teach the classifier
        print("Teach classifier...")
        pipeline.fit(lyrics_train, genres_train)
        # gs_clf.fit(lyrics_train, genres_train)

        # Pickle taught classifier
        # pickled_clf_path = app_data.PICKLE_FILE_PATH_METAL if metal_only else app_data.PICKLE_FILE_PATH_ALL
        pickled_clf_path = app_data.get_classifier_pickle_path()
        joblib.dump(pipeline, pickled_clf_path)
        print("\n\n\t\t DUMPED PIPELINE CLASSIFIER \n\n")
    else:
        # pickled_clf_path = app_data.PICKLE_FILE_PATH_METAL if metal_only else app_data.PICKLE_FILE_PATH_ALL
        pickled_clf_path = app_data.get_classifier_pickle_path()
        pipeline = joblib.load(pickled_clf_path)

    print("Teaching finished.")
    print("Test classifier...")
    # Test the precision of the taught classifier on the test set
    print("\n pipeline: ")
    print(pipeline)
    print("\n")
    score = pipeline.score(lyrics_test, genres_test)
    pipeline.score
    print("Done.")
    print("Score = " + str(score))

    tf = pipeline.named_steps.transformers
    clf = pipeline.named_steps.clf

    labels = app_data.get_genres()

    vectorized_lyrics = tf.transform(lyrics_test)
    predicted_classes = clf.predict(vectorized_lyrics)
    true_classes = genres_test

    print_confusion_matrix(true_classes, predicted_classes, labels=labels)
    print_classification_report(true_classes, predicted_classes)


def _get_lyrics_vectorizers():

    union = FeatureUnion(
        transformer_list=[

            ('verse_count', Pipeline([
                                ('extractor', ct.LyricsVerseCountVectorizer()),
                            ])
            ),

            ('stanza_count', Pipeline([
                                ('extractor', ct.LyricsStanzaCountVectorizer()),
                             ])
            ),

            ('avg_verse_words', Pipeline([
                                    ('extractor', ct.LyricsAvgVerseWordCountVectorizer())
                                ])
            ),

            ('word_count', Pipeline([
                                ('extractor', ct.LyricsWordCountVectorizer())
                           ])
            ),

            ('pos_tags_map', Pipeline([
                                ('extractor', ct.LyricsPartOfSpeechVectorizer()),
                                ('vectorizer', DictVectorizer())
                             ])
            ),

            ('word_endings', Pipeline([
                                ('extractor', ct.LyricsWordEndingsVectorizer()),
                                ('vectorizer', DictVectorizer()),
                                ('transformer', TfidfTransformer())
                             ])
            ),

            ('lyrics_bow', Pipeline([
                                ('vectorizer', TfidfVectorizer(stop_words='english', max_df=0.6))
                           ])
            )

        ],

        transformer_weights={
            'verse_count'     : 2,
            'stanza_count'    : 2,
            'avg_verse_words' : 3,
            'word_count'      : 3,
            'pos_tags_map'    : 10,
            'word_endings'    : 7,
            'lyrics_bow'      : 10
        }
    )
    print("\n\n\t\t GOT THE VECTORIZER UNION \n\n")
    return union


def print_confusion_matrix(true_classes, predicted_classes, labels=None, normalize=False):

    matrix = confusion_matrix(true_classes, predicted_classes, labels)

    if labels is not None:
        dutils.plot_confusion_matrix(matrix, labels, title="Genre confusion matrix", normalize=normalize)

    print("\n")
    print("-" * 64)
    print("Confusion matrix: " )
    print(matrix)
    print("-" * 64)
    print("\n")


def print_classification_report(true_classes, predicted_classes, labels=None):

    report = classification_report(true_classes, predicted_classes, labels=labels)

    print("\n")
    print("-" * 64)
    print("Classification report: ")
    print(report)
    print("-" * 64)
    print("\n")



#######################################################################################################################
#                                               OLD FUNCTIONS                                                         #
#######################################################################################################################
def classify_lyrics_all_features(genre_lyrics_map, metal_only=True):

    all_lyrics        = []
    all_lyrics_genres = []
    for genre in genre_lyrics_map.keys():

        current_genre_lyrics_count = 0

        genre_lyrics = genre_lyrics_map[genre]
        print("Lyrics count for genre " + genre + ": " + str(len(genre_lyrics)))

        for song_lyrics in genre_lyrics:
            # if current_genre_lyrics_count > 2700:
            #     break

            all_lyrics.append(song_lyrics)
            all_lyrics_genres.append(genre)

    lyrics_train, lyrics_test, genres_train, genres_test = train_test_split(all_lyrics, all_lyrics_genres, test_size=0.25)

    # lyrics_features_extractor = AdvancedLyricsFeaturesExtractor()

    # transformers_union = _get_advanced_features_transformator_union()
    #
    # pipeline = Pipeline([
    #     ('extractor', lyrics_features_extractor),
    #     ('transformes', transformers_union),
    #     ('clf', LinearSVC())
    # ])

    if app_data.TRAIN_CLASSIFIER:

        lyrics_features_extractor = BasicLyricsFeaturesExtractor()

        transformers_union = _get_basic_features_transformator_union()

        pipeline = Pipeline([
            ('extractor', lyrics_features_extractor),
            ('transformers', transformers_union),
            ('clf', LinearSVC(dual=False, C=0.9))
        ])

        # gs_clf = GridSearchCV(pipeline, parameters, n_jobs=1)
        print("Teach...")
        pipeline.fit(lyrics_train, genres_train)
        #
        # gs_clf.fit(lyrics_train, genres_train)

    else:
        pickled_clf_path = app_data.PICKLE_FILE_PATH_METAL if metal_only else app_data.PICKLE_FILE_PATH_ALL
        pipeline = joblib.load(pickled_clf_path)

    print("Done.")
    print("Test...")
    score = pipeline.score(lyrics_test, genres_test)
    print("Done.")
    print("Score = " + str(score))
    # extracted = pipeline.named_steps['extractor'].transform([lyrics_test[0]])
    # transformed = pipeline.named_steps['transformers'].transform(extracted)
    # pred = pipeline.predict(transformed)
    pred = pipeline.predict([lyrics_test[0]])
    print("Pred: ")
    print(pred)


    # score = gs_clf.score(lyrics_test, genres_test, n_splits=1)
    # print("Score = " + str(score))
    # print("Decision function: ")
    # print(pipeline.decision_function)
    #
    # print("\n---\tBest score: ")
    # print(gs_clf.best_score_)
    #
    # print("\n---\tBest params: ")
    # for param_name in sorted(parameters.keys()):
    #     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    print("Pickling classifier...")
    pickled_clf_path = app_data.PICKLE_FILE_PATH_METAL if metal_only else app_data.PICKLE_FILE_PATH_ALL
    joblib.dump(pipeline, pickled_clf_path)
    print("Pickled!")


def classify_lyrics_mixed_features( genre_lyrics_map):

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

    classifiers_to_use = get_classifiers()
    partial_fit_classifiers = classifiers_to_use["partial"]
    full_fit_classifiers = classifiers_to_use["full"]

    teach_classifiers(partial_fit_classifiers, full_fit_classifiers, vectorizer, features_train, genres_train, app_data.LYRICS_GENRES_METAL)

    test_classifiers(partial_fit_classifiers, full_fit_classifiers, vectorizer, features_test, genres_test)

    print_top_features(partial_fit_classifiers + full_fit_classifiers, vectorizer, app_data.LYRICS_GENRES_METAL)

    print_classification_report(full_fit_classifiers[0], vectorizer, features_test, genres_test)


def classify_lyrics_pos(genre_lyrics_map):

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

    classifiers_to_use      = get_classifiers()
    partial_fit_classifiers = classifiers_to_use["partial"]
    full_fit_classifiers    = classifiers_to_use["full"]

    teach_classifiers(partial_fit_classifiers, full_fit_classifiers, vectorizer, pos_train, genres_train, app_data.LYRICS_GENRES_METAL)

    test_classifiers(partial_fit_classifiers, full_fit_classifiers, vectorizer, pos_test, genres_test)

    print_top_features(partial_fit_classifiers + full_fit_classifiers, vectorizer, app_data.LYRICS_GENRES_METAL)


def classify_lyrics_bag_of_words(genre_lyrics_map):

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

            if current_genre_lyrics_count > 2000:
                break

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


    classifiers_to_use      = get_classifiers()
    partial_fit_classifiers = classifiers_to_use["partial"]
    full_fit_classifiers    = classifiers_to_use["full"]

    teach_classifiers(partial_fit_classifiers, full_fit_classifiers, vectorizer, lyrics_train, genres_train, app_data.LYRICS_GENRES_METAL)

    test_classifiers(partial_fit_classifiers, full_fit_classifiers, vectorizer, lyrics_test, genres_test)

    print_top_features(partial_fit_classifiers + full_fit_classifiers, vectorizer, app_data.LYRICS_GENRES_METAL)

    print_classification_report(full_fit_classifiers[0], vectorizer, lyrics_test, genres_test)

    print_confusion_matrix(full_fit_classifiers[0], vectorizer, lyrics_test, genres_test)

    predict_sample_lyrics_genre(app_data.SB_SAMPLE, full_fit_classifiers[0], vectorizer)


def predict_sample_lyrics_genre(lyrics, classifier, vectorizer):

    vect = vectorizer.transform([lyrics])
    pred = classifier.predict(vect)
    dec  = classifier.decision_function(vect)

    print("Prediction: " + str(pred))
    print("Decision function: " + str(dec))


def teach_classifiers(partial_fit_classifiers, full_fit_classifiers, vectorizer, data, classes, available_classes):

    print("\n\n---\tTeach partial fit classifiers... \n")

    if len(partial_fit_classifiers) != 0:
        for clf in partial_fit_classifiers:
            teach_classifier_in_batches(clf, vectorizer, data, classes, available_classes)

    print("\n\n---\tTeach full fit classifiers... \n")

    if len(full_fit_classifiers) != 0:
        for clf in full_fit_classifiers:
            teach_classifier(clf, vectorizer, data, classes)


def test_classifiers(partial_fit_classifiers, full_fit_classifiers, vectorizer, test_data, test_classes):

    print("\n\n***\tTest partial classifiers: \n\n")

    if len(partial_fit_classifiers) != 0:
        for clf in partial_fit_classifiers:
            test_classifier_in_batches(clf, vectorizer, test_data, test_classes)

    print("\n\n***\tTest full fit classifiers: \n\n")

    if len(full_fit_classifiers) != 0:
        for clf in full_fit_classifiers:
            test_classifier(clf, vectorizer, test_data, test_classes)


def print_top_features(classifiers, vectorizer, available_classes, n_features=50):

    print("\n\nTop features: \n\n")
    for clf in classifiers:
        print_classifier_top_features(vectorizer, clf, available_classes, n_features)
        print("\n\n")


# def print_classification_report(classifier, vectorizer, data, true_classes):
#
#     data = vectorizer.transform(data)
#     pred_classes = classifier.predict(data)
#     report = classification_report(true_classes, pred_classes)
#
#     print("\n")
#     print("-" * 64)
#     print("Classification report: ")
#     print(report)
#     print("-" * 64)
#     print("\n")


# def print_confusion_matrix(classifier, vectorizer, data, true_classes):
#
#     labels = ["black", "death", "doom", "thrash"]
#     data = vectorizer.transform(data)
#     pred_classes = classifier.predict(data)
#     matrix = confusion_matrix(true_classes, pred_classes, labels=labels)
#
#     dutils.plot_confusion_matrix(matrix, labels, title="Genre confusion matrix")
#
#     print("\n")
#     print("-" * 64)
#     print("Confusion matrix: " )
#     print(matrix)
#     print("-" * 64)
#     print("\n")


def get_classifiers():

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


def teach_classifier_in_batches(classifier, vectorizer, dataset, classes, available_classes):
    """ Go through the training set, vectorize a batch of data, and teach it to the classifier
        Repeat while there is more data to teach"""

    batch_num = 0
    dataset_batches = _get_batches(dataset)
    classes_batches = _get_batches(classes)
    for dataset_batch, classes_batch in zip(dataset_batches, classes_batches):
        batch_num = batch_num + 1
        print("Teaching batch number " + str(batch_num) + " with size " + str(len(dataset_batch)))

        dataset_batch_vect = vectorizer.transform(dataset_batch)
        # dataset_batch_vect = prep.scale(dataset_batch_vect, with_mean=False)
        classifier.partial_fit(dataset_batch_vect.toarray(), classes_batch, available_classes)


def teach_classifier(classifier, vectorizer, dataset, classes ):

    dataset_vect = vectorizer.transform(dataset)
    # dataset_vect = prep.scale(dataset_vect, with_mean=False)
    classifier.fit(dataset_vect, classes)


def test_classifier_in_batches(classifier, vectorizer, dataset, classes):

    print("*\tTesting classifier: " + str(classifier))

    dataset_batches = _get_batches(dataset)
    classes_batches = _get_batches(classes)

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


def test_classifier(classifier, vectorizer, dataset, classes):
    print("*\tTesting classifier: " + str(classifier))
    dataset_vect = vectorizer.transform(dataset)
    # dataset_vect = prep.scale(dataset_vect, with_mean=False)
    score = classifier.score(dataset_vect, classes)
    print("Score: ", score)


def _get_batches(l):

    batch_size = app_data.BATCH_SIZE
    list_size = len(l)

    for i in range(0, list_size, batch_size):
        yield l[i:i + batch_size]


def print_classifier_top_features(vectorizer, clf, class_labels, n_features=10):
    """Prints features with the highest coefficient values, per class"""

    if not hasattr(clf, 'coef_'):
        print("Can't find top features for classifier: ")
        print(clf)
        return

    print("Top features for classifier: ")
    print(clf)

    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top = np.argsort(clf.coef_[i])[-n_features:]
        print("%s: %s" % (class_label, " ".join(feature_names[j] for j in top)))


def _get_advanced_features_transformator_union():

    union = FeatureUnion(transformer_list=[

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
                # ('transformer', TfidfTransformer())
            ])
             ),

            ('lyrics_bow', Pipeline([
                ('selector', LyricsFeatureSelector(key='lyrics')),
                ('vectorizer', CountVectorizer(max_features=10000))
                # ('transformer', TfidfTransformer())
            ])
             )

        ], transformer_weights={
            # 'verse_count'      : 0.5,
            # 'stanza_count'     : 0.5,
            # 'avg_verse_length' : 0.8,
            'pos_tags_map': 0.6,
            'lyrics_bow'  : 0.9
        }

    )

    return union


def _get_basic_features_transformator_union():

    union = FeatureUnion(
        transformer_list=[

            ('features_dict', Pipeline([
                                ('selector', LyricsFeatureSelector(key='features')),
                                ('vectorizer', DictVectorizer())
                              ])
            ),

            ('pos_tags_map', Pipeline([
                                ('selector', LyricsFeatureSelector(key='pos_tags_map')),
                                ('vectorizer', DictVectorizer())
                            ])
            ),

            ('word_endings', Pipeline([
                                ('selector', LyricsFeatureSelector(key='word_endings')),
                                ('vectorizer', DictVectorizer()),
                                ('transformer', TfidfTransformer())
                            ])

            ),

            ('lyrics_bow', Pipeline([
                                ('selector', LyricsFeatureSelector(key='lyrics')),
                                ('vectorizer', TfidfVectorizer(stop_words='english', max_df=0.6, analyzer='word'))
                           ])
            )

        ],
        transformer_weights={
            'features_dict': 0.2,
            'pos_tags_map' : 1.0,
            'word_endings' : 0.7,
            'lyrics_bow'   : 0.4
        }
    )

    return union

