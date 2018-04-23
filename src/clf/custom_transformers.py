from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import app_data
import nltk
import operator
import sys
import itertools


class LyricsVerseCountVectorizer(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self


    def transform(self, lyrics_array):

        vector = []
        verse_break  = app_data.LYRICS_VERSE_BREAK

        for lyrics in lyrics_array:
            verse_count = lyrics.count(verse_break) + 1
            vector.append(float(verse_count))

        return np.asmatrix(vector).T


class LyricsStanzaCountVectorizer(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self


    def transform(self, lyrics_array):

        vector = []
        stanza_break = app_data.LYRICS_STANZA_BREAK

        for lyrics in lyrics_array:
            stanza_count = lyrics.count(stanza_break) + 1
            vector.append(float(stanza_count))

        return np.asmatrix(vector).T


class LyricsAvgVerseLengthVectorizer(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, lyrics_array):

        vector = []
        verse_break = app_data.LYRICS_VERSE_BREAK

        for lyrics in lyrics_array:

            verses              = lyrics.split(verse_break)
            number_of_verses    = len(verses)
            total_verses_length = 0

            for verse in verses:
                verse_length = len(verse)
                total_verses_length += verse_length

            average_verse_length = total_verses_length / number_of_verses

            vector.append(float(average_verse_length))

        return np.asmatrix(vector).T


class LyricsAvgVerseWordCountVectorizer(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self


    def transform(self, lyrics_array):

        vector = []
        verse_break = app_data.LYRICS_VERSE_BREAK

        for lyrics in lyrics_array:

            verses = lyrics.split(verse_break)
            number_of_verses = len(verses)
            total_words_count = 0

            for verse in verses:
                verse_words = verse.split(" ")
                verse_words_count = len(verse_words)
                total_words_count += verse_words_count

            average_verse_word_count = total_words_count // number_of_verses

            vector.append(float(average_verse_word_count))

        return np.asmatrix(vector).T


class LyricsWordCountVectorizer(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self


    def transform(self, lyrics_array):

        vector = []
        verse_break = app_data.LYRICS_VERSE_BREAK

        for lyrics in lyrics_array:

            verses = lyrics.split(verse_break)
            total_words_count = 0

            for verse in verses:
                verse_words = verse.split(" ")
                verse_words_count = len(verse_words)
                total_words_count += verse_words_count

            vector.append(float(total_words_count))

        return np.asmatrix(vector).T


class LyricsPartOfSpeechVectorizer(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self


    def transform(self, lyrics_array):

        vector = []
        pos_tags = app_data.POS_TAGS

        for lyrics in lyrics_array:

            # Get words from lyrics and POS tags from them
            words = nltk.word_tokenize(lyrics)
            pos_arr = nltk.pos_tag(words)

            # Create dictionary with POS tags as keys
            pos_tags_map = {}
            for tag in pos_tags:
                pos_tags_map[tag] = 0

            # Count occurences of POS tags in lyrics
            for pos in pos_arr:
                tag = pos[1]

                if pos_tags_map.get(tag) is None:
                    continue

                pos_tags_map[tag] += 1

            # Get maximum and minimum frequencies of POS tags
            max_freq = max(pos_tags_map.items(), key=operator.itemgetter(1))[1]
            min_freq = min(pos_tags_map.items(), key=operator.itemgetter(1))[1]

            # Normalize POS tags counts
            for tag, count in pos_tags_map.items():
                normalized = (count - min_freq) / (max_freq - min_freq)
                pos_tags_map[tag] = float(normalized)

            vector.append(pos_tags_map)

        return vector


class LyricsWordEndingsVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, top_endings_count=400):
        self._word_endings = []
        self._top_endings_count = top_endings_count


    def fit(self, lyrics_array, genres=None):
        """
        Fit vectorizer to given lyrics array, and optionally their genres. <br/>
        The vectorizer learns <i>top_endings_count</i> most important endings and uses them when
        transforming future lyrics. If genres are given, ending importance calculation is better.

        :param lyrics_array: Lyrics from which to learn word endings
        :param genres: Genres of the corresponding lyrics (used for better ending importance calculation)
        """

        # Get ending dictionaries for all lyrics
        all_lyrics_ending_dicts = self.__get_ending_dicts(lyrics_array)

        # Determine whether to fit using genres or without them
        if genres is None:
            self.__fit_without_genres(all_lyrics_ending_dicts)
        else:
            self.__fit_with_genres(all_lyrics_ending_dicts, genres)

        return self


    def __fit_with_genres(self, ending_dicts, genres):

        """
        Fit vectorizer to given list of endings dictionaries, using their genres to calculate
        the importance of each ending.

        :param ending_dicts: List of word ending dictionaries with their counts
        :param genres: List of genres corresponding to the given word ending dictionaries
        """

        all_endings_dict = {}

        # Get per-genre occurence counts for all endings
        for endings_dict, genre in zip(ending_dicts, genres):
            for ending, count in endings_dict.items():

                # If this ending appeared for the first time, instantiate its genre dictionary
                if all_endings_dict.get(ending) is None:
                    all_endings_dict[ending] = {}

                # Increase the count for this ending in this genre
                all_endings_dict[ending][genre] = all_endings_dict[ending].get(genre, 0) + count

        # Keep only most important endings
        # Importance of an ending is calculated as its power to differentiate between genres
        # So if an ending appears often in one genre, and rarely in another, it has a high importance
        endings_with_importance = {}
        for ending, genre_counts in all_endings_dict.items():

            max_importance = -sys.maxsize
            counts = genre_counts.values()

            # Go through all pairs of genres and compare occurence counts in them for this ending
            for pair in itertools.combinations(counts, 2):

                if pair[0] < pair[1]:
                    smaller = pair[0]
                    bigger  = pair[1]
                else:
                    smaller = pair[1]
                    bigger  = pair[0]

                # Get difference between occurences in current two genres
                diff = bigger - smaller

                # Substract the smaller occurence count because:
                # If two endings have pairs of genres which have the same occurence difference (e.g. 300 & 400, 0 & 100)
                # then the pair which has the smaller number of occurences is more important (in given example, 0 & 100)
                importance = diff - smaller

                if importance > max_importance:
                    max_importance = importance

            endings_with_importance[ending] = float(max_importance)

        # Sort endings by their importances and create an ending-importance dictionary
        endings_sorted_tuples = sorted(endings_with_importance.items(), key=operator.itemgetter(1), reverse=True)
        max_importance_endings_tuples = endings_sorted_tuples[:self._top_endings_count]
        max_importance_endings = [ending_with_importance[0] for ending_with_importance in max_importance_endings_tuples]

        # Free memory
        del endings_with_importance
        del endings_sorted_tuples

        # Remember the top endings
        self._word_endings = max_importance_endings


    def __fit_without_genres(self, ending_dicts):

        """
        Fit vectorizer to given listo of ending dictionaries, calculating their importance
        based solely on their frequency

        :param ending_dicts: List of word ending dictionaries with their counts
        """

        all_endings_dict = {}
        for ending_dict in ending_dicts:
            for ending, count in ending_dict.items():
                all_endings_dict[ending] = all_endings_dict.get(ending, 0) + count

        endings_sorted_tuples = sorted(all_endings_dict.items(), key=operator.itemgetter(1), reverse=True)

        max_importance_endings_tuples = endings_sorted_tuples[:self._top_endings_count]
        max_importance_endings = [ending_with_importance[0] for ending_with_importance in max_importance_endings_tuples]

        # Free memory
        del all_endings_dict
        del endings_sorted_tuples
        del max_importance_endings

        # Remember the top endings
        self._word_endings = max_importance_endings


    def __get_ending_dicts(self, lyrics_array):

        """
        Get dictionaries from lyrics with word endings as keys and their counts as values

        :param lyrics_array: Lyrics for which to generate ending-frequency dictionaries
        :return: Array of ending-frequency dictionaries for the given lyrics
        """

        all_lyrics_ending_dicts = []

        # Get word endings dictionaries from all lyrics
        for lyrics in lyrics_array:
            words = lyrics.split(" ")
            endings_dict = {}

            # Remember 3-character endings from words longer than 3 characters
            # Ignore words shorter than 3 characters
            for word in words:
                ending = word[-3] if len(word) > 3 else None

                if ending is None:
                    continue
                else:
                    endings_dict[ending] = endings_dict.get(ending, 0) + 1

            all_lyrics_ending_dicts.append(endings_dict)

        return all_lyrics_ending_dicts


    def transform(self, lyrics_array):

        vector = []

        # Get endings which to count in all lyrics
        learned_endings = self._word_endings
        lyrics_endings_dicts = self.__get_ending_dicts(lyrics_array)

        for lyrics_endings_dict in lyrics_endings_dicts:

            # Initialize an empty dictionary for current lyrics containing endings as keys
            top_endings_dict = {ending : 0 for ending in learned_endings}

            for ending, count in lyrics_endings_dict.items():

                # Count only endings which the vectorizer has learned (when fit was called)
                if top_endings_dict.get(ending) is None:
                    continue
                else:
                    top_endings_dict[ending] = count

            vector.append(top_endings_dict)

        return vector


#############################################################################
#                              OLG LOGIC                                    #
#############################################################################
class AdvancedLyricsFeaturesExtractor(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, lyrics_array):

        features = np.recarray(shape=(len(lyrics_array), ),
                               dtype=[
                                        ('verse_count', int), ('stanza_count', int),
                                        ('avg_verse_length', float), ('pos_tags_map', dict),
                                        ('lyrics', list)
                                     ]
                               )

        for index, lyrics in enumerate(lyrics_array):
            lyrics_features = lyrics["features"]
            lyrics_content  = lyrics["lyrics"]

            features["verse_count"][index]      = lyrics_features["verse_count"]
            features["stanza_count"][index]     = lyrics_features["stanza_count"]
            features["avg_verse_length"][index] = lyrics_features["avg_verse_length"]
            features["pos_tags_map"][index]     = lyrics_features["pos_tags_map"]
            features["lyrics"][index]           = lyrics_content

        return features


class BasicLyricsFeaturesExtractor(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, lyrics_array):

        features = np.recarray(shape=(len(lyrics_array), ),
                               dtype=[('features', dict), ('pos_tags_map', dict), ('word_endings', dict), ('lyrics', list)]
                               )

        for index, lyrics in enumerate(lyrics_array):
            lyrics_features = lyrics["features"]
            lyrics_content  = lyrics["lyrics"]

            features_dict = {
                "verse_count"      : lyrics_features["verse_count"],
                "stanza_count"     : lyrics_features["stanza_count"],
                "avg_verse_length" : lyrics_features["avg_verse_length"],
                "avg_verse_words"  : lyrics_features["avg_verse_words"],
                "word_count"       : lyrics_features["word_count"]
            }

            pos_tags_map = lyrics_features["pos_tags_map"]
            word_endings = lyrics_features["endings"]

            features["features"][index]     = features_dict
            features["pos_tags_map"][index] = pos_tags_map
            features["lyrics"][index]       = lyrics_content
            features["word_endings"][index] = word_endings

        return features


class LyricsFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]