from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LyricsFeaturesExtractor(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, lyrics_array):

        features = np.recarray(shape=(len(lyrics_array),),
                               dtype=[
                                        ('verse_count', int), ('stanza_count', int),
                                        ('avg_verse_length', float), ('pos_tags_map', dict),
                                        ('lyrics', list)
                                     ]
                               )

        for index, lyrics in enumerate(lyrics_array):
            lyrics_features = lyrics["features"]
            lyrics_content = lyrics["lyrics"]

            features["verse_count"][index]      = lyrics_features["verse_count"]
            features["stanza_count"][index]     = lyrics_features["stanza_count"]
            features["avg_verse_length"][index] = lyrics_features["avg_verse_length"]
            features["pos_tags_map"][index]     = lyrics_features["pos_tags_map"]
            features["lyrics"][index]           = lyrics_content

        return features