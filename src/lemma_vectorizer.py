from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

class LemmaVectorizer(CountVectorizer):

    def __init__(self):
        super(LemmaVectorizer, self).__init__(stop_words='english', max_df=0.4)
        self.lemmatizer = WordNetLemmatizer()


    def build_analyzer(self):
        analyzer = super(LemmaVectorizer, self).build_analyzer()
        return lambda doc: [self.lemmatizer.lemmatize(t) for t in analyzer(doc)]


    def __remove_newline_start(self, token):
        return token[2:] if token[:2] == "\\n" else token

