from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

class LemmaVectorizer(CountVectorizer):

    def __init__(self):
        super(LemmaVectorizer, self).__init__()
        self.lemmatizer = WordNetLemmatizer()


    def build_analyzer(self):
        analyzer = super(LemmaVectorizer, self).build_analyzer()
        return lambda doc: [self.lemmatizer.lemmatize(t) for t in analyzer(doc)]



