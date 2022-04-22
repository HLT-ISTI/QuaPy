from quapy.data.base import LabelledCollection
from quapy.method.base import BaseQuantifier
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfPipeline(BaseQuantifier):
    def __init__(self, quantifier):
        super().__init__()
        self.vectorizer = TfidfVectorizer(min_df=5, sublinear_tf=True, ngram_range=(1, 2))
        self.quantifier = quantifier

    def quantify(self, instances):
        instances = self.vectorizer.transform(instances)
        return self.quantifier.quantify(instances)

    def fit(self, data: LabelledCollection):
        instances = self.vectorizer.fit_transform(data.instances)
        tfidfdata = LabelledCollection(instances, data.labels)
        return self.quantifier.fit(tfidfdata)

    def set_params(self, **parameters):
        return self.quantifier.set_params(**parameters)

    def get_params(self, deep=True):
        return self.quantifier.get_params(deep)

    def classes_(self):
        return self.quantifier.classes_


