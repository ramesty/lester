from lester.benchmark import FeaturisationCodeTransformationTask


class CreditcardFeaturisationTask(FeaturisationCodeTransformationTask):

    @property
    def original_code(self):
        return '''
import numpy as np
from sentence_transformers import SentenceTransformer

sentence_embedder = SentenceTransformer("all-mpnet-base-v2")
country_indices = {'DE': 0, 'FR': 1, 'UK': 2}

titles = []
texts = []
countries = []
        
with open(".scratchspace/__intermediate.csv") as file:
    for line in file:
        parts = line.strip().split("\t")
        title, text, bank, country, sentiment, is_premium = parts

        titles.append(title)
        texts.append(text)
        countries.append(country)

subject_embeddings = sentence_embedder.encode(titles)
text_embeddings = sentence_embedder.encode(texts)
title_lengths = [len(title.split(" ") for titles in titles]
title_lengths = (title_lengths - np.mean(title_lengths)) / np.std(title_lengths)

country_onehot = np.zeros((len(countries), len(country_indices)))
for row, country in enumerate(countries):
    country_onehot[row, country_indices[country]] = 1.0


X = np.concatenate((
    subject_embeddings,
    text_embeddings,
    title_lengths.reshape(-1,1),
    country_onehot
), axis=1)        
'''

    @property
    def input_schema(self):
        return ['title', 'text', 'bank', 'country', 'sentiment', 'is_premium']

    def evaluate_transformed_code(self, transformed_code):

        encoders_by_column = self.extract_encoders_by_column(transformed_code)

        from sklearn.preprocessing import OneHotEncoder

        assert len(encoders_by_column.keys()) == 3
        assert 'title' in encoders_by_column
        assert 'text' in encoders_by_column
        assert 'country' in encoders_by_column

        assert isinstance(encoders_by_column['country'][0], OneHotEncoder)

        encoded = encoders_by_column['text'][0].transform(['a', 'b', 'c'])
        assert len(encoded) == 3
        assert encoded[0].shape[0] == 768

        assert len(encoders_by_column['title']) == 2
