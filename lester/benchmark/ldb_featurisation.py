# https://github.com/LittleDevilBig/Systems-for-AI-Quality/blob/main/main.py
from lester.benchmark import FeaturisationCodeTransformationTask


class LdbFeaturisationTask(FeaturisationCodeTransformationTask):

    @property
    def original_code(self):
        return """
def preprocess(data_total, lower_dimension=100):
    # build label set
    label = data_total['product_category'] == 'Jewelry'

    # build vocabulary
    vocab = [re.sub('[^A-Za-z]+', ' ', str(title)).strip().lower()
             + re.sub('[^A-Za-z]+', ' ', str(comment)).strip().lower()
             for title, comment in zip(data_total['product_title'], data_total['product_review'])]
    vec = TfidfVectorizer()
    feature = vec.fit_transform(vocab)
    print('feature shape: {}'.format(feature.shape))
    # reduce feature dimension from vocab size to lower_dimension using SVD
    # this technique also made the dimension of training data and test data consistent
    svd = TruncatedSVD(n_components=lower_dimension, n_iter=7, random_state=42)
    feature = svd.fit_transform(feature)
    print('feature shape after dimension reduction: {}'.format(feature.shape))
    rating = data_total['product_rating']
    # add rating to the first column of feature
    feature = np.insert(feature, 0, rating, axis=1)
    print('feature shape after adding rating: {}'.format(feature.shape))
    return feature, label
"""

    @property
    def input_schema(self):
        return ['product_id', 'product_category', 'product_name', 'rating', 'review']

    def evaluate_transformed_code(self, transformed_code):
        encoders_by_column = self.extract_encoders_by_column(transformed_code)
        assert len(encoders_by_column.keys()) == 3

        assert 'product_name' in encoders_by_column
        assert 'review' in encoders_by_column
        assert 'rating' in encoders_by_column

        _, rating_encoder = encoders_by_column['rating'][0].steps[-1]
        import pandas as pd
        ratings = pd.Series([[1], [2], [3], [4], [5]])
        transformed_ratings = rating_encoder.transform(ratings).squeeze()
        assert (transformed_ratings == ratings).all()

