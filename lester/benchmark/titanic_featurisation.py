# https://github.com/josephmisiti/kaggle-titanic/blob/master/Titanic%20Classification.ipynb
from lester.benchmark import FeaturisationCodeTransformationTask


class TitanicFeaturisationTask(FeaturisationCodeTransformationTask):

    @property
    def original_code(self):
        return """
import pandas as pd
import numpy as np

import sklearn
from sklearn import preprocessing

# drop the useless columns that we know are not going to be good for prediction
df.drop(["Name","Ticket","Cabin"], axis=1, inplace=True)

label_encoder = preprocessing.LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

df.drop(['Survived'], axis=1, inplace=True)

X = df.as_matrix().astype(np.float)

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)        
"""

    @property
    def input_schema(self):
        return ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
                'Ticket', 'Fare', 'Cabin', 'Embarked']

    def evaluate_transformed_code(self, transformed_code):
        encoders_by_column = self.extract_encoders_by_column(transformed_code)
        assert len(encoders_by_column.keys()) == 8

        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.pipeline import Pipeline

        for column in ['Sex', 'Embarked']:
            assert column in encoders_by_column
            encoder = encoders_by_column[column][0]
            if isinstance(encoder, Pipeline):
                _, encoder = encoder.steps[-1]
                assert isinstance(encoder, OneHotEncoder)
            else:
                assert isinstance(encoder, OneHotEncoder)

        for column in ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
            assert column in encoders_by_column
            encoder = encoders_by_column[column][0]
            if isinstance(encoder, Pipeline):
                _, encoder = encoder.steps[-1]
                assert isinstance(encoder, StandardScaler)
            else:
                assert isinstance(encoder, StandardScaler)
