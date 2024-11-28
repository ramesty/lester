from abc import ABC, abstractmethod


class DataprepCodeTransformationTask(ABC):

    @property
    @abstractmethod
    def original_code(self):
        pass

    @property
    @abstractmethod
    def input_arg_names(self):
        pass

    @property
    @abstractmethod
    def input_schemas(self):
        pass

    @property
    @abstractmethod
    def output_columns(self):
        pass

    @abstractmethod
    def run_manually_rewritten_code(self, params):
        pass

    @abstractmethod
    def evaluate_transformed_code(self, transformed_code):
        pass


class FeaturisationCodeTransformationTask(ABC):

    @property
    @abstractmethod
    def original_code(self):
        pass

    @property
    @abstractmethod
    def input_schema(self):
        pass

    @staticmethod
    def extract_encoders_by_column(transformed_code):
        variables_for_exec = {}
        exec(transformed_code, variables_for_exec)
        column_transformer = variables_for_exec['__featurise']()

        from sklearn.compose import ColumnTransformer
        assert isinstance(column_transformer, ColumnTransformer)

        encoders_by_column = {}
        for (_, encoder, columns) in column_transformer.transformers:

            if isinstance(columns, str):
                columns = [columns]

            for column in columns:
                if column not in encoders_by_column:
                    encoders_by_column[column] = []
                encoders_by_column[column].append(encoder)

        return encoders_by_column

    @abstractmethod
    def evaluate_transformed_code(self, transformed_code):
        pass


class ModelCodeTransformationTask(ABC):

    @property
    @abstractmethod
    def original_code(self):
        pass

    @abstractmethod
    def evaluate_transformed_code(self, transformed_code):
        pass
