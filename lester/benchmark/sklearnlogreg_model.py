import torch.nn

from lester.benchmark import ModelCodeTransformationTask


class SklearnLogisticRegressionTransformationTask(ModelCodeTransformationTask):
    @property
    def original_code(self):
        return """
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss="log_loss", penalty=None)        
        """

    def evaluate_transformed_code(self, transformed_code):

        model_func = self.extract_model_func(transformed_code)

        num_features = 100
        model, loss = model_func(num_features)

        import torch
        from torch.nn import BCELoss
        assert isinstance(loss, BCELoss)

        parameters = list(model.parameters())
        assert len(parameters) == 2
        assert parameters[0].shape == (1, num_features)
        assert parameters[1].shape == torch.Size([1])
