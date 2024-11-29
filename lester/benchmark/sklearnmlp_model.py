from lester.benchmark import ModelCodeTransformationTask


class SklearnMLPTransformationTask(ModelCodeTransformationTask):
    @property
    def original_code(self):
        return """
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=[64, 16], activation='relu')
        """

    def evaluate_transformed_code(self, transformed_code):
        model_func = self.extract_model_func(transformed_code)

        num_features = 100
        model, loss = model_func(num_features)

        import torch
        from torch.nn import BCELoss
        assert isinstance(loss, BCELoss)

        parameters = list(model.parameters())
        assert len(parameters) == 6
        assert parameters[0].shape == (64, num_features)
        assert parameters[1].shape == torch.Size([64])
        assert parameters[2].shape == (16, 64)
        assert parameters[3].shape == torch.Size([16])
        assert parameters[4].shape == (1, 16)
        assert parameters[5].shape == torch.Size([1])