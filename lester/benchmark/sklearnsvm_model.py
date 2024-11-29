from lester.benchmark import ModelCodeTransformationTask


class SklearnSVMTransformationTask(ModelCodeTransformationTask):
    @property
    def original_code(self):
        return """
from sklearn import svm
model = svm.LinearSVC(penalty='l2', loss='hinge')
        """

    def evaluate_transformed_code(self, transformed_code):
        model_func = self.extract_model_func(transformed_code)

        num_features = 100
        model, loss = model_func(num_features)

        import torch

        parameters = list(model.parameters())
        assert len(parameters) == 2
        assert parameters[0].shape == (1, num_features)
        assert parameters[1].shape == torch.Size([1])

        # Some tests for hinge loss
        assert loss(torch.tensor(1.0), torch.tensor(2.0)) == 0.0
        assert loss(torch.tensor(1.0), torch.tensor(0.5)) == 0.5
        assert loss(torch.tensor(-1.0), torch.tensor(1.5)) == 2.5
        assert loss(torch.tensor(-1.0), torch.tensor(-1.0)) == 0.0
