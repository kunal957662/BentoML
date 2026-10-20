
import numpy as np
import bentoml

@bentoml.service(name="iris_classifier")
class IrisClassifier:

    def __init__(self):
        self.model = bentoml.sklearn.load_model(
            "iris_clf:zmsvyrwmzsl4xtch"
        )

    @bentoml.api
    def classify(self, input_series: np.ndarray) -> np.ndarray:
        return self.model.predict(input_series)
