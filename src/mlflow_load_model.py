from pathlib import Path

from .mlflow_model_loader import MLFlowModelLoader


def _load_pyfunc(data_path: str):
    """
    Return the model for inference.
    It should implement the `predict` method.
    
    :param data_path: path where model artifacts are stored
    :type data_path: str
    :return: model
    """
    return MyModelLoader.load_model(data_path)


class MyModelLoader(MLFlowModelLoader):
    @staticmethod
    def save_model(model, data_path: str):
        pass
    
    @staticmethod
    def load_model(data_path: str):
        return MockModel()
        

class MockModel:   
    def predict(self, data):
        import numpy as np
        import torch

        print(f"Type of data: {type(data)}")
        print(f"data: {data}")
        print(f"data.shape: {data.shape}")
        print(f"data[0, :2, :2, :2]: {data[0, :2, :2, :2]}")
        return np.expand_dims(np.transpose(data, (2, 0, 1)), axis=0)
    