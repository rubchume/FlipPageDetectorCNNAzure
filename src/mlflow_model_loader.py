from abc import ABC, abstractmethod
import inspect
import logging
from pathlib import Path
from typing import Any

import mlflow


class MLFlowModelLoader(ABC):
    def __init__(self, model_artifact_path="model_weights.pth"):
        self.model_artifact_path = model_artifact_path
        logging.basicConfig(level=logging.INFO)
    
    @staticmethod
    @abstractmethod
    def save_model(model: Any, data_path: str):
        """
        Saves the model as artifacts.
        
        It is advisable in cloud scenarios to import the necessary packages inside this function instead of doing it
        at module level.
        This is to save resources in case the module is loaded multiple times (as you don't have direct control over how the 
        cloud platform works)

        :param model: model object that will be saved to artifacts in `data_path`
        :type model: Any
        :param data_path: path where model artifacts will be stored
        :type data_path: str
        """
        
    @staticmethod
    @abstractmethod
    def load_model(data_path: str):
        """
        Return the model for inference.
        It should implement the `predict` method.
        
        It is advisable in cloud scenarios to import the necessary packages inside this function instead of doing it
        at module level.
        This is to save resources in case the module is loaded multiple times (as you don't have direct control over how the 
        cloud platform works)

        :param data_path: path where model artifacts are stored. Artifacts are saved in :meth:`save_model` with the same path as input
        :type data_path: str
        :return: model
        :rtype: Any
        """
        
    def log_model(
        self,
        model: Any,
        model_name: str,
        signature: mlflow.models.signature.ModelSignature,
        **kwargs
    ):
        """
        Logs the model into MLFlow
        
        Adds the folder containing this file as the code path.
        All code should be in this folder
        
        Adds the file where the current class (the one that inherits from this one) is defined as loader module.
        This means that it should implement the method _load_pyfunc. That method should call :meth:`load_model`.
        """
        logging.info("Log model")
        
        self.save_model(model, self.model_artifact_path)

        return mlflow.pyfunc.log_model(
            model_name,
            data_path=self.model_artifact_path,
            code_path=[Path(__file__).parent],
            loader_module=type(self).__module__,
            signature=signature,
            **kwargs
        )