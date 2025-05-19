


from library.utils.decorators.timer import timer

import joblib
import pickle

from abc import ABC, abstractmethod
            

class Deserialization:
      """ This class is responsible for deserializing the pipelines and models. """
      def __init__(self):
            pass
      
      @timer("Deserializing pipelines")
      def deserialize_pipelines(self, pipelines_to_deserialize: dict[str, str]):
            deserialized_pipelines = {}
            for pipeline_name, pipeline_path in pipelines_to_deserialize.items():
                  deserialized_pipelines[pipeline_name] = self._deserialize_pipeline(pipeline_name, pipeline_path)
            return deserialized_pipelines

      @abstractmethod
      def _deserialize_pipeline(self, pipeline_name: str, pipeline_path: str):
            pass
      
      @timer("Deserializing models")
      def deserialize_models(self, models_to_deserialize: dict[str, str]):
            deserialized_models = {}
            for model_name, model_path in models_to_deserialize.items():
                  deserialized_models[model_name] = self._deserialize_model(model_name, model_path)
            return deserialized_models

      @abstractmethod
      def _deserialize_model(self, model_name: str, model_path: str):
            pass


class DeserializationJoblib(Deserialization):
      def __init__(self):
            super().__init__()

      def _deserialize_pipeline(self, pipeline_name: str, pipeline_path: str):
            print(f"Deserializing pipeline {pipeline_name} from {pipeline_path}")
            obj = joblib.load(pipeline_path)
            return obj
      
      def _deserialize_model(self, model_name: str, model_path: str):
            print(f"Deserializing model {model_name} from {model_path}")
            obj = joblib.load(model_path)
            return obj
      

class DeserializationPickle(Deserialization):
      def __init__(self):
            super().__init__()
      
      def _deserialize_pipeline(self, pipeline_name: str, pipeline_path: str):
            print(f"Deserializing pipeline {pipeline_name} from {pipeline_path}")
            obj = pickle.load(open(pipeline_path, "rb"))
            return obj
      
      def _deserialize_model(self, model_name: str, model_path: str):
            print(f"Deserializing model {model_name} from {model_path}")
            obj = pickle.load(open(model_path, "rb"))
            return obj
      
