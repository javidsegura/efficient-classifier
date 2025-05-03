from library.utils.ownModels.neuralNets.utils.earlyStopping import get_early_stopping


import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.layers import Dropout

from sklearn.base import BaseEstimator, ClassifierMixin


import numpy as np
class FeedForwardNeuralNetwork(BaseEstimator, ClassifierMixin):
      """
      Defines a standard non-optimized architecture 

      Parameters
      ----------
      input_shape : tuple
            The shape of the input data
      """
      def __init__(self, num_features: int, num_classes: int, model_keras: object = None) -> None:
            self.num_features = num_features
            self.num_classes = num_classes
            if model_keras is None:
                  self.model = self._compile_model()
            else:
                  self.model = model_keras

      def _compile_model(self):
            model = Sequential([
                  tf.keras.Input(shape=(self.num_features,)),  
                  Dense(128, activation='relu', kernel_initializer='glorot_uniform'),
                  Dense(self.num_classes, activation='softmax', kernel_initializer='glorot_uniform') 
            ])
            model.compile(
                  optimizer=AdamW(learning_rate=0.001, weight_decay=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']  # Only built-in metrics unless you define custom ones
            )
            assert model is not None, "Model is not built"
            return model
      
      def _get_compiled_model_optimized(self, num_features, num_classes):
            def _compiled_model_optimized(hp):
                  model = Sequential()
                  model.add(Input(shape=(num_features, )))

                  for i in range(hp.Int('num_layers', 1, 5)):
                        neurons = hp.Choice(f'units_{i}', [32, 64, 128, 256, 512])
                        model.add(Dense(neurons, activation='relu'))

                  # Output
                  model.add(Dense(num_classes, activation='softmax'))

                  # 3) Learning rate: log-uniform from 1e-4 to 1e-2
                  lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
                  model.compile(
                        optimizer=AdamW(learning_rate=lr, weight_decay=1e-4),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                  )
                  return model
            return _compiled_model_optimized
      
      def fit(self, X_data, y_data, **kwargs):
            X_val = kwargs.get("X_val", None)
            y_val = kwargs.get("y_val", None)
            if X_val is not None and y_val is not None:
                  print(f"Fitting model with validation data")
                  self.history = self.model.fit(
                              X_data, 
                              y_data, 
                              epochs=5,
                              batch_size=128, 
                              validation_data=(X_val, y_val),
                              callbacks=[get_early_stopping()])
            else:
                  print(f"Fitting model without validation data")
                  self.history = self.model.fit(
                              X_data, 
                              y_data, 
                              epochs=5,
                              batch_size=128, 
                              callbacks=[get_early_stopping()])
            
            self.is_fitted_ = True
            return self
      
      def predict(self, X_data):
            self.soft_predictions = self.model.predict(X_data)
            self.hard_predictions = np.argmax(self.soft_predictions, axis=1)
            return self.hard_predictions
      
      def get_params(self, deep=True):
            return {
                  "num_layers": len(self.model.layers),
                  "num_neurons": [layer.units for layer in self.model.layers if isinstance(layer, Dense)],
                  "activations": [layer.activation.__name__ for layer in self.model.layers if isinstance(layer, Dense)],
                  "optimizer": type(self.model.optimizer).__name__,
                  "loss": self.model.loss,
                  "metrics": [m.name if hasattr(m, 'name') else m for m in self.model.metrics]
            }
      