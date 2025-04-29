import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

class FeedForwardNeuralNetwork():
      """
      Defines a standard non-optimized architecture 

      Parameters
      ----------
      input_shape : tuple
            The shape of the input data
      """
      def __init__(self, X_shape: tuple, y_shape: tuple) -> None:
            self.X_shape = X_shape
            self.y_shape = y_shape
            self.model = self._build_model()

      def _build_model(self):
            model = Sequential([
                  tf.keras.Input(shape=(self.X_shape[1],)),  # Explicit Input layer
                  Dense(32, activation='relu', kernel_initializer='glorot_uniform'),
                  Dense(32, activation='relu', kernel_initializer='glorot_uniform'),
                  Dense(self.y_shape[0], activation='softmax', kernel_initializer='glorot_uniform')  # Softmax for multiclass classification
            ])
            model.compile(
                  optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']  # Only built-in metrics unless you define custom ones
            )
            assert model is not None, "Model is not built"
            return model
      
      def fit(self, X_data, y_data, X_val, y_val):
            history = self.model.fit(X_data, 
                           y_data, 
                           epochs=2,
                           batch_size=32, 
                           validation_data=(X_val, y_val))
            self.history = history
            return self, history
      
      def predict(self, X_data):
            self.soft_predictions = self.model.predict(X_data)
            self.hard_predictions = np.argmax(self.soft_predictions, axis=1)
            return self.hard_predictions
      
      def get_params(self):
            return {}
      
