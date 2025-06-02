from efficient_classifier.utils.ownModels.neuralNets.utils.earlyStopping import get_early_stopping


import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.layers import Dropout

from sklearn.base import BaseEstimator, ClassifierMixin

import kerastuner as kt
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

import yaml

class FeedForwardNeuralNetwork(BaseEstimator, ClassifierMixin):
      def __init__(self,
                  num_features: int, # Variables in the model
                  num_classes:   int, # Classes to predict
                  batch_size:    int = None,
                  epochs:        int = None,
                  n_layers:      int = None,
                  units_per_layer: list = None,
                  activations:   list = None,
                  learning_rate: float = None,
                  kernel_initializer: str = None,
                  class_weights: dict = None
                  ):
            """

            Notes:
                  This class is keras model with a scikit-learn like interface.
                  Activations and units per layers are arrays where each correspods to a layer.

                  This model class also contains part of the optimizer for the model itself. This is different to scikit-learn native models.
            """
            nn_config = {
                  "batch_size": 128,
                  "epochs": 10,
                  "n_layers": 4,
                  "units_per_layer": [512, 256, 128, 64],
                  "activations": ['relu', 'relu', 'relu', 'relu'],
                  "learning_rate": 0.001,
                  "kernel_initializer": 'glorot_uniform',
            }

            self.num_features = num_features
            self.num_classes = num_classes
            self.batch_size = batch_size if batch_size is not None else nn_config["batch_size"]
            self.epochs = epochs if epochs is not None else nn_config["epochs"]
            self.n_layers = n_layers if n_layers is not None else nn_config["n_layers"]
            self.units_per_layer = units_per_layer if units_per_layer is not None else nn_config["units_per_layer"]
            self.activations = activations if activations is not None else nn_config["activations"]
            self.learning_rate = learning_rate if learning_rate is not None else nn_config["learning_rate"]
            self.kernel_initializer = kernel_initializer if kernel_initializer is not None else nn_config["kernel_initializer"]
            self.class_weights = class_weights

            # Validate the parameters if they are provided
            if units_per_layer is not None and n_layers is not None:
                  assert len(units_per_layer) == n_layers, f"Number of units per layer must be equal to the number of layers. Units per layer: {units_per_layer}, Number of layers: {n_layers}"
            if activations is not None and n_layers is not None:
                  assert len(activations) == n_layers, f"Number of activations must be equal to the number of layers. Activations: {activations}, Number of layers: {n_layers}"

            self.model = None

      def _build_parametrized_model(self):
            """
            Purpose: compiles the model for when non-optimizing the model.
            """
            model = Sequential()
            model.add(Input(shape=(self.num_features,)))
            for i in range(self.n_layers):
                  model.add(Dense(self.units_per_layer[i], activation=self.activations[i], kernel_initializer=self.kernel_initializer))
            model.add(Dense(self.num_classes, activation='softmax', kernel_initializer=self.kernel_initializer))

            lr = self.learning_rate
            model.compile(
                  optimizer=AdamW(learning_rate=lr, weight_decay=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
            )
            
            return model
      
      def _build_optimizeable_model(self, hp):
            """
            Purpose:
               - used by the keras tuner to find the best hyperparameters
            Notes:
              - Hyperparmeters values to tune are hardcoded here
              - If you want to tune more parameters, please adhere to the current structure (adding as initilizer parameters)
            """
            model = Sequential()
            model.add(Input(shape=(self.num_features,)))

            # Tune the number of layers: between 1 and 5
            n_layers = hp.Int('n_layers', 1, 5, default=self.n_layers)
            for i in range(n_layers):
                  # Tune units per layer
                  units = hp.Choice(f'units_{i}', [32, 64, 128, 256, 512],
                                    default=self.units_per_layer[i]
                                    if i < len(self.units_per_layer) else 128)
                  # Tune activation per layer
                  act   = hp.Choice(f'act_{i}', ['relu', 'tanh', 'selu'],
                                    default=self.activations[i]
                                    if i < len(self.activations) else 'relu')
                  model.add(Dense(units, activation=act))

            model.add(Dense(self.num_classes, activation='softmax'))

            # Tune learning rate
            lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log',
                        default=self.learning_rate)

            model.compile(
                  optimizer=AdamW(learning_rate=lr, weight_decay=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
            )
            return model

      def get_tuned_model(self,
                  max_trials:  int,
                  executions_per_trial: int = 1,
                  directory:   str = 'kt_tuning',
                  project_name: str = 'ffnn'):
            """
            Purpose: compiles the otpimizer and  returns it
            """
            self.tuner = kt.BayesianOptimization(
                  hypermodel=self._build_optimizeable_model,
                  objective='val_accuracy',
                  max_trials=max_trials,
                  executions_per_trial=executions_per_trial,
                  directory=directory,
                  project_name=project_name,
                  overwrite=True
            )

            return self.tuner
      
      def tuner_search(self,
                       X_train,
                       y_train,
                       X_val,
                       y_val):
            """
            Purpose: activate the tuner search
            """
            self.tuner.search(
                        X_train, 
                        y_train,
                        validation_data=(X_val, y_val),
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        callbacks=[get_early_stopping()],
                        class_weight=self.class_weights
                  )

      def fit(self, X, y, **kwargs):
            self.model = self._build_parametrized_model()
            fit_args = dict(
                  x=X, y=y,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  callbacks=[get_early_stopping()] #  We use early to stop execution when it become ineffective
            )
            
            if self.class_weights is not None:
                 fit_args["class_weight"] = self.class_weights
            
            
            if self.class_weights is not None:
                 fit_args["class_weight"] = self.class_weights
            
            if "X_val" in kwargs and "y_val" in kwargs:
                  fit_args["validation_data"] = (kwargs["X_val"], kwargs["y_val"])
            self.history = self.model.fit(**fit_args)
            self.is_fitted_ = True # Needed for sklearn compatibility
            self.classes_ = np.unique(y)  # Add this line to store unique class labels
            
            return self

      def predict(self, X):
            preds = self.model.predict(X) # Softmax originally returns soft-probabilities. We then take the class with the highest probability of being right 
            return np.argmax(preds, axis=1)
      
      def predict_proba(self, X):
            return self.model.predict(X)  # Simply return the model's predictions directly

      def get_params(self, deep=True):
            return {
                  'num_features':  self.num_features,
                  'num_classes':   self.num_classes,
                  'batch_size':    self.batch_size,
                  'epochs':        self.epochs,
                  'n_layers':      self.n_layers,
                  'units_per_layer': self.units_per_layer,
                  'activations':   self.activations,
                  'learning_rate': self.learning_rate,
                  'kernel_initializer': self.kernel_initializer,
                  'class_weights': self.class_weights
            }