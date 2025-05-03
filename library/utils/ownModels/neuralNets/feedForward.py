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

import kerastuner as kt
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class FeedForwardNeuralNetwork(BaseEstimator, ClassifierMixin):
      def __init__(self,
                  num_features: int,
                  num_classes:   int,
                  batch_size:    int = 128,
                  epochs:        int = 20,
                  n_layers:      int = 1,
                  units_per_layer: list = [128],
                  activations:   list = ['relu'],
                  learning_rate: float = 1e-3
                  ):
            # store all hyper‑parameters
            self.num_features  = num_features
            self.num_classes   = num_classes
            self.batch_size    = batch_size
            self.epochs        = epochs
            self.n_layers      = n_layers
            self.units_per_layer = units_per_layer
            self.activations   = activations
            self.learning_rate = learning_rate

            # placeholder for the trained model
            self.model = None

      def _build_optimizeable_model(self, hp):
            """
            Model‑building function for the tuner.
            Uses `hp` to sample:
            - number of layers
            - units per layer
            - activation
            - learning rate
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
      
      def _build_parametrized_model(self):
            model = Sequential()
            model.add(Input(shape=(self.num_features,)))
            for i in range(self.n_layers):
                  model.add(Dense(self.units_per_layer[i], activation=self.activations[i]))
            model.add(Dense(self.num_classes, activation='softmax'))

            lr = self.learning_rate
            model.compile(
                  optimizer=AdamW(learning_rate=lr, weight_decay=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
            )
            
            return model
      
      def tuner_search(self,
                       X_train,
                       y_train,
                       X_val,
                       y_val):
            
            self.tuner.search(
                        X_train, 
                        y_train,
                        validation_data=(X_val, y_val),
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        callbacks=[get_early_stopping()])

      def get_tuned_model(self,
                  max_trials:  int = 20,
                  executions_per_trial: int = 1,
                  directory:   str = 'kt_tuning',
                  project_name: str = 'ffnn'):
            """
            Run Bayesian hyperparameter search.
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

      def fit(self, X, y, **kwargs):
            self.model = self._build_parametrized_model()
            fit_args = dict(
                  x=X, y=y,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  callbacks=[get_early_stopping()]
            )
            if "X_val" in kwargs and "y_val" in kwargs:
                  fit_args["validation_data"] = (kwargs["X_val"], kwargs["y_val"])
            self.history = self.model.fit(**fit_args)
            self.is_fitted_ = True
            return self

      def predict(self, X):
            preds = self.model.predict(X)
            return np.argmax(preds, axis=1)

      def get_params(self, deep=True):
            return {
                  'num_features':  self.num_features,
                  'num_classes':   self.num_classes,
                  'activations':   self.activations,
                  'learning_rate': self.learning_rate,
                  'batch_size':    self.batch_size,
                  'epochs':        self.epochs
            }
