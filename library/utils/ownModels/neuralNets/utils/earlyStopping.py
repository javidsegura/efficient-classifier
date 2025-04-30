
from tensorflow.keras.callbacks import EarlyStopping

def get_early_stopping(monitor: str = "val_loss", patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True, verbose: int = 3):
      early_stop = EarlyStopping(
                  monitor=monitor,     
                  patience=patience,  
                  min_delta=min_delta,           
                  restore_best_weights=restore_best_weights,  
                  verbose=verbose
            )
      return early_stop