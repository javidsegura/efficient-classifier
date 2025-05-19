class LockedKeysDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._locked_keys = set(self.keys())

    def __setitem__(self, key, value):
        if key not in self._locked_keys:
            raise KeyError(f"Key '{key}' not allowed. Only these keys are permitted: {self._locked_keys}")
        super().__setitem__(key, value)
    
