
from mmaction.datasets.builder import PIPELINES


class SecureDict(dict):
    def __init__(self, *args, **kwargs):
        self.unauthorized_keys = kwargs.pop("unauthorized_keys", [])
        super().__init__(*args, **kwargs)
    def __setitem__(self, key,val) -> None:
        if key in self.unauthorized_keys and super().__getitem__(key) != val:
            raise KeyError(f"{key} in unauthorized keys, so it cannot be changed")
        super().__setitem__(key,val)


@PIPELINES.register_module()
class EnsureFixedKeys:
    """
    Ensure that going forward in the pipeline, 
    the provided keys don't change, if they do raise KeyError.
    EnsureFixedKey effects stop after calling Collect(),
    or other pipeline operation that create a new dict. 
    """
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        return SecureDict(results,unauthorized_keys=self.keys)

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'
