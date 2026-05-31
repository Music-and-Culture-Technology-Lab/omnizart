import os
import collections
import collections.abc

# Patch collections module for backward compatibility with old libraries under Python 3.10+
for name in ['MutableSequence', 'Iterable', 'Mapping', 'Sequence', 'Callable', 'Container', 'MutableMapping']:
    if not hasattr(collections, name) and hasattr(collections.abc, name):
        setattr(collections, name, getattr(collections.abc, name))

# Patch numpy for backward compatibility with old libraries under NumPy 2.0+
import numpy as np
for name, target in [('float', float), ('int', int), ('bool', bool), ('complex', complex)]:
    if not hasattr(np, name):
        setattr(np, name, target)

MODULE_PATH = os.path.abspath(f"{__file__}/..")
SETTING_DIR = os.path.join(MODULE_PATH, "defaults")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['VAMP_PATH'] = os.path.join(MODULE_PATH, "resource", "vamp")

__version__ = "0.6.2"
