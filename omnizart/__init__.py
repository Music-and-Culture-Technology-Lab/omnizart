import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


MODULE_PATH = os.path.abspath(__file__ + "/..")
SETTING_DIR = os.path.join(MODULE_PATH, "defaults")

__version__ = "0.1.0"
