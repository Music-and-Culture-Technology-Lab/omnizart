import os


MODULE_PATH = os.path.abspath(__file__ + "/..")
SETTING_DIR = os.path.join(MODULE_PATH, "defaults")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['VAMP_PATH'] = os.path.join(MODULE_PATH, "resource", "vamp")

__version__ = "0.3.2"
