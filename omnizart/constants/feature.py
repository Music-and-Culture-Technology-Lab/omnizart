"""Some constant feature settings."""

## --------------- For Music module --------------- ##
# Mapping feature name to channel number
FEATURE_NAME_TO_NUMBER = {"S": 1, "G": 2, "C": 3, "Spec": 1, "GCoS": 2, "Ceps": 3}

## --------------- For Drum module --------------- ##
# Padding zeros to the raw audio data
PAD_LEN_SEC = 1

# MIDI number of the lowest target note
LOWEST_NOTE = 16

# Total number of notes to be extracted
NUMBER_OF_NOTES = 120

# In sample numbers, different from `HOP` that is in seconds.
HOP_SIZE = 256

# Number of mini beats in a single 4/4 measure.
MINI_BEAT_DEVISION_NUMBER = 32
