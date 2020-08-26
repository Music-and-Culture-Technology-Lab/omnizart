"""Some constant feature settings."""

# Mapping feature name to channel number
FEATURE_NAME_TO_NUMBER = {"S": 1, "G": 2, "C": 3, "Spec": 1, "GCoS": 2, "Ceps": 3}

# Defines number of harmonic bins of HCFP feature
HARMONIC_NUM = 6

# Hop size in seconds
HOP = 0.02

# Window size of STFT
WINDOW_SIZE = 7939

# Frequency resolution
FREQUENCY_RESOLUTION = 2.0

# Lower bound of frequency
FREQUENCY_CENTER = 27.5

# Upper bound of frequency
TIME_CENTER = 1 / 4487.0

# Power factor in CFP approach
GAMMA = [0.24, 0.6, 1.0]

# Number of bins for each octave
BIN_PER_OCTAVE = 48

# Down sample the input audio to this sampling rate
DOWN_SAMPLE_TO_SAPMLING_RATE = 44100
