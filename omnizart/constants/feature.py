"""Some constant feature settings.

"""

## --------------- For Music module --------------- ##
#: Feature name to channel number mapping (for ``music`` module)
FEATURE_NAME_TO_NUMBER = {"S": 1, "G": 2, "C": 3, "Spec": 1, "GCoS": 2, "Ceps": 3}


## --------------- For Drum module --------------- ##
#: Weighting factor for different drum notes (for ``drum`` module).
NOTE_PRIORITY_ARRAY = [
    2.0364304, 1.4848346, 0.5027617, 0.5768271, 2.8335114, 0.733738,
    0.83764803, 0.5139924, 0.4998506, 0.4733462, 0.5940674, 0.7641602,
    1.148832
]


## --------------- For Chord module --------------- ##
#: Table of major chord to corresponding enharmonic chord (for ``chord`` module)
ENHARMONIC_TABLE = {'Cb': 'B', 'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}

#: Mapping of chord names to integers (for ``chord`` module).
CHORD_INT_MAPPING = {
    'C:maj': 0,
    'C#:maj': 1,
    'D:maj': 2,
    'D#:maj': 3,
    'E:maj': 4,
    'F:maj': 5,
    'F#:maj': 6,
    'G:maj': 7,
    'G#:maj': 8,
    'A:maj': 9,
    'A#:maj': 10,
    'B:maj': 11,
    'C:min': 12,
    'C#:min': 13,
    'D:min': 14,
    'D#:min': 15,
    'E:min': 16,
    'F:min': 17,
    'F#:min': 18,
    'G:min': 19,
    'G#:min': 20,
    'A:min': 21,
    'A#:min': 22,
    'B:min': 23,
    'N': 24,
    'X': 25
}
