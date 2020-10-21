"""Stores MIDI-related constant variables.

"""
import os

from librosa import note_to_midi

from omnizart import MODULE_PATH


LOWEST_MIDI_NOTE = note_to_midi("A0")
HIGHEST_MIDI_NOTE = note_to_midi("C8")


#: Mappings of MIDI program number and the corresponding name derived from the
#: Wiki page: https://en.wikipedia.org/wiki/General_MIDI
MIDI_PROGRAM_NAME_MAPPING = {
    "0": "Acoustic Grand Piano",  # Piano
    "1": "Bright Acoustic Piano",
    "2": "Electric Grand Piano",
    "3": "Honky-tonk Piano",
    "4": "Electric Piano 1",
    "5": "Electric Piano 2",
    "6": "Harpsichord",
    "7": "Clavi",
    "8": "Celesta",  # Chromatic Percussion
    "9": "Glockenspiel",
    "10": "Music Box",
    "11": "Vibraphone",
    "12": "Marimba",
    "13": "Xylophone",
    "14": "Tubular Bells",
    "15": "Dulcimer",
    "16": "Drawbar Organ",  # Organ
    "17": "Percussive Organ",
    "18": "Rock Organ",
    "19": "Church Organ",
    "20": "Reed Organ",
    "21": "Accordion",
    "22": "Harmonica",
    "23": "Tango Accordion",
    "24": "Acoustic Guitar (nylon)",  # Guitar
    "25": "Acoustic Guitar (steel)",
    "26": "Electric Guitar (jazz)",
    "27": "Electric Guitar (clean)",
    "28": "Electric Guitar (muted)",
    "29": "Overdriven Guitar",
    "30": "Distortion Guitar",
    "31": "Guitar Harmonics",
    "32": "Acoustic Bass",  # Bass
    "33": "Electric Bass (finger)",
    "34": "Electric Bass (pick)",
    "35": "Retless Bass",
    "36": "Slap Bass 1",
    "37": "Slap Bass 2",
    "38": "Synth Bass 1",
    "39": "Synth Bass 2",
    "40": "Violin",  # Strings
    "41": "Viola",
    "42": "Cello",
    "43": "Contrabass",
    "44": "Tremolo Strings",
    "45": "Pizzicato Strings",
    "46": "Orchestral Harp",
    "47": "Timpani",
    "48": "String Ensemble 1",  # Ensemble
    "49": "String Ensemble 2",
    "50": "Synth Strings 1",
    "51": "Synth Strings 2",
    "52": "Choir Aahs",
    "53": "Voice Oohs",
    "54": "Synth Voice",
    "55": "Orchestra Hit",
    "56": "Trumpet",  # Brass
    "57": "Trombone",
    "58": "Tuba",
    "59": "Muted Trumpet",
    "60": "French Horn",
    "61": "Brass Section",
    "62": "Synth Brass 1",
    "63": "Synth Brass 2",
    "64": "Soprano Sax",  # Reed
    "65": "Alto Sax",
    "66": "Tenor Sax",
    "67": "Baritone Sax",
    "68": "Oboe",
    "69": "English Horn",
    "70": "Bassoon",
    "71": "Clarinet",
    "72": "Piccolo",  # Pipe
    "73": "Flute",
    "74": "Recorder",
    "75": "Pan Flute",
    "76": "Blown Bottle",
    "77": "Shakuhachi",
    "78": "Whistle",
    "79": "Ocarina",
    "80": "Lead 1 (square)",
    "81": "Lead 2 (sawtooth)",
    "82": "Lead 3 (calliope)",
    "83": "Lead 4 (chiff)",
    "84": "Lead 5 (charang)",
    "85": "Lead 6 (voice)",
    "86": "Lead 7 (fifths)",
    "87": "Lead 8 (bass+lead)",
    "88": "Pad 1 (new age)",  # Synth Pad
    "89": "Pad 2 (warm)",
    "90": "Pad 3 (polysynth)",
    "91": "Pad 4 (choir)",
    "92": "Pad 5 (bowed)",
    "93": "Pad 6 (metallic)",
    "94": "Pad 7 (halo)",
    "95": "Pad 8 (sweep)",
    "96": "FX 1 (rain)",  # Synth Effects
    "97": "FX 2 (soundtrack)",
    "98": "FX 3 (crystal)",
    "99": "FX 4 (atmosphere)",
    "100": "FX 5 (brightness)",
    "101": "FX 6 (goblins)",
    "102": "FX 7 (echoes)",
    "103": "FX 8 (sci-fi)",
    "104": "Sitar",  # Ethinic
    "105": "Banjo",
    "106": "Shamisen",
    "107": "Koto",
    "108": "Kalimba",
    "109": "Bag pipe",
    "110": "Fiddle",
    "111": "Shanai",
    "112": "Tinkle Bell",  # Percussive
    "113": "Agogo",
    "114": "Steel Drums",
    "115": "Woodblock",
    "116": "Taiko Drum",
    "117": "Melodic Tom",
    "118": "Synth Drum",
    "119": "Reverse Cymbal",
    "120": "Guitar Fret Noise",  # Sound Effects
    "121": "Breath Noise",
    "122": "Seashore",
    "123": "Bird Tweet",
    "124": "Telephone Ring",
    "125": "Helicopter",
    "126": "Applause",
    "127": "Gunshot",
}

#: Program numbers that are used in MusicNet dataset.
MUSICNET_INSTRUMENT_PROGRAMS = [0, 6, 40, 41, 42, 43, 60, 68, 70, 71, 73]

#: Program numbers that represent different groups of channels used in Pop dataset.
#: Guitar, bass, strings, organ, piano, and others
POP_INSTRUMENT_PROGRAMES = [24, 32, 40, 0, 56]

#: Path to the soundfont.
SOUNDFONT_PATH = os.path.join(MODULE_PATH, "resource/soundfonts.sf2")
