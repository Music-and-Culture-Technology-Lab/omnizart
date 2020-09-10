import pretty_midi


def process(apps, **kwargs):
    mix_midi = pretty_midi.PrettyMIDI()
    for app in apps.values():
        midi = app.transcribe(output=None, **kwargs)
        mix_midi.instruments += midi.instruments
    return mix_midi
    
