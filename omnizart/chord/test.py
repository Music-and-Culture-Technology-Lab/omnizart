
import numpy as np


from omnizart.feature.chroma import extract_chroma
from omnizart.chord import app


def predict(audio_path):
    t_unit, chroma = extract_chroma(audio_path)
    model, settings = app._load_model()

    pad_size = settings.feature.segment_width // 2
    chroma_pad = np.pad(chroma, ((pad_size, pad_size), (0, 0)), constant_values=0)
    segments = np.array([
        chroma_pad[i-pad_size:i+pad_size+1] for i in range(pad_size, pad_size+len(chroma))  # noqa: E226
    ])
    segments = segments.reshape([-1, chroma.shape[1] * settings.feature.segment_width])

    num_steps = settings.feature.num_steps
    pad_end = num_steps - len(segments) % num_steps
    segments_pad = np.pad(segments, ((0, pad_end), (0, 0)), constant_values=0)

    num_seqs = len(segments_pad) // num_steps
    segments_pad = segments_pad.reshape([num_seqs, num_steps, segments_pad.shape[1]])

    chord, chord_change, _, _ = model.predict(segments_pad)
    chord = chord.reshape(np.prod(chord.shape))[:-pad_end]
    chord_change = chord_change.reshape(np.prod(chord_change.shape))[:-pad_end]
    return chord, chord_change



if __name__ == "__main__":
    audio = "/data/omnizart/checkpoints/collage.wav"
    chord, chord_change = predict(audio)