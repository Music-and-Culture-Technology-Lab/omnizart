import os
from functools import partial

import click

from omnizart.music import transcribe as transcribe_music


click.option = partial(click.option, show_default=True)


@click.command()
@click.option(
    "-i",
    "--input-audio",
    help="Path to the target audio for transcriptioin",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "-m",
    "--model-path",
    help="Path to the pre-trained model for transcription",
    required=True,
    type=click.Path(exists=True),
)
@click.option("-o", "--output", help="Path to output the MIDI file", default="./", type=click.Path(writable=True))
def transcribe(input_audio, model_path, output):
    """Transcribe a single audio and output as a MIDI file.

    This will output a MIDI file with the same name as the given audio, except the
    extension will be replaced with '.mid'.

    \b
    Example Usage
    $ omnizart music transcribe \ 
        --input-audio example.wav \ 
        --model-path path/to/model \ 
        --output example.mid
    """
    transcribe_music(input_audio, model_path, output=output)
