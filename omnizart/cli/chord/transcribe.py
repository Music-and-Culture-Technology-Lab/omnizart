import click

from omnizart.cli import silence_tensorflow
from omnizart.cli.common_options import add_common_options, COMMON_TRANSCRIBE_OPTIONS
from omnizart.utils import LazyLoader


chord = LazyLoader("chord", globals(), "omnizart.chord")


@click.command()
@add_common_options(COMMON_TRANSCRIBE_OPTIONS)
def transcribe(input_audio, model_path, output):
    """Transcribe a single audio and output both MIDI and CSV file."""
    silence_tensorflow()
    chord.app.transcribe(input_audio, model_path=model_path, output=output)
