import click

from omnizart.cli.common_options import add_common_options, COMMON_TRANSCRIBE_OPTIONS
from omnizart.chord import app


@click.command()
@add_common_options(COMMON_TRANSCRIBE_OPTIONS)
def transcribe(input_audio, model_path, output):
    """Transcribe a single audio and output both MIDI and CSV file."""
    app.transcribe(input_audio, model_path=model_path, output=output)
