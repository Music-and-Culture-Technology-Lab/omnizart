import click

from omnizart.cli.common_options import add_common_options, COMMON_TRANSCRIBE_OPTIONS
from omnizart.chord import app


@click.command()
@add_common_options(COMMON_TRANSCRIBE_OPTIONS)
def transcribe(input_audio, model_path, output):
    app.transcribe(input_audio, model_path=model_path, output=output)
