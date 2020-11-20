import click

from omnizart.cli.vocal_frame.generate_feature import generate_feature
from omnizart.cli.vocal_frame.train_model import train_model
from omnizart.cli.vocal_frame.transcribe import transcribe


@click.group()
def vocal_frame():
    """Transcribe vocal melody (frame-based) in the audio."""


vocal_frame.add_command(generate_feature)
vocal_frame.add_command(train_model)
vocal_frame.add_command(transcribe)
