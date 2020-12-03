import click

from omnizart.cli.vocal.generate_feature import generate_feature
from omnizart.cli.vocal.train_model import train_model
from omnizart.cli.vocal.transcribe import transcribe


@click.group()
def vocal():
    """Transcribe vocal notes in the audio."""


vocal.add_command(generate_feature)
vocal.add_command(train_model)
vocal.add_command(transcribe)
