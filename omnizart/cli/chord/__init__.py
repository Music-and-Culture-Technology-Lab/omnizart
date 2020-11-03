import click

from omnizart.cli.chord.transcribe import transcribe
from omnizart.cli.chord.generate_feature import generate_feature
from omnizart.cli.chord.train_model import train_model


@click.group()
def chord():
    """Transcribe chord progression"""


chord.add_command(transcribe)
chord.add_command(generate_feature)
chord.add_command(train_model)
