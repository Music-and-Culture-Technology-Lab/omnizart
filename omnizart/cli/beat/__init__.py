import click

from omnizart.cli.beat.generate_feature import generate_feature
from omnizart.cli.beat.train_model import train_model
from omnizart.cli.beat.transcribe import transcribe


@click.group()
def beat():
    """Beat tracking on symbolic domain."""


beat.add_command(transcribe)
beat.add_command(generate_feature)
beat.add_command(train_model)
