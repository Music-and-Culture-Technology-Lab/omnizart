import click

from omnizart.cli.music.generate_feature import generate_feature
from omnizart.cli.music.train_model import train_model
from omnizart.cli.music.transcribe import transcribe


@click.group()
def music():
    """Transcribe instruments and corresponding pitch in the audio."""


music.add_command(generate_feature)
music.add_command(train_model)
music.add_command(transcribe)
