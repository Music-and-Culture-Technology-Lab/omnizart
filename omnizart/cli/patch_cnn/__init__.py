import click

from omnizart.cli.patch_cnn.generate_feature import generate_feature
from omnizart.cli.patch_cnn.train_model import train_model
from omnizart.cli.patch_cnn.transcribe import transcribe


@click.group()
def patch_cnn():
    """Trancribes vocal melody (frame-based) in the audio."""


patch_cnn.add_command(generate_feature)
patch_cnn.add_command(train_model)
patch_cnn.add_command(transcribe)
