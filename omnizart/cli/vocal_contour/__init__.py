import click

from omnizart.cli.vocal_contour.generate_feature import generate_feature
from omnizart.cli.vocal_contour.train_model import train_model
from omnizart.cli.vocal_contour.transcribe import transcribe


@click.group()
def vocal_contour():
    """Transcribe vocal melody (frame-based) in the audio."""


vocal_contour.add_command(generate_feature)
vocal_contour.add_command(train_model)
vocal_contour.add_command(transcribe)
