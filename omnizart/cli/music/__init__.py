import click

from .generate_feature import generate_feature
from .train_model import train_model
from .evaluate import evaluate
from .transcribe import transcribe


@click.group()
def music():
    """Transcribe instruments and corresponding pitch in the audio."""
    pass


music.add_command(generate_feature)
music.add_command(train_model)
music.add_command(evaluate)
music.add_command(transcribe)
