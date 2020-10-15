import click

from omnizart.cli.chord.generate_feature import generate_feature


@click.group()
def chord():
    """Transcribe chord progression"""


chord.add_command(generate_feature)
